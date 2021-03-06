#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from gridworld import GridWorld
from collections import deque as ReplayMemory
import tqdm
import os
import time
import math
from argparse import ArgumentParser
import json


def parse_args():
    HIDDEN_DIMS = (256,)
    USE_CUDA = False
    MAX_ITER = 10000
    BASE_LR = 0.01
    LR_STEP = 100
    LR_DECAY = None
    BASE_EPSILON = 0.95
    MIN_EPSILON = 0.05
    EPS_STEP = 100
    EPS_DECAY = None
    MEM_SIZE = 1000
    MAX_T = 200
    BATCH_SIZE = 64
    BATCH_COUNT = 100
    DISCOUNT = 0.99
    FREEZE_PERIOD = 100

    parser = ArgumentParser()
    parser.add_argument("env")
    parser.add_argument("--hidden-dims", "-d", type=int, nargs='+', default=HIDDEN_DIMS)
    parser.add_argument("--use-cuda", action="store_true", default=USE_CUDA)
    parser.add_argument("--max-iter", "-n", type=int, default=MAX_ITER)
    parser.add_argument("--base-lr", "-r", type=float, default=BASE_LR)
    parser.add_argument("--lr-step", type=int, default=LR_STEP)
    parser.add_argument("--lr-decay", type=float, default=LR_DECAY)
    parser.add_argument("--base-epsilon", type=float, default=BASE_EPSILON)
    parser.add_argument("--min-epsilon", type=float, default=MIN_EPSILON)
    parser.add_argument("--eps-step", type=int, default=EPS_STEP)
    parser.add_argument("--eps-decay", type=float, default=EPS_DECAY)
    parser.add_argument("--mem-size", "-m", type=int, default=MEM_SIZE)
    parser.add_argument("--max-t", type=int, default=MAX_T)
    parser.add_argument("--batch-size", "-b", type=int, default=BATCH_SIZE)
    parser.add_argument("--batch-count", "-c", type=int, default=BATCH_COUNT)
    parser.add_argument("--discount", "-g", type=float, default=DISCOUNT)
    parser.add_argument("--freeze-period", "-t", type=int, default=FREEZE_PERIOD)
    parser.add_argument("--output-dir", "-o", default=os.path.join("output", time.strftime("%y%m%d_%H%M%S")))

    return parser.parse_args()


def build_MLP(*sizes):
    return nn.Sequential(*chain.from_iterable(
        (nn.Linear(s_in, s_out), nn.ReLU()) for s_in, s_out in zip(sizes[:-2], sizes[1:-1])),
        nn.Linear(sizes[-2], sizes[-1]))

def normalize_state(z, min_x=1, max_x=9):
    return (z - min_x) / (max_x - min_x)


def epsilon_greedy(z, q_net, epsilon=0):
    dist = q_net(normalize_state(z.float()).unsqueeze(0)).softmax(1)
    n_a = dist.size(1)
    if torch.rand((1,)) < epsilon:
        a = torch.randint(0, n_a, (1, 1))
    else:
        a = dist.argmax(1, keepdim=True)
    p = epsilon / n_a + (1 - epsilon) * dist.gather(1, a)
    return a.item(), p.item()


def sample_trajectory(env, policy, max_t=None):
    cumul = 0
    trajectory = []
    done = False
    with torch.no_grad():
        z = env.reset()
        while not done:
            a, p = policy(z)
            nxt, r, done = env.step(a)
            cumul += r
            trajectory.append((z, a, r, nxt, done, p))
            if len(trajectory) == max_t:
                break
            z = nxt
        env.terminate()
    return trajectory, cumul, int(done)


def sample_batch(memory, batch_size, batch_count, dev):
    n = len(memory)
    indices = torch.randperm(n)
    for b in range(min(batch_count, math.ceil(n / batch_size))):
        batch = (memory[i] for i in indices[b * batch_size:(b + 1) * batch_size])
        batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p = zip(*batch)

        batch_z = normalize_state(torch.stack(batch_z).float()).to(dev)
        batch_a = torch.tensor(batch_a, dtype=torch.long).unsqueeze(1).to(dev)
        batch_r = torch.tensor(batch_r, dtype=torch.float).unsqueeze(1).to(dev)
        batch_nxt = normalize_state(torch.stack(batch_nxt).float()).to(dev)
        batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(1).to(dev)
        batch_p = torch.tensor(batch_p, dtype=torch.float).unsqueeze(1).to(dev)
        yield batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p


def update_weights(q_net, target_net, optim, batch_z, batch_a, batch_r, batch_nxt, batch_done, discount):
    nxt_val = target_net(batch_nxt).max(1, keepdim=True)[0]
    nxt_val.masked_fill_(batch_done, 0)
    target = batch_r + discount * nxt_val
    qval = q_net(batch_z).gather(1, batch_a)
    loss = F.mse_loss(qval, target.detach())
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def main(args):
    env = GridWorld.load(args.env)

    dev = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    q_net = build_MLP(2, *args.hidden_dims, 4)
    target_net = build_MLP(2, *args.hidden_dims, 4) 
    target_net.load_state_dict(q_net.state_dict())
    q_net.to(dev)
    target_net.to(dev)

    optim = torch.optim.SGD(q_net.parameters(), lr=args.base_lr)
    if args.lr_decay is not None:
        lr_sched = torch.optim.lr_scheduler.StepLR(optim, args.lr_step, args.lr_decay)
    epsilon = args.base_epsilon

    memory = ReplayMemory(maxlen=args.mem_size)

    avg_cumul = None
    avg_success = None
    avg_loss = None
    AVG_R = 0.05
    stats = []
    try:
        with tqdm.trange(args.max_iter) as progress:
            for it in progress:
                trajectory, cumul, success = sample_trajectory(env,
                        lambda z: epsilon_greedy(z, q_net, epsilon), args.max_t)
                memory.extend(trajectory)

                loss = 0
                for b, batch in enumerate(sample_batch(memory, args.batch_size, args.batch_count, dev)):
                    loss += update_weights(q_net, target_net, optim, *batch[:-1], args.discount)
                if b > 0:
                    loss /= b

                avg_cumul = cumul if avg_cumul is None else (1 - AVG_R) * avg_cumul + AVG_R * cumul
                avg_success = success if avg_success is None else (1 - AVG_R) * avg_success + AVG_R * success
                avg_loss = loss if avg_loss is None else (1 - AVG_R) * avg_loss + AVG_R * loss
                lr = optim.param_groups[0]["lr"]
                progress.set_postfix(cumul=avg_cumul, success=avg_success, loss=avg_loss,
                        lr=lr, eps=epsilon)
                stats.append((it, avg_cumul, avg_success, avg_loss, lr, epsilon))

                if it % args.freeze_period == args.freeze_period - 1:
                    target_net.load_state_dict(q_net.state_dict())
                if args.lr_decay is not None:
                    lr_sched.step()
                if args.eps_decay is None:
                    epsilon = (args.base_epsilon - args.min_epsilon) * (1 - it / args.max_iter) + args.min_epsilon
                elif it % args.eps_step == args.eps_step - 1:
                    epsilon = max(epsilon * args.eps_decay, args.min_epsilon)
    except KeyboardInterrupt:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    torch.save(q_net.state_dict(), os.path.join(args.output_dir, "trained_mlp_{}.pkl".format(it)))
    with open(os.path.join(args.output_dir, "training_stats.csv"), 'w') as f:
        for it_stat in stats:
            f.write(', '.join(str(s) for s in it_stat))
            f.write('\n')


if __name__ == "__main__":
    main(parse_args())
