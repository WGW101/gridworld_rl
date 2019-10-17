import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from gridworld import GridWorld
from collections import deque as ReplayMemory
import tqdm
import os
import time
from argparse import ArgumentParser
import json


def parse_args():
    HIDDEN_DIMS = (256, 128)
    USE_CUDA = False
    MAX_EPOCH = 1000
    BASE_LR = 0.001
    LR_STEP = 10
    LR_DECAY = None
    BASE_EPSILON = 0.9
    EPS_STEP = 10
    EPS_DECAY = None
    MEM_SIZE = 1000
    MAX_T = 200
    BATCH_SIZE = 32
    STEP_SAMPLE_COUNT = 320
    MAX_BATCH_COUNT = 12
    DISCOUNT = 0.99
    AVG_RATE = 0.05
    FREEZE_PERIOD = 50

    parser = ArgumentParser()
    parser.add_argument("--hidden-dims", type=int, default=HIDDEN_DIMS)
    parser.add_argument("--use-cuda", action="store_true", default=USE_CUDA)
    parser.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    parser.add_argument("--base-lr", type=float, default=BASE_LR)
    parser.add_argument("--lr-step", type=int, default=LR_STEP)
    parser.add_argument("--lr-decay", type=float, default=LR_DECAY)
    parser.add_argument("--base-epsilon", type=float, default=BASE_EPSILON)
    parser.add_argument("--eps-step", type=int, default=EPS_STEP)
    parser.add_argument("--eps-decay", type=float, default=EPS_DECAY)
    parser.add_argument("--mem-size", type=int, default=MEM_SIZE)
    parser.add_argument("--max-t", type=int, default=MAX_T)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--step-sample-count", type=int, default=STEP_SAMPLE_COUNT)
    parser.add_argument("--max-batch-count", type=int, default=MAX_BATCH_COUNT)
    parser.add_argument("--discount", type=float, default=DISCOUNT)
    parser.add_argument("--avg-rate", type=float, default=AVG_RATE)
    parser.add_argument("--freeze-period", type=int, default=FREEZE_PERIOD)
    parser.add_argument("--output-dir", default=time.strftime("%y%m%d_%H%M%S"))

    return parser.parse_args()


def build_env():
    env = GridWorld(9, 9)
    env.add_horizontal_wall(5, 1, 9)
    env.add_clear_surface(4, 5, 6, 5)
    env.add_start(1, 1)
    env.add_start(9, 1)
    env.add_goal(9, 9)
    return env


def build_MLP(*sizes):
    return nn.Sequential(*chain.from_iterable(
        (nn.Linear(s_in, s_out), nn.ReLU()) for s_in, s_out in zip(sizes[:-2], sizes[1:-1])),
        nn.Linear(sizes[-2], sizes[-1]))


def epsilon_greedy(z, q_net, epsilon=0):
    dist = q_net(z.float().unsqueeze(0)).softmax(1)
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
    return trajectory, cumul, done


def make_batches(memory, batch_size, max_batch_count, dev):
    n = len(memory)
    if n < batch_size:
        raise StopIteration()
    indices = torch.randperm(n)
    for b in range(min(max_batch_count, n // batch_size)):
        batch = (memory[i] for i in indices[batch_size * b:batch_size * (b + 1)])
        batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p = zip(*memory)

        batch_z = torch.stack(batch_z).float().to(dev)
        batch_a = torch.tensor(batch_a, dtype=torch.long).unsqueeze(1).to(dev)
        batch_r = torch.tensor(batch_r, dtype=torch.float).unsqueeze(1).to(dev)
        batch_nxt = torch.stack(batch_nxt).float().to(dev)
        batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(1).to(dev)
        batch_p = torch.tensor(batch_p, dtype=torch.float).unsqueeze(1).to(dev)
        yield batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p


def update_params(q_net, target_net, optim, batch_z, batch_a, batch_r, batch_nxt, batch_done, discount):
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
    env = build_env()
    print(env)

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
    avg_loss = None
    avg_success = None
    stats = []
    try:
        with tqdm.trange(args.max_epoch) as progress:
            for ep in progress:
                new_sample_count = 0
                while new_sample_count < args.step_sample_count:
                    trajectory, cumul, success = sample_trajectory(
                            env, lambda z: epsilon_greedy(z, q_net, epsilon), args.max_t)
                    memory.extend(trajectory)
                    new_sample_count += len(trajectory)
                    if avg_cumul is None:
                        avg_cumul = cumul
                        avg_success = int(success)
                    else:
                        avg_cumul = (1 - args.avg_rate) * avg_cumul + args.avg_rate * cumul
                        avg_success = (1 - args.avg_rate) * avg_success + args.avg_rate * int(success)

                tot_loss = 0
                batch_count = 0
                for batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p in make_batches(
                        memory, args.batch_size, args.max_batch_count, dev):
                    tot_loss += update_params(q_net, target_net, optim,
                            batch_z, batch_a, batch_r, batch_nxt, batch_done, args.discount)
                    batch_count += 1
                if batch_count > 0:
                    if avg_loss is None:
                        avg_loss = tot_loss / batch_count
                    else:
                        avg_loss = (1 - args.avg_rate) * avg_loss + args.avg_rate * tot_loss / batch_count
                    stats.append((ep, avg_loss, avg_cumul, avg_success, epsilon, optim.param_groups[0]["lr"]))
                    progress.set_postfix(loss=avg_loss, cumul=avg_cumul, success=avg_success,
                            epsilon=epsilon, lr=optim.param_groups[0]["lr"])

                if ep % args.freeze_period == args.freeze_period - 1:
                    target_net.load_state_dict(q_net.state_dict())
                if args.lr_decay is not None:
                    lr_sched.step()
                if args.eps_decay is None:
                    epsilon = args.base_epsilon * (1 - ep / args.max_epoch)
                elif ep % args.eps_step == args.eps_step - 1:
                    epsilon *= args.eps_decay
    except KeyboardInterrupt:
        pass

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
    torch.save(q_net.state_dict(), os.path.join(args.output_dir, "trained_mlp_gridworld_{}.pkl".format(ep)))
    with open(os.path.join(args.output_dir, "training_stats.csv"), 'w') as f:
        for ep_stat in stats:
            f.write(', '.join(str(s) for s in ep_stat))
            f.write('\n')


if __name__ == "__main__":
    main(parse_args())
