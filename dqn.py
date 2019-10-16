import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from gridworld import GridWorld
from collections import deque as ReplayMemory
import tqdm


def buildMLP(*sizes):
    return nn.Sequential(*chain.from_iterable(
        (nn.Linear(s_in, s_out), nn.ReLU()) for s_in, s_out in zip(sizes[:-2], sizes[1:-1])),
        nn.Linear(sizes[-2], sizes[-1]))


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


def epsilon_greedy(z, q_net, epsilon=0):
    dist = q_net(z.float().unsqueeze(0)).softmax(1)
    n_a = dist.size(1)
    if torch.rand((1,)) < epsilon:
        a = torch.randint(0, n_a, (1,1))
    else:
        a = dist.argmax(1, keepdim=True)
    p = epsilon / n_a + (1 - epsilon) * dist.gather(1, a)
    return a.item(), p.item()


def make_batches(memory, batch_size, max_batch_count, dev):
    n = len(memory)
    if n < batch_size:
        raise StopIteration()
    indices = torch.randperm(n)
    for b in range(min(max_batch_count, n / batch_size)):
        batch = (memory[i] for i in indices[batch_size * b:batch_size * (b + 1)])
        batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p = zip(*memory)

        batch_z = torch.stack(batch_z).float().to(dev)
        batch_a = torch.tensor(batch_a, dtype=torch.long).unsqueeze(1).to(dev)
        batch_r = torch.tensor(batch_r, dtype=torch.float).unsqueeze(1).to(dev)
        batch_nxt = torch.stack(batch_nxt).float().to(dev)
        batch_done = torch.tensor(batch_done, dtype=torch.bool).unsqueeze(1).to(dev)
        batch_p = torch.tensor(batch_p, dtype=torch.float).unsqueeze(1).to(dev)
        yield batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p


if __name__ == "__main__":
    HIDDEN_DIMS = (256, 128)
    USE_CUDA = False
    MAX_EPOCHS = 1000
    BASE_LR = 0.0005
    BASE_EPSILON = 0.9
    MIN_EPSILON = 0.05
    MEM_SIZE = 10000
    MAX_T = 200
    BATCH_SIZE = 32
    STEP_SAMPLE_COUNT = 320
    MAX_BATCH_COUNT = 10
    DISCOUNT = 0.99
    AVG_RATE = 0.05
    FREEZE_PERIOD = 50

    env = GridWorld(9, 9)
    env.add_horizontal_wall(5, 1, 9)
    env.add_clear_surface(4, 5, 6, 5)
    env.add_start(1, 1)
    env.add_start(9, 1)
    env.add_goal(9, 9)

    dev = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    q_net = buildMLP(2, *HIDDEN_DIMS, 4)
    target_net = buildMLP(2, *HIDDEN_DIMS, 4) 
    target_net.load_state_dict(q_net.state_dict())
    q_net.to(dev)
    target_net.to(dev)

    optim = torch.optim.SGD(q_net.parameters(), lr=BASE_LR)
    epsilon = BASE_EPSILON

    memory = ReplayMemory(maxlen=MEM_SIZE)

    avg_cumul = None
    avg_loss = None
    avg_success = None
    stats = []
    try:
        with tqdm.trange(MAX_EPOCHS) as progress:
            for ep in progress:
                new_samples = 0
                while new_samples < STEP_SAMPLE_COUNT:
                    trajectory, cumul, success = sample_trajectory(env, lambda z: epsilon_greedy(z, q_net, epsilon), MAX_T)
                    memory.extend(trajectory)
                    new_samples += len(trajectory)
                    if avg_cumul is None:
                        avg_cumul = cumul
                        avg_success = int(success)
                    else:
                        avg_cumul = (1 - AVG_RATE) * avg_cumul + AVG_RATE * cumul
                        avg_success = (1 - AVG_RATE) * avg_success + AVG_RATE * int(success)

                tot_loss = 0
                batch_count = 0
                for batch_z, batch_a, batch_r, batch_nxt, batch_done, batch_p in make_batches(memory, BATCH_SIZE, MAX_BATCH_COUNT, dev):
                    nxt_val = target_net(batch_nxt).max(1, keepdim=True)[0]
                    nxt_val.masked_fill_(batch_done, 0)
                    target = batch_r + DISCOUNT * nxt_val

                    qval = q_net(batch_z).gather(1, batch_a)

                    loss = F.mse_loss(qval, target.detach())
                    tot_loss += loss.item()
                    batch_count += 1

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                if batch_count > 0:
                    if avg_loss is None:
                        avg_loss = tot_loss / batch_count
                    else:
                        avg_loss = (1 - AVG_RATE) * avg_loss + AVG_RATE * tot_loss / batch_count
                    stats.append((it, avg_loss, avg_cumul, avg_success, epsilon, optim.param_groups[0]["lr"]))
                    progress.set_postfix(loss=avg_loss, cumul=avg_cumul, success=avg_success, epsilon=epsilon)

                if ep % FREEZE_PERIOD == FREEZE_PERIOD - 1:
                    temp = target_net.state_dict()
                    target_net.load_state_dict(q_net.state_dict())
                    q_net.load_state_dict(temp)

                epsilon = (1 - ep / MAX_ITER) * (BASE_EPSILON - MIN_EPSILON) + MIN_EPSILON
    except KeyboardInterrupt:
        pass
    torch.save(q_net.state_dict(), "trained_mlp_gridworld_{}.pkl".format(ep))
    with open("training_stats.csv", 'w') as f:
        for ep_stat in stats:
            f.write(', '.join(str(s) for s in ep_stat))
            f.write('\n')
