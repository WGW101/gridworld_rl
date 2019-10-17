import torch
import tqdm
from gridworld import GridWorld


def epsilon_greedy(z, q, eps):
    if torch.rand((1,)) < eps:
        return torch.randint(0, 4, (1,)).item()
    else:
        return q[z[0]-1, z[1]-1].argmax().item()


if __name__ == "__main__":
    MAX_ITER = 10000

    env = GridWorld(9, 9)
    env.add_horizontal_wall(5, 1, 9)
    env.add_clear_surface(4, 5, 6, 5)
    env.add_start(1, 1)
    env.add_start(9, 1)
    env.add_goal(9, 9)

    q = torch.zeros((9, 9, 4))

    eps = 0.95
    lr = 0.05
    discount = 0.99

    avg_cumul = None
    avg_err = None
    avg_success = None

    with tqdm.trange(MAX_ITER) as progress:
        for it in progress:
            z = env.reset()
            cumul = 0
            mean_err = 0
            for t in range(200):
                a = epsilon_greedy(z, q, eps)
                nxt, r, done = env.step(a)
                cumul += r

                if done:
                    nxt_val = 0
                else:
                    nxt_val = q[nxt[0]-1, nxt[1]-1].max().item()
                target = r + discount * nxt_val
                td_err = target - q[z[0]-1, z[1]-1, a].item()
                q[z[0]-1, z[1]-1, a] += lr * td_err

                mean_err += td_err

                if done:
                    break
                z = nxt
            avg_cumul = cumul if avg_cumul is None else 0.98 * avg_cumul + 0.02 * cumul
            avg_err = mean_err / t if avg_err is None else 0.98 * avg_err + 0.02 * mean_err / t
            avg_success = int(done) if avg_success is None else 0.98 * avg_success + 0.02 * int(done)
            progress.set_postfix(td_err=avg_err, eps=eps, cumul=avg_cumul, success=avg_success)
            env.terminate()
            eps = 0.95 * (1 - it / MAX_ITER)

    print(env)
    val, pi = q.max(2)
    print('\n'.join(' '.join("{:6.3f}".format(v) for v in val_row) for val_row in val))
    print('\n'.join(' '.join({GridWorld.Direction.NORTH:'^',
        GridWorld.Direction.WEST:'<',
        GridWorld.Direction.SOUTH:'v',
        GridWorld.Direction.EAST:'>'}.get(a.item()) for a in pi_row) for pi_row in pi))
