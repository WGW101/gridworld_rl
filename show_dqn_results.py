from gridworld import GridWorld
from train_dqn import build_MLP
import sys
import os.path
import torch
import numpy
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec


if __name__ == "__main__":
    all_z = torch.stack((torch.arange(1, 10).expand(9, 9).t(), torch.arange(1, 10).expand(9, 9)), 2)
    all_z = (all_z.float().view(81, 2) - 1) / 8
    
    q_net = build_MLP(2, 256, 4)
    q_net.load_state_dict(torch.load(sys.argv[1]))
    q_net.eval()

    with torch.no_grad():
        val, pi = q_net(all_z).view(9, 9, 4).max(2)
    print('\n'.join(' '.join("{:6.3f}".format(v) for v in val_row) for val_row in val))
    print('\n'.join(' '.join({
        GridWorld.Direction.NORTH:'^',
        GridWorld.Direction.WEST:'<',
        GridWorld.Direction.SOUTH:'v',
        GridWorld.Direction.EAST:'>'}.get(a.item()) for a in pi_row) for pi_row in pi))
    
    stats_file = os.path.join(os.path.dirname(sys.argv[1]), "training_stats.csv")
    IT, CUMUL, SUCCESS, LOSS, LR, EPS = range(6)
    data = numpy.loadtxt(stats_file, delimiter=',').T

    fig = pyplot.figure(constrained_layout=True)
    gs = GridSpec(4, 4, fig)

    ax = fig.add_subplot(gs[:, :3])
    ax.plot(data[IT], data[CUMUL])
    ax.set_title("Cumulated reward")

    ax = fig.add_subplot(gs[0, 3])
    ax.plot(data[IT], data[SUCCESS])
    ax.set_title("Success rate")

    ax = fig.add_subplot(gs[1, 3])
    ax.plot(data[IT], data[LOSS])
    ax.set_title("TD MSE")

    ax = fig.add_subplot(gs[2, 3])
    ax.plot(data[IT], data[LR])
    ax.set_title("LR")

    ax = fig.add_subplot(gs[3, 3])
    ax.plot(data[IT], data[EPS])
    ax.set_title(u"ε-explor")

    pyplot.show()
