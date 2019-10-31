#!/usr/bin/env python3

from gridworld import GridWorld
import sys

if __name__ == "__main__":
    g = GridWorld.load(sys.argv[1])

    z = g.reset()
    done = False
    trajectory = []
    cumul = 0
    while not done:
        print(g)
        a = {'w': GridWorld.Direction.NORTH,
                'a': GridWorld.Direction.WEST,
                's': GridWorld.Direction.SOUTH,
                'd': GridWorld.Direction.EAST}.get(input("z = {} > ".format(z)))
        nxt, r, done = g.step(a)
        cumul += r
        trajectory.append((z, a, r, nxt, done))
        z = nxt
    print("Cumul = {}".format(cumul))
    print('\n'.join(str(step) for step in trajectory))
