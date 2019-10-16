from gridworld import GridWorld

if __name__ == "__main__":
    g = GridWorld(9, 9)
    g.add_horizontal_wall(5, 1, 9)
    g.add_clear_surface(4, 5, 6, 5)
    g.add_start(1, 1)
    g.add_start(9, 1)
    g.add_goal(9, 9)

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
