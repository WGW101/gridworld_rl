#!/usr/bin/env python3

import os
import time
import random
import torch
import tqdm
from argparse import ArgumentParser
from collections import deque
from gridworld import GridWorld


def parse_args():
    COUNT = 100
    MAX_SIZE = 25
    MIN_SIZE = 13
    MAX_WALL_COUNT = 8
    MAX_START_COUNT = 3
    MAX_GOAL_COUNT = 3
    MAX_TRAP_COUNT = 2

    parser = ArgumentParser()
    parser.add_argument("--count", "-c", type=int, default=COUNT)
    parser.add_argument("--max-size", "-M", type=int, default=MAX_SIZE)
    parser.add_argument("--min-size", "-m", type=int, default=MIN_SIZE)
    parser.add_argument("--wall-count", "-w", type=int, default=MAX_WALL_COUNT)
    parser.add_argument("--start-count", "-s", type=int, default=MAX_START_COUNT)
    parser.add_argument("--goal-count", "-g", type=int, default=MAX_GOAL_COUNT)
    parser.add_argument("--trap-count", "-t", type=int, default=MAX_TRAP_COUNT)
    parser.add_argument("--output-dir", "-o", default=os.path.join("data", time.strftime("%y%m%d_%H%M%S")))

    return parser.parse_args()


def check_connectivity(grid):
    connected_components = []

    def neighbours(x, y):
        return (x, y - 1), (x - 1, y), (x, y + 1), (x + 1, y)

    empty_mask = grid._initial_tiles != GridWorld.TileType.WALL
    to_visit = {(pos[1].item(), pos[0].item()) for pos in torch.nonzero(empty_mask)}

    while to_visit:
        start = to_visit.pop()
        visited = {start}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for nx, ny in neighbours(x, y):
                if (nx, ny) not in visited and empty_mask[ny, nx]:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        to_visit -= visited
        connected_components.append(visited)

    return max(connected_components, key=lambda comp: len(comp))


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    for k in tqdm.trange(args.count):
        g = GridWorld(args.max_size, args.max_size)
        size = 2 * random.randint(args.min_size // 2, args.max_size // 2) + 1
        if size < args.max_size:
            pad = (args.max_size - size) // 2
            g._fill_rect(1, 1, pad, args.max_size)
            g._fill_rect(args.max_size - pad + 1, 1, args.max_size, args.max_size)
            g._fill_rect(1, 1, args.max_size, pad)
            g._fill_rect(1, args.max_size - pad + 1, args.max_size, args.max_size)
        else:
            pad = 0

        wall_count = random.randint(1, args.wall_count)
        for _ in range(wall_count):
            is_vert = random.random() > 0.5
            wall_coord = random.randint(2, size - 1)
            wall_len = random.randint(2, size - 2)
            wall_start = random.randint(1, size - wall_len)
            if is_vert:
                g.add_vertical_wall(pad + wall_coord, pad + wall_start, pad + wall_start + wall_len - 1)
            else:
                g.add_horizontal_wall(pad + wall_coord, pad + wall_start, pad + wall_start + wall_len - 1)

        connected_component = list(check_connectivity(g))
        random.shuffle(connected_component)

        start_count = random.randint(1, args.start_count)
        for _ in range(start_count):
            g.add_start(*connected_component.pop())

        goal_count = random.randint(1, args.goal_count)
        for _ in range(goal_count):
            g.add_goal(*connected_component.pop())

        trap_count = random.randint(0, args.trap_count)
        for _ in range(trap_count):
            g.add_trap(*connected_component.pop())

        g.save(os.path.join(args.output_dir, "grid{0:03d}_{1}x{1}_w{2}_s{3}_g{4}_t{5}.pkl".format(
            k, size, wall_count, start_count, goal_count, trap_count)))


if __name__ == "__main__":
    main(parse_args())
