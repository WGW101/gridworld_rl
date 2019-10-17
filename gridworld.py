import torch


class GridWorld:
    class TileType:
        CLEAR = 255
        DIRTY = 250
        GOAL = 200
        TRAP = 190
        START = 100
        AGENT = 1
        WALL = 0

    class Direction:
        NORTH = 0
        WEST = 1
        SOUTH = 2
        EAST = 3

    def __init__(self, width, height, default_tile=TileType.CLEAR,
                 drift_probability=0, clean_probability=1,
                 move_cost=1, bump_cost=0, trap_cost=10):
        self.width = width
        self.height = height
        self._initial_tiles = torch.full((height + 2, width + 2), default_tile, dtype=torch.long)
        self._initial_tiles[0, :] = GridWorld.TileType.WALL
        self._initial_tiles[-1, :] = GridWorld.TileType.WALL
        self._initial_tiles[:, 0] = GridWorld.TileType.WALL
        self._initial_tiles[:, -1] = GridWorld.TileType.WALL

        self.drift_probability = drift_probability
        self.clean_probability = clean_probability
        self.move_cost = move_cost
        self.bump_cost = bump_cost
        self.trap_cost = trap_cost

        self.terminate()

    def save(self, file_name):
        torch.save({"width": self.width, "height": self.height, "tiles": self._initial_tiles,
            "drift_probability": self.drift_probability, "clean_probability": self.clean_probability,
            "move_cost": self.move_cost, "bump_cost": self.bump_cost, "trap_cost": self.trap_cost}, file_name)

    @classmethod
    def load(cls, file_name):
        saved = torch.load(file_name)
        tiles = saved.pop("tiles", None)
        inst = cls(**saved)
        if tiles is not None:
            inst._initial_tiles = tiles
        return inst

    def __str__(self):
        tiles = self._initial_tiles if self._tiles is None else self._tiles
        return '\n'.join(' '.join("{}" for _ in range(self.width+2)) for _ in range(self.height+2)).format(
            *({GridWorld.TileType.CLEAR: " ",
               GridWorld.TileType.DIRTY: ".",
               GridWorld.TileType.GOAL: "$",
               GridWorld.TileType.TRAP: "%",
               GridWorld.TileType.START: "o",
               GridWorld.TileType.AGENT: "+",
               GridWorld.TileType.WALL: "x"}.get(t.item()) for t in tiles.view(-1))
        )

    def _check_coordinates(self, x, y):
        if not (1 <= x <= self.width and 1 <= y <= self.height):
            raise IndexError("Coordinates out of bound: ({}, {})".format(x, y))

    def _fill_rect(self, from_x, from_y, to_x, to_y, tile_type=TileType.WALL):
        self._check_coordinates(from_x, from_y)
        self._check_coordinates(to_x, to_y)
        self._initial_tiles[from_y:to_y + 1, from_x:to_x + 1] = tile_type

    def add_horizontal_wall(self, at_y, from_x, to_x):
        self._fill_rect(from_x, at_y, to_x, at_y)

    def add_vertical_wall(self, at_x, from_y, to_y):
        self._fill_rect(at_x, from_y, at_x, to_y)

    def add_room(self, from_x, from_y, to_x, to_y):
        self.add_vertical_wall(from_x, from_y, to_y)
        self.add_vertical_wall(to_x, from_y, to_y)
        self.add_horizontal_wall(from_y, from_x, to_x)
        self.add_horizontal_wall(to_y, from_x, to_x)

    def add_start(self, at_x, at_y):
        self._fill_rect(at_x, at_y, at_x, at_y, GridWorld.TileType.START)

    def add_goal(self, at_x, at_y):
        self._fill_rect(at_x, at_y, at_x, at_y, GridWorld.TileType.GOAL)

    def add_trap(self, at_x, at_y):
        self._fill_rect(at_x, at_y, at_x, at_y, GridWorld.TileType.TRAP)

    def add_clear_surface(self, from_x, from_y, to_x, to_y):
        self._fill_rect(from_x, from_y, to_x, to_y, GridWorld.TileType.CLEAR)

    def add_dirty_surface(self, from_x, from_y, to_x, to_y):
        self._fill_rect(from_x, from_y, to_x, to_y, GridWorld.TileType.DIRTY)

    def terminate(self):
        self._tiles = None
        self._has_goal = False
        self._agent_pos = None
        self._agent_over = None

    def reset(self):
        self._tiles = self._initial_tiles.clone()
        start_mask = self._tiles == GridWorld.TileType.START
        if not start_mask.any():
            raise RuntimeError("Cannot reset episode on a GridWorld that has no starting position")
        dirty_mask = self._tiles == GridWorld.TileType.DIRTY
        self._has_goal = (self._tiles == GridWorld.TileType.GOAL).any()
        if not dirty_mask.any() and not self._has_goal:
            raise RuntimeError("Cannot reset episode on a GridWorld that has no terminal condition "
                               + "(goal or dirty tiles)")
        self._tiles[start_mask] = GridWorld.TileType.CLEAR
        start_pos = torch.nonzero(start_mask)
        self._agent_pos = start_pos[torch.randint(0, start_pos.size(0), (1,))].squeeze(0)
        self._agent_over = GridWorld.TileType.CLEAR
        self._tiles[self._agent_pos[0], self._agent_pos[1]] = GridWorld.TileType.AGENT
        return self.observation()

    def observation(self):
        return self._agent_pos.clone()

    def full_observation(self):
        return self._tiles.clone()

    def step(self, action):
        drift = torch.rand((1,))
        if drift < 0.5 * self.drift_probability:
            action = (action + 1) % 4
        elif drift < self.drift_probability:
            action = (action - 1) % 4

        self._tiles[self._agent_pos[0], self._agent_pos[1]] = self._agent_over

        x, y = {GridWorld.Direction.NORTH: (self._agent_pos[1], self._agent_pos[0] - 1),
                GridWorld.Direction.WEST: (self._agent_pos[1] - 1, self._agent_pos[0]),
                GridWorld.Direction.SOUTH: (self._agent_pos[1], self._agent_pos[0] + 1),
                GridWorld.Direction.EAST: (self._agent_pos[1] + 1, self._agent_pos[0])}.get(action)

        reward = -self.move_cost
        if self._tiles[y, x] == GridWorld.TileType.WALL:
            reward -= self.bump_cost
        else:
            self._agent_pos[1], self._agent_pos[0] = x, y
        over = self._tiles[self._agent_pos[0], self._agent_pos[1]].item()

        clean = torch.rand((1,))
        if clean < self.clean_probability and over == GridWorld.TileType.DIRTY:
            self._agent_over = GridWorld.TileType.CLEAR
        else:
            self._agent_over = over

        self._tiles[self._agent_pos[0], self._agent_pos[1]] = GridWorld.TileType.AGENT

        done = False
        dirty_mask = self._tiles == GridWorld.TileType.DIRTY
        if not dirty_mask.any():
            done = (over == GridWorld.TileType.GOAL) or not self._has_goal
        if over == GridWorld.TileType.TRAP:
            reward -= self.trap_cost
            done = True

        return self.observation(), reward, done
