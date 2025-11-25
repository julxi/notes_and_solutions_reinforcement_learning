"""
Race track environment utilities.

Concepts / behaviour:
- Coordinates are (x, y) with (0, 0) at the bottom-left (y increases upward).
- At start and after resets velocity is (0, 0).
- During an episode, slowing to a full stop (velocity (0, 0)) is not allowed. (staying at (0,0) right after start, however is)
- Action order: 1) velocity changes (according to acceleration and clipping) 2) position changes.
- ASCII track uses characters:
    '#' wall
    'S' start
    '-' track
    'F' finish

Example
-------
ASCII track:

    y ↑
      4 |  ####
      3 |  #--F
      2 |  #--#
      1 |  S--#
      0 |  ####
         +-------→ x
           0123

In this layout:
- (0, 0) is the bottom-left corner ('#')
- The start 'S' is at (0, 1)
- The finish 'F' is at (3, 3)

This module provides:
- RaceTrackGrid: small helper for parsing and querying ASCII tracks
- RaceTrack: a compact discrete racetrack environment (precomputed transitions)
"""

from typing import NamedTuple, List
import numpy as np
import random

# Types
State = NamedTuple(
    "State", [("position", tuple[int, int]), ("velocity", tuple[int, int])]
)
Action = tuple[int, int]
Reward = float

# Cell constants
WALL, WALL_ASCII = 0, "#"
START, START_ASCII = 1, "S"
TRACK, TRACK_ASCII = 2, "-"
FINISH, FINISH_ASCII = 3, "F"

ascii_to_code = {
    WALL_ASCII: WALL,
    START_ASCII: START,
    TRACK_ASCII: TRACK,
    FINISH_ASCII: FINISH,
}

code_to_ascii = {v: k for k, v in ascii_to_code.items()}


class RaceTrackGrid:
    """
    Minimal wrapper around an array representation of an ASCII track.
    """

    def __init__(self, ascii_track: str):
        self._track = RaceTrackGrid.parse_ascii_track(ascii_track)
        self.width, self.height = self._track.shape

    @staticmethod
    def parse_ascii_track(ascii_track: str) -> np.ndarray:
        # Parse and flip lines so y == 0 is the bottom row.
        lines = ascii_track.strip().splitlines()
        lines = lines[::-1]
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0
        grid = np.full((width, height), WALL, dtype=np.uint8)
        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                if ch not in ascii_to_code:
                    raise ValueError(f"Unknown ASCII character '{ch}' in track")
                grid[x, y] = ascii_to_code[ch]
        return grid

    def in_bounds(self, x: int, y: int) -> bool:
        """Return True when (x, y) is inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell_type(self, x: int, y: int) -> int:
        """Return the integer cell code at (x, y)."""
        return int(self._track[x, y])

    def get_coordinates_of_cell_type(self, piece: int) -> List[tuple[int, int]]:
        """Return coordinates of all cells equal to `piece` (e.g. START/FINISH)."""
        positions = np.argwhere(self._track == piece)
        return [(int(pos[0]), int(pos[1])) for pos in positions]


class RaceTrack:
    """
    Discrete racetrack environment.

    Public behaviour summary
    ------------------------
    - Construct with an ASCII track and dynamics spec.
    - Use reset()/step(action)/set_state(state) to interact.
    - step(action) returns (next_state, reward, terminated).
      * Steering failure probability may replace the chosen action with (0,0).
      * On crash (hit wall / out-of-bounds) the environment is reset to a random start.
      * Reaching a finish yields a terminal state (terminated == True).

    Design note
    -----------
    Config parameters (max_velocity, max_acceleration, only_positive_velocity) are
    fixed at construction. To change them, construct a new RaceTrack instance.
    """

    CRASH = -1  # transition hits a wall / out-of-bounds
    REWARD = -1.0  # each step has this reward

    def __init__(
        self,
        ascii_track: str,
        max_velocity: int = 4,
        max_acceleration: int = 1,
        prob_steering_failure: float = 0.1,
        only_positive_velocity: bool = True,
        rng: random.Random | None = None,
    ):
        # RNG used for steering failures and crash resets.
        self.rng: random.Random = random.Random() if rng is None else rng

        # configuration (fixed for the lifetime of this instance)
        self.grid = RaceTrackGrid(ascii_track)
        self.max_velocity = int(max_velocity)
        self.max_acceleration = int(max_acceleration)
        self.only_positive_velocity = bool(only_positive_velocity)

        # dynamics
        self.prob_steering_failure = float(prob_steering_failure)

        # build model objects (state/action spaces)
        self.initial_states = self._build_initial_states()
        self.terminal_states = self._build_terminal_states()
        self.state_space = self._build_state_space()
        self.action_space = self._build_action_space()

        # fast lookups
        self.state_to_index_map = self._build_state_to_index_map()
        self.action_to_index_map = self._build_action_to_index_map()

        # deterministic transitions (indices into state_space or CRASH)
        self.indexed_transitions = self._build_transition_table()

        # cached indices derived from ordering
        self.count_initial_states = len(self.initial_states)
        self.min_terminal_state = len(self.state_space) - len(self.terminal_states)
        self.zero_acc_idx = self.action_space.index((0, 0))

        # runtime pointer to current state index
        self.state_idx = None

    # --- Public API --------------------------------------------------------
    def reset(self):
        """Place the environment in a random start state and return it."""
        s_idx = self.reset_idx()
        return self.state_space[s_idx], 0.0, False

    def set_state(self, state):
        """Force the environment into a specific state (must be from state_space)."""
        self.state_idx = self.state_to_index_map[state]
        return self.state_space[self.state_idx], 0.0, False

    def step(self, action: Action) -> tuple[State, Reward, bool]:
        """
        Apply action and return (next_state, reward, terminated).
        """
        action_index = self.action_to_index_map[action]
        next_state_idx, terminated = self.step_idx(action_index)
        return (self.state_space[next_state_idx], RaceTrack.REWARD, terminated)

    # --- Expert API --------------------------------------------------------
    def reset_idx(self):
        self.state_idx = self.rng.randrange(0, self.count_initial_states)
        return self.state_idx

    def step_idx(self, a_idx):
        taken_a_idx = (
            self.zero_acc_idx
            if self.rng.random() < self.prob_steering_failure
            else a_idx
        )

        next_state_idx = self.indexed_transitions[self.state_idx, taken_a_idx]

        # crash handling: pick a random initial state
        if next_state_idx == RaceTrack.CRASH:
            next_state_idx = self.get_random_initial_state_idx()

        self.state_idx = next_state_idx
        terminated = bool(next_state_idx >= self.min_terminal_state)

        return (self.state_idx, terminated)

    def get_random_initial_state_idx(self) -> int:
        return self.rng.randrange(0, self.count_initial_states)

    # --- Builders -----------------------------------------------------------
    def _build_initial_states(self) -> List[State]:
        """Create State entries for every START cell (velocity (0,0))."""
        coords = self.grid.get_coordinates_of_cell_type(START)
        return [State(position=(int(x), int(y)), velocity=(0, 0)) for (x, y) in coords]

    def _build_terminal_states(self) -> List[State]:
        """Create terminal State entries for every FINISH cell (velocity (0,0))."""
        coords = self.grid.get_coordinates_of_cell_type(FINISH)
        return [State(position=(int(x), int(y)), velocity=(0, 0)) for (x, y) in coords]

    def _build_state_space(self) -> List[State]:
        """
        Enumerate the state-space:

        ordering: [initial_states] [non-terminal states with allowed velocities (except (0,0))] [terminal_states]
        """
        state_space: List[State] = []
        state_space.extend(self.initial_states)

        if self.only_positive_velocity:
            vel_range = range(0, self.max_velocity + 1)
        else:
            vel_range = range(-self.max_velocity, self.max_velocity + 1)
        velocity_vectors = [(vx, vy) for vx in vel_range for vy in vel_range]

        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell_type = self.grid.get_cell_type(x, y)
                if cell_type in (WALL, FINISH):
                    continue
                for velocity in velocity_vectors:
                    if velocity == (0, 0):
                        continue
                    state_space.append(State(position=(x, y), velocity=velocity))

        state_space.extend(self.terminal_states)
        return state_space

    def _build_action_space(self) -> List[Action]:
        """List of permitted acceleration actions (ax, ay)."""
        acc_range = range(-self.max_acceleration, self.max_acceleration + 1)
        return [(ax, ay) for ax in acc_range for ay in acc_range]

    def _build_state_to_index_map(self) -> dict:
        """Map State -> index according to state_space ordering."""
        return {state: idx for idx, state in enumerate(self.state_space)}

    def _build_action_to_index_map(self) -> dict:
        """Map Action -> index according to action_space ordering."""
        return {action: idx for idx, action in enumerate(self.action_space)}

    def _build_transition_table(self) -> np.ndarray:
        """
        Build the transition table (num_states x num_actions).

        Values are either a valid state index or CRASH.
        """
        num_states = len(self.state_space)
        num_actions = len(self.action_space)
        table = np.empty((num_states, num_actions), dtype=int)

        for s_idx, state in enumerate(self.state_space):
            for a_idx, action in enumerate(self.action_space):
                next_state, crash = self._apply_action(state, action)
                if crash:
                    table[s_idx, a_idx] = RaceTrack.CRASH
                else:
                    next_idx = self.state_to_index_map[next_state]
                    table[s_idx, a_idx] = int(next_idx)
        return table

    # --- Physics ------------------------------------------------------------
    def _apply_action(self, state: State, action: Action) -> tuple[State, bool]:
        """
        Apply action to a State; return (next_state, crash_flag).

        - Velocity is applied first, then position is updated.
        - Line between old and new position is checked for walls/finish cells.
        """
        new_velocity = self._apply_acceleration(state.velocity, action)
        new_position = (
            state.position[0] + new_velocity[0],
            state.position[1] + new_velocity[1],
        )
        line_points = RaceTrack.bresenham_line_points(state.position, new_position)

        for x, y in line_points:
            if (not self.grid.in_bounds(x, y)) or (
                self.grid.get_cell_type(x, y) == WALL
            ):
                return (state, True)
            elif self.grid.get_cell_type(x, y) == FINISH:
                return (State(position=(x, y), velocity=(0, 0)), False)
        return (State(position=new_position, velocity=new_velocity), False)

    def _apply_acceleration(
        self, vel: tuple[int, int], acc: tuple[int, int]
    ) -> tuple[int, int]:
        """
        Per-component clamp acceleration; forbids coming to a full stop (0,0) mid-episode.
        """
        new_vx = min(max(vel[0] + acc[0], -self.max_velocity), self.max_velocity)
        new_vy = min(max(vel[1] + acc[1], -self.max_velocity), self.max_velocity)
        if self.only_positive_velocity:
            new_vx = max(0, new_vx)
            new_vy = max(0, new_vy)
        if new_vx == 0 and new_vy == 0:
            return vel
        return (int(new_vx), int(new_vy))

    # --- Utilities ----------------------------------------------------------
    @staticmethod
    def bresenham_line_points(
        start: tuple[int, int], end: tuple[int, int]
    ) -> List[tuple[int, int]]:
        """Return integer points on the line between start and end (inclusive)."""
        x0, y0 = start
        x1, y1 = end
        points: List[tuple[int, int]] = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return points
