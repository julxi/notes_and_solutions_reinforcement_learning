from race_track import RaceTrack, WALL, START, TRACK, FINISH, bresenham_line_points
from collections import defaultdict
from typing import NamedTuple
import numpy as np
from itertools import product


class State(NamedTuple):
    position: tuple[int, int]  # [x, y]
    velocity: tuple[int, int]  # [vx, vy]


class Action(NamedTuple):
    acceleration: tuple[int, int]  # [ax, ay]


class RaceTrackEnv:
    # Index constants
    DOES_NOT_EXIST = -1
    CRASH = -9000

    def __init__(
        self,
        race_track: RaceTrack,
        max_velocity: int = 4,
        max_acceleration: int = 1,
        noise_prob: float = 0.1,
        seed: int = 0,
    ):
        # parameters
        self.RACE_TRACK = race_track
        self.MAX_VELOCITY = max_velocity
        self.MAX_ACCELERATION = max_acceleration
        self.noise_prob = noise_prob
        self.rng = np.random.default_rng(seed)

        # Compute possible starts and precompute transitions
        self.initial_states = self._compute_initial_states()
        self.terminal_states = self._compute_terminal_states()

        # This creates a list of (state, [valid_actions]) pairs
        self.states_with_actions = self._compute_all_states_with_actions()
        self.count_total_states = len(self.states_with_actions)
        self.count_initial_states = len(self.initial_states)
        self.count_non_terminal_states = self.count_total_states - len(
            self.terminal_states
        )
        self.max_count_actions = max(len(a) for _, a in self.states_with_actions)
        self.actions_per_state = [len(a) for _, a in self.states_with_actions]

        # Create a quick state -> index lookup
        self.state_to_index_map, self.state_action_to_index_map = (
            self._compute_state_to_index_maps()
        )

        # Precompute the transition table
        self.indexed_transitions = self._build_transition_dict_indexed()

    # -------------------------------------------------
    # “object-based” environment methods
    # -------------------------------------------------
    def _compute_initial_states(self) -> list[State]:
        return [
            State(
                position=(int(pos[0]), int(pos[1])),
                velocity=(0, 0),
            )
            for pos in self.RACE_TRACK.get_position_of_piece(START)
        ]

    def _compute_terminal_states(self) -> list[State]:
        return [
            State(
                position=(int(pos[0]), int(pos[1])),
                velocity=(0, 0),
            )
            for pos in self.RACE_TRACK.get_position_of_piece(FINISH)
        ]

    def _compute_all_states_with_actions(self) -> list[tuple[State, list[Action]]]:
        """
        Builds a list of (state, list_of_valid_actions) tuples.

        - intial states at the beginning, terminal states at the end.
        - For all non-start states, the list of actions always starts with (0, 0) acceleration.
        - termianl states have no actions.
        """
        state_actions = defaultdict(list)

        acc_range = range(-self.MAX_ACCELERATION, self.MAX_ACCELERATION + 1)
        all_accels = list(product(acc_range, repeat=2))

        vel_range = range(-self.MAX_VELOCITY, self.MAX_VELOCITY + 1)
        all_vels = list(product(vel_range, repeat=2))

        for y in range(self.RACE_TRACK.height):
            for x in range(self.RACE_TRACK.width):
                cell_type = self.RACE_TRACK.get_piece(x, y)
                # we add termianl states later
                if cell_type in (WALL, FINISH):
                    continue

                for vx, vy in all_vels:
                    velocity = (vx, vy)

                    # Skip velocity == (0,0) these are only valid for intital states
                    if velocity == (0, 0):
                        continue

                    for ax, ay in all_accels:
                        acceleration = (ax, ay)
                        new_vel = (
                            velocity[0] + acceleration[0],
                            velocity[1] + acceleration[1],
                        )

                        if (
                            abs(new_vel[0]) > self.MAX_VELOCITY
                            or abs(new_vel[1]) > self.MAX_VELOCITY
                        ):
                            continue
                        if new_vel == (0, 0):
                            continue  # final velocity must not be zero

                        state = State(position=(x, y), velocity=velocity)
                        action = Action(acceleration=acceleration)
                        state_actions[state].append(action)

        result = []

        # First, add all start states
        initial_actions = [
            Action(acceleration=(ax, ay))
            for (ax, ay) in all_accels
            if (ax, ay) != (0, 0)
        ]
        for state in self.initial_states:
            result.append((state, initial_actions))

        # Then add all other states, putting Action((0, 0)) first if it's allowed
        zero_acc = Action(acceleration=(0, 0))

        for state, actions in state_actions.items():
            # Ensure zero acceleration is first if valid
            actions_sorted = [zero_acc] if zero_acc in actions else []
            actions_sorted += [a for a in actions if a != zero_acc]

            result.append((state, actions_sorted))

        for state in self.terminal_states:
            result.append((state, []))

        return result

    def _compute_action(self, state: State, action: Action) -> tuple[State, bool]:
        """
        Applies an action to a given state and returns (next_state, crash).

        - If car goes into wall or out of bounds, it returns the same state and crash=True
        - If car reaches the finish, it returns a terminal state and crash=False
        """
        new_velocity = (
            state.velocity[0] + action.acceleration[0],
            state.velocity[1] + action.acceleration[1],
        )
        new_position = (
            state.position[0] + new_velocity[0],
            state.position[1] + new_velocity[1],
        )

        # Trace all points from old position to new position
        line_points = bresenham_line_points(state.position, new_position)

        for xx, yy in line_points:
            if (not self.RACE_TRACK.in_bounds(xx, yy)) or (
                self.RACE_TRACK.get_piece(xx, yy) == WALL
            ):
                return (state, True)
            elif self.RACE_TRACK.get_piece(xx, yy) == FINISH:
                return (State(position=(xx, yy), velocity=(0, 0)), False)
        return (State(position=new_position, velocity=new_velocity), False)

    # -------------------------------------------------
    # Index-based methods: building tables and usage
    # -------------------------------------------------
    def _compute_state_to_index_maps(self):
        state_to_index_map = {}
        state_action_to_index_map = {}
        for i, (state, actions) in enumerate(self.states_with_actions):
            state_to_index_map[state] = i
            for a, action in enumerate(actions):
                state_action_to_index_map[(state, action)] = (i, a)
        return state_to_index_map, state_action_to_index_map

    def _build_transition_dict_indexed(self) -> np.ndarray:
        num_states = len(self.states_with_actions)
        max_actions = (
            max(len(a) for _, a in self.states_with_actions) if num_states else 0
        )

        # Prepare the table with DOES_NOT_EXIST by default
        table = np.full((num_states, max_actions), self.DOES_NOT_EXIST, dtype=int)

        for i, (state, actions) in enumerate(self.states_with_actions):
            for a_idx, action in enumerate(actions):
                next_state, crash = self._compute_action(state, action)

                if crash:
                    table[i, a_idx] = self.CRASH
                else:
                    next_i = self.state_to_index(next_state)
                    table[i, a_idx] = next_i
        return table

    def state_to_index(self, state: State) -> int:
        return self.state_to_index_map.get(state, self.DOES_NOT_EXIST)

    def index_to_state(self, state_idx: int) -> State:
        if state_idx < 0 or state_idx >= len(self.states_with_actions):
            return None
        state, actions = self.states_with_actions[state_idx]
        return state

    def state_action_to_index(self, state: State, action: Action) -> tuple[int, int]:
        return self.state_action_to_index_map.get((state, action), self.DOES_NOT_EXIST)

    def index_to_state_action(self, i: int, a: int) -> tuple[State, Action] | None:
        if i < 0 or i >= len(self.states_with_actions):
            return None
        state, actions = self.states_with_actions[i]
        if a < 0 or a >= len(actions):
            return None
        return (state, actions[a])

    # -------------------------------------------------
    # Environment interactions
    # -------------------------------------------------
    def reset(self) -> int:
        self.state_idx = self.rng.integers(0, self.count_initial_states)
        return self.state_idx

    def is_terminal(self, state_idx: int) -> bool:
        return state_idx >= self.count_non_terminal_states

    def get_random_state_action_pair(self):
        """Returns non-terminal states."""
        s = self.rng.integers(self.count_non_terminal_states)
        a = self.rng.integers(self.actions_per_state[s])
        return np.array([s, a], dtype=int)

    def get_state_index(self) -> int:
        return self.state_idx

    def get_state(self) -> State:
        return self.index(self.get_state_index())

    def set_state_index(self, state_idx: int):
        self.state_idx = state_idx

    def set_state(self, state: State) -> None:
        self.set_state_index(self.state_to_index(state))

    def take_action_index(self, action_idx: int) -> int:
        next_state_idx = self.indexed_transitions[self.state_idx, action_idx]

        if next_state_idx == RaceTrackEnv.CRASH:
            self.reset()
        else:
            self.state_idx = next_state_idx

        return self.state_idx

    def take_action(self, action: Action) -> State:
        state = self.index_to_state(self.state_idx)
        _, action_idx = self.state_action_to_index(state, action)
        self.take_action_index(action_idx)
        return self.index_to_state(self.state_idx)
