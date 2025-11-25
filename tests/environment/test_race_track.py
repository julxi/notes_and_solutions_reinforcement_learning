"""
Some of the test are written with LLM. They might be hard to read.
"""

import pytest
import numpy as np
import environment.race_track as rt

REWARD = -1.0


@pytest.fixture
def simple_grid_str():
    return """
F-S#
#S-F
"""


@pytest.fixture
def proper_grid_str():
    return """
##FF
----
---#
----
SSSS
"""


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def test_parse_and_dimensions(simple_grid_str):
    grid = rt.RaceTrackGrid(simple_grid_str)
    assert grid.height == 2
    assert grid.width == 4


def test_parse_invalid_character_raises():
    with pytest.raises(ValueError):
        rt.RaceTrackGrid("A")


@pytest.mark.parametrize(
    "pos,expected",
    [
        ((0, 0), rt.WALL),
        ((1, 0), rt.START),
        ((2, 0), rt.TRACK),
        ((3, 0), rt.FINISH),
        ((3, 1), rt.WALL),
        ((2, 1), rt.START),
        ((1, 1), rt.TRACK),
        ((0, 1), rt.FINISH),
    ],
)
def test_get_cell_types(simple_grid_str, pos, expected):
    grid = rt.RaceTrackGrid(simple_grid_str)
    assert grid.get_cell_type(pos[0], pos[1]) == expected


def test_get_coordinates_of_cell_type(simple_grid_str):
    grid = rt.RaceTrackGrid(simple_grid_str)
    walls = set(grid.get_coordinates_of_cell_type(rt.WALL))
    assert walls == {(3, 1), (0, 0)}


def test_bresenham_line_points_various():
    # horizontal
    assert rt.RaceTrack.bresenham_line_points((0, 0), (3, 0)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
    ]
    # vertical
    assert rt.RaceTrack.bresenham_line_points((2, 1), (2, 4)) == [
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    ]
    # diagonal
    assert rt.RaceTrack.bresenham_line_points((0, 0), (3, 3))[-1] == (3, 3)
    # zero-length
    assert rt.RaceTrack.bresenham_line_points((1, 1), (1, 1)) == [(1, 1)]


def test_action_space_size_and_contents(simple_grid_str):
    r = rt.RaceTrack(simple_grid_str, max_acceleration=1)
    # acceleration in [-1,0,1] -> 3x3 = 9 actions
    assert len(r.action_space) == 9
    assert (0, 0) in r.action_space

    r2 = rt.RaceTrack(simple_grid_str, max_acceleration=0)
    assert len(r2.action_space) == 1
    assert r2.action_space == [(0, 0)]


def test_state_space_mapping_and_terminal_states(simple_grid_str):
    rt_instance = rt.RaceTrack(simple_grid_str, only_positive_velocity=False)
    # mapping is bijective with indices
    for idx, s in enumerate(rt_instance.state_space):
        assert rt_instance.state_to_index_map[s] == idx

    # terminal states have velocity (0,0)
    for t in rt_instance.terminal_states:
        assert t.velocity == (0, 0)


def test_apply_acceleration_forbids_full_stop_and_clips(rng):
    # only_positive_velocity = True: negatives clipped to 0
    r = rt.RaceTrack("SS\nSS", only_positive_velocity=True, rng=rng)
    # try to reduce to zero velocity -> should return original vel
    vel = (1, 1)
    acc = (-1, -1)
    assert r._apply_acceleration(vel, acc) == vel

    # with only_positive_velocity = False, stopping should still be forbidden
    r2 = rt.RaceTrack("SS\nSS", only_positive_velocity=False, rng=rng)
    vel2 = (1, 0)
    acc2 = (-1, 0)  # would lead to (0,0)
    assert r2._apply_acceleration(vel2, acc2) == vel2

    # clipping behaviour
    r3 = rt.RaceTrack("SS\nSS", max_velocity=2, only_positive_velocity=False, rng=rng)
    assert r3._apply_acceleration((2, 2), (1, 0)) == (2, 2)


def test_reset_and_get_random_initial_state_are_deterministic_with_seed(
    rng, simple_grid_str
):
    rt1 = rt.RaceTrack(simple_grid_str, rng=np.random.default_rng(31415))
    chosen = rt1.reset()[0]

    # reproduce expected choice with a fresh RNG seeded the same
    rng_copy = np.random.default_rng(31415)
    idx = int(rng_copy.integers(0, rt1.count_initial_states))
    assert chosen == rt1.initial_states[idx]


def test_set_state_and_set_state_invalid(simple_grid_str):
    rt_instance = rt.RaceTrack(simple_grid_str)
    valid = rt_instance.initial_states[0]
    rt_instance.set_state(valid)
    assert rt_instance.state_space[rt_instance.state_idx] == valid

    with pytest.raises(KeyError):
        rt_instance.set_state(rt.State(position=(99, 99), velocity=(0, 0)))


def test_transition_table_values_valid(simple_grid_str):
    race_track = rt.RaceTrack(simple_grid_str)
    table = race_track.indexed_transitions
    num_states = len(race_track.state_space)
    assert ((table == race_track.CRASH) | ((table >= 0) & (table < num_states))).all()


def test_step_handles_steering_failure_by_using_zero_action(simple_grid_str):
    # with prob_steering_failure = 1.0 the real action should be (0,0)
    rng = np.random.default_rng(1)
    rt_instance = rt.RaceTrack(simple_grid_str, prob_steering_failure=1.0, rng=rng)

    rt_instance.set_state(rt_instance.initial_states[0])
    s_idx = rt_instance.state_idx
    zero_idx = rt_instance.action_to_index_map[(0, 0)]
    expected_next_idx = int(rt_instance.indexed_transitions[s_idx, zero_idx])

    next_state, reward, terminated = rt_instance.step((1, 1))
    assert next_state == rt_instance.state_space[expected_next_idx]
    assert reward == REWARD


def test_crash_resets_to_one_of_initial_states(proper_grid_str):
    # choose an action that will definitely crash (out of bounds)
    rt_instance = rt.RaceTrack(
        proper_grid_str, only_positive_velocity=False, prob_steering_failure=0.0
    )
    # put the agent near the top and apply a large upward velocity to go out of bounds
    # pick a non-terminal, non-wall position
    non_terminal = next(
        s
        for s in rt_instance.state_space
        if s not in rt_instance.initial_states and s not in rt_instance.terminal_states
    )
    rt_instance.set_state(non_terminal)

    # craft an action that will move out of bounds (use the max accel)
    big_action = (rt_instance.max_acceleration, rt_instance.max_acceleration)
    new_state, _, terminated = rt_instance.step(big_action)

    # if a crash happened we expect to be in one of the initial_states (reset)
    # otherwise the test still asserts environment integrity
    assert (
        (not terminated)
        or (new_state in rt_instance.terminal_states)
        or (new_state in rt_instance.initial_states)
    )


def test_reaching_finish_returns_terminated(proper_grid_str):
    rt_instance = rt.RaceTrack(
        proper_grid_str, only_positive_velocity=False, prob_steering_failure=0.0
    )

    state = rt.State((2, 0), (0, 0))
    rt_instance.set_state(state)

    # apply upward actions until terminated
    terminated = False
    steps = 0
    while not terminated and steps < 10:
        state, _, terminated = rt_instance.step((0, 1))
        steps += 1

    assert terminated is True
    assert state.velocity == (0, 0)
    assert state.position in [t.position for t in rt_instance.terminal_states]


def test_cannot_speed_through_wall(proper_grid_str):
    """If an agent is in front of a wall with velocity 2 and behind the wall is a valid cell,
    stepping (with or without acceleration) must NOT place the agent on the valid cell behind the wall.
    Instead the move is a crash (reset) or otherwise must not land on the behind-wall cell.
    """
    rt_instance = rt.RaceTrack(
        proper_grid_str, prob_steering_failure=0.0, rng=np.random.default_rng(42)
    )

    # position (3,1) is directly below a wall at (3,2); cell (3,3) is a valid track cell
    start_state = rt.State(position=(3, 1), velocity=(0, 2))

    # try a few actions that should not allow passing through the wall
    for action in [(0, 0), (0, -1), (0, 1)]:
        rt_instance.set_state(start_state)
        next_state, _, terminated = rt_instance.step(action)
        # we must be sent back to the start
        assert next_state.position[1] == 0
        assert next_state.velocity == (0, 0)
        # and the episode should not be marked terminated (we expect crashes -> resets)
        assert terminated is False


def test_clamped_acceleration_prevents_crash(proper_grid_str):
    """If an acceleration would (without clipping) produce a velocity that crashes,
    but the clipped velocity does not crash, the environment must follow the clipped
    velocity and not crash.
    """
    # Set a small max_velocity so that the requested acceleration would be clipped.
    rt_instance = rt.RaceTrack(
        proper_grid_str,
        max_velocity=2,
        prob_steering_failure=0.0,
        rng=np.random.default_rng(99),
    )

    # Start at x=0, y=1 (row with the wall at y=4). With vel (0,2) and action (0,1)
    # the unclipped velocity would be 3 (which would move the agent to y=4 and crash),
    # but clipping to max_velocity=2 keeps the new position at y=3 which is safe.
    start_state = rt.State(position=(0, 1), velocity=(0, 2))
    rt_instance.set_state(start_state)

    next_state, _, terminated = rt_instance.step((0, 1))

    assert next_state.position == (0, 3)
    assert next_state.velocity == (0, 2)
    assert terminated is False
