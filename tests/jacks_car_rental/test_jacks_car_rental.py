import pytest
import numpy as np
from jacks_car_rental.jacks_car_rental import JacksCarRental, JacksCarRentalConfig
from pytest import approx
from scipy.stats import poisson

# -----------------------------
# Test Initialization
# -----------------------------


def test_initialization():
    config = JacksCarRentalConfig()
    env = JacksCarRental(config)
    assert env.config == config
    assert hasattr(env, "state_space")
    assert hasattr(env, "action_space")
    assert hasattr(env, "_poisson_cache")
    assert hasattr(env, "_expected_revenue")
    assert hasattr(env, "_transition_prob")


# -----------------------------
# Test State and Action Spaces
# -----------------------------


def test_state_space_size():
    config = JacksCarRentalConfig(max_cars=2)
    env = JacksCarRental(config)
    assert len(env.state_space) == 3 * 3


def test_action_space():
    config = JacksCarRentalConfig(max_move=5)
    env = JacksCarRental(config)
    assert env.action_space == [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]


# -----------------------------
# Test Move Method
# -----------------------------


def test_move_positive_action():
    config = JacksCarRentalConfig(max_cars=10, move_cost=2, free_moves_from_1_to_2=1)
    env = JacksCarRental(config)
    state = (5, 5)
    action = 3
    new_state, cost = env.move(state, action)
    assert new_state == (2, 8)
    assert cost == -2 * (3 - 1)


def test_move_negative_action():
    config = JacksCarRentalConfig(max_cars=10, move_cost=1)
    env = JacksCarRental(config)
    state = (3, 5)
    action = -2
    new_state, cost = env.move(state, action)
    assert new_state == (5, 3)
    assert cost == -1 * 2


def test_move_clipping():
    config = JacksCarRentalConfig(max_cars=10, max_move=10, move_cost=1)
    env = JacksCarRental(config)
    state = (5, 5)
    action = 10
    new_state, cost = env.move(state, action)
    assert new_state == (0, 10)
    assert cost == -1 * 10


def test_move_clipping_negative_action():
    config = JacksCarRentalConfig(max_cars=10, max_move=10, move_cost=1)
    env = JacksCarRental(config)
    state = (5, 5)
    action = -10
    new_state, cost = env.move(state, action)
    assert new_state == (10, 0)
    assert cost == -1 * 10


def test_parking_cost():
    config = JacksCarRentalConfig(max_cars=10, max_free_parking=5, extra_parking_cost=3)
    env = JacksCarRental(config)
    state = (7, 3)
    action = 0
    new_state, cost = env.move(state, action)
    assert new_state == (7, 3)
    assert cost == -3


# -----------------------------
# Test Expected Revenue
# -----------------------------


def expected_revenue_for_location(cars, λ_rent, max_cars, rental_revenue):
    expected = 0.0
    for req in range(cars + 1):
        if req < cars:
            prob = poisson.pmf(req, λ_rent)
        else:
            prob = 1.0 - sum(poisson.pmf(r, λ_rent) for r in range(req))
        expected += req * prob * rental_revenue
    return expected


def test_expected_revenue_known_values():
    config = JacksCarRentalConfig(
        max_cars=2, λ1_rent=1.0, λ2_rent=0.5, rental_revenue=10.0
    )
    env = JacksCarRental(config)

    # Test Case 1: (0, 0) - no cars, no revenue
    assert env.get_expected_revenue((0, 0)) == 0.0

    # Test Case 2: (1, 0) - only location 1 has 1 car
    expected_loc1 = expected_revenue_for_location(1, 1.0, 2, 10.0)
    assert env.get_expected_revenue((1, 0)) == approx(expected_loc1)

    # Test Case 3: (2, 1) - both locations have cars
    expected_loc1 = expected_revenue_for_location(2, 1.0, 2, 10.0)
    expected_loc2 = expected_revenue_for_location(1, 0.5, 2, 10.0)
    assert env.get_expected_revenue((2, 1)) == approx(expected_loc1 + expected_loc2)


def test_expected_revenue_with_zero_rental_rate():
    config = JacksCarRentalConfig(
        max_cars=5, λ1_rent=0.0, λ2_rent=0.0, rental_revenue=10.0
    )
    env = JacksCarRental(config)

    for s1 in range(6):
        for s2 in range(6):
            assert env.get_expected_revenue((s1, s2)) == 0.0


def test_expected_revenue_with_high_rental_rate():
    config = JacksCarRentalConfig(
        max_cars=2, λ1_rent=10.0, λ2_rent=10.0, rental_revenue=10.0
    )
    env = JacksCarRental(config)

    expected_per_location = config.max_cars * config.rental_revenue
    assert env.get_expected_revenue((2, 2)) == approx(
        expected_per_location * 2, rel=1e-3
    )


def test_expected_revenue_additivity():
    config = JacksCarRentalConfig(max_cars=10)
    env = JacksCarRental(config)

    total = env.get_expected_revenue((4, 6))
    part1 = env.get_expected_revenue((4, 0))
    part2 = env.get_expected_revenue((0, 6))
    assert total == approx(part1 + part2)


def test_expected_revenue_with_poisson_cache():
    config = JacksCarRentalConfig(
        max_cars=1, λ1_rent=0.5, λ2_rent=0.5, rental_revenue=10.0
    )
    env = JacksCarRental(config)

    expected = 20 * (1 - np.exp(-0.5))
    assert env.get_expected_revenue((1, 1)) == approx(expected, rel=1e-3)


# -----------------------------
# Test Transition Probabilities
# -----------------------------


def test_transition_prob_sum():
    config = JacksCarRentalConfig(max_cars=5)
    env = JacksCarRental(config)
    total = 0.0
    for s1_new in range(config.max_cars + 1):
        for s2_new in range(config.max_cars + 1):
            total += env.get_transition_probability((0, 0), (s1_new, s2_new))
    assert np.isclose(total, 1.0)


def test_transition_probability_zero_state():
    config = JacksCarRentalConfig()
    env = JacksCarRental(config)
    s1, s2 = (0, 0)
    s1_after, s2_after = (0, 0)
    prob = env.get_transition_probability((s1, s2), (s1_after, s2_after))

    p_return1 = poisson.pmf(0, config.λ1_return)
    p_return2 = poisson.pmf(0, config.λ2_return)

    expected = p_return1 * p_return2  # no condition on the rents
    assert prob == approx(expected)


def test_transition_probability_deterministic():
    config = JacksCarRentalConfig(
        max_cars=3, λ1_rent=0.0, λ2_rent=0.0, λ1_return=0.0, λ2_return=0.0
    )
    env = JacksCarRental(config)

    for cars1 in range(4):
        for cars2 in range(4):
            state = (cars1, cars2)
            prob = env.get_transition_probability(state, state)
            assert prob == approx(1.0)


def test_per_location_transition_prob():
    config = JacksCarRentalConfig(
        max_cars=1, λ1_rent=0.5, λ1_return=0.5, λ2_rent=0.0, λ2_return=0.0
    )
    env = JacksCarRental(config)

    cars_before = 1
    cars_after = 1

    λ_rent = config.λ1_rent
    λ_return = config.λ1_return

    p_rent_0 = poisson.pmf(0, λ_rent)

    p_rent_any = 1 - p_rent_0
    p_return_any = 1 - poisson.pmf(0, λ_return)

    expected = p_rent_0 + p_rent_any * p_return_any

    actual = env.get_transition_probability((cars_before, 0), (cars_after, 0))
    assert actual == approx(expected, rel=1e-3)


# -----------------------------
# Test Configuration Defaults
# -----------------------------


def test_config_defaults():
    config = JacksCarRentalConfig()
    defaults = {
        "max_cars": 20,
        "λ1_rent": 3,
        "λ2_rent": 4,
        "λ1_return": 3,
        "λ2_return": 2,
        "max_move": 5,
        "move_cost": 2.0,
        "rental_revenue": 10.0,
        "free_moves_from_1_to_2": 0,
        "max_free_parking": 0,
        "extra_parking_cost": 0,
    }
    for key, value in defaults.items():
        assert getattr(config, key) == value
