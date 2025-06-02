import pytest
import numpy as np
from jacks_car_rental.jacks_car_rental import JacksCarRental, JacksCarRentalConfig
from dataclasses import asdict
from scipy.stats import poisson
from pytest import approx

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
# Test Poisson Cache
# -----------------------------


def test_poisson_cache_shape():
    config = JacksCarRentalConfig()
    env = JacksCarRental(config)
    assert env._poisson_cache.shape == (4, config.max_cars + 1)


def test_poisson_cache_sums_to_one():
    config = JacksCarRentalConfig()
    env = JacksCarRental(config)
    for i in range(4):
        assert np.isclose(env._poisson_cache[i].sum(), 1.0)


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
    assert cost == -2 * (3 - 1)  # move_cost * (move_amount - free_moves)


def test_move_negative_action():
    config = JacksCarRentalConfig(max_cars=10, move_cost=1)
    env = JacksCarRental(config)
    state = (3, 5)
    action = -2
    new_state, cost = env.move(state, action)
    assert new_state == (5, 3)
    assert cost == -1 * 2  # move_cost * move_amount


def test_move_clipping():
    config = JacksCarRentalConfig(max_cars=10, move_cost=1)
    env = JacksCarRental(config)
    state = (5, 5)
    action = 10
    new_state, cost = env.move(state, action)
    assert new_state == (0, 10)
    assert cost == -1 * 10  # move_cost * move_amount (even if clipped)


def test_parking_cost():
    config = JacksCarRentalConfig(max_cars=10, max_free_parking=5, extra_parking_cost=3)
    env = JacksCarRental(config)
    state = (7, 3)
    action = 0
    new_state, cost = env.move(state, action)
    assert new_state == (7, 3)
    assert cost == -3  # only first location exceeds max_free_parking


# -----------------------------
# Test Expected Revenue
# -----------------------------


# Helper to compute expected revenue for a single location
def expected_revenue_for_location(cars, λ_rent, max_cars, rental_revenue):
    expected = 0.0
    for req in range(cars + 1):
        if req < cars:
            prob = poisson.pmf(req, λ_rent)
        else:
            # Account for all remaining probability mass
            prob = 1.0 - sum(poisson.pmf(r, λ_rent) for r in range(req))
        expected += req * prob * rental_revenue
    return expected


def test_expected_revenue_known_values():
    """
    Test expected revenue against manually calculated values
    using small λ and max_cars for easy verification.
    """
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
    """
    Test expected revenue when λ_rent = 0 (no rentals possible).
    """
    config = JacksCarRentalConfig(
        max_cars=5, λ1_rent=0.0, λ2_rent=0.0, rental_revenue=10.0
    )
    env = JacksCarRental(config)

    # No matter how many cars, no revenue possible
    for s1 in range(6):
        for s2 in range(6):
            assert env.get_expected_revenue((s1, s2)) == 0.0


def test_expected_revenue_with_high_rental_rate():
    """
    Test expected revenue when λ_rent > max_cars.
    Expected revenue should be capped by available cars.
    """
    config = JacksCarRentalConfig(
        max_cars=2, λ1_rent=10.0, λ2_rent=10.0, rental_revenue=10.0  # High λ_rent
    )
    env = JacksCarRental(config)

    # Expected revenue should be capped at max_cars * rental_revenue
    expected_per_location = config.max_cars * config.rental_revenue
    assert env.get_expected_revenue((2, 2)) == approx(
        expected_per_location * 2, rel=1e-3
    )


def test_expected_revenue_additivity():
    """
    Test that expected revenue is additive across locations.
    """
    config = JacksCarRentalConfig(max_cars=10)
    env = JacksCarRental(config)

    total = env.get_expected_revenue((4, 6))
    part1 = env.get_expected_revenue((4, 0))
    part2 = env.get_expected_revenue((0, 6))
    assert total == approx(part1 + part2)


def test_expected_revenue_with_poisson_cache():
    """
    Test that expected revenue matches direct Poisson calculation.
    """
    config = JacksCarRentalConfig(
        max_cars=1, λ1_rent=0.5, λ2_rent=0.5, rental_revenue=10.0
    )
    env = JacksCarRental(config)

    # Expected revenue for (1, 1):
    # E[loc1] = 10 * (1 - e^-0.5)
    # E[loc2] = 10 * (1 - e^-0.5)
    # Total = 20 * (1 - e^-0.5) ≈ 7.87
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
