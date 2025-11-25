from dataclasses import dataclass
from scipy.stats import poisson
from typing import Tuple
import numpy as np


State = Tuple[int, int]
Action = int  # Positive: loc1 → loc2, Negative: loc2 → loc1


@dataclass(frozen=True)
class JacksCarRentalConfig:
    max_cars: int = 20

    # poisson means for rentals and returns at each location
    λ1_rent: float = 3.0
    λ2_rent: float = 4.0
    λ1_return: float = 3.0
    λ2_return: float = 2.0

    # moving cars between locations
    max_move: int = 5
    move_cost: float = 2.0
    rental_revenue: float = 10.0

    # for exercise 4.7
    free_moves_from_1_to_2: int = 0
    max_free_parking: int = 0
    extra_parking_cost: float = 0


class JacksCarRental:
    """Implementation of Jack's Car Rental MDP environment"""

    RENT, RETURN = 0, 1  # event types
    POINT, TAIL = 0, 1  # P(X = k), P(X >= k)
    LOC1, LOC2 = 0, 1  # location indices

    def __init__(self, config):
        self.config = config
        self._build_state_space()
        self._build_action_space()
        self._build_poisson_probability_cache()
        self._build_transition_model()
        self._build_action_model()

    def _build_state_space(self):
        """All (cars at loc1, cars at loc2) pairs from 0..max_cars."""
        max_cars = self.config.max_cars
        self.state_space = [
            (cars1, cars2)
            for cars1 in range(max_cars + 1)
            for cars2 in range(max_cars + 1)
        ]

    def _build_action_space(self):
        """Generate all possible move actions
        (includes invalid moves, but they get clipped in moves)"""
        max_move = self.config.max_move
        self.action_space = list(range(-max_move, max_move + 1))

    def _build_poisson_probability_cache(self):
        """
        Precompute Poisson PMF and tail-function.

        Cache Dimensions:
        [dist_type][event_type][n_events][location]
        - dist_type   ∈ {POINT=0, TAIL=1}
        - event_type  ∈ {RENT=0, RETURN=1}
        - n_events    ∈ {0,1,...,max_cars}
        - location    ∈ {LOC1=0, LOC2=1}
        """
        max_cars = self.config.max_cars

        λ_rentals = [self.config.λ1_rent, self.config.λ2_rent]
        λ_returns = [self.config.λ1_return, self.config.λ2_return]
        λ_values = np.array([λ_rentals, λ_returns])

        events = np.arange(max_cars + 1)
        events_reshaped = events.reshape((max_cars + 1, 1, 1))

        pmf = poisson.pmf(events_reshaped, λ_values)
        sf = poisson.sf(events_reshaped - 1, λ_values)

        cache = np.stack([pmf, sf], axis=0)
        self._poisson_cache = np.transpose(cache, (0, 2, 1, 3))

    def _build_transition_model(self):
        """Build expected rental revenue and transition probabilities"""
        config = self.config
        max_cars = config.max_cars

        # precompute per-location rental revenue and transition probabilities
        exp_rev_local = np.zeros((max_cars + 1, 2))  # [cars, location]
        trans_prob_local = np.zeros(
            (max_cars + 1, max_cars + 1, 2)
        )  # [cars_before, cars_after, location]

        for cars_before in range(max_cars + 1):
            for requests in range(cars_before + 1):

                request_prob = self._poisson_cache[
                    self.TAIL if requests == cars_before else self.POINT,
                    self.RENT,
                    requests,
                ]

                # expected revenue
                cars_after_rent = cars_before - requests
                exp_rev_local[cars_before] += (
                    requests * config.rental_revenue * request_prob
                )

                max_returns = max_cars - cars_after_rent
                for returns in range(max_returns + 1):
                    return_prob = self._poisson_cache[
                        self.TAIL if returns == max_returns else self.POINT,
                        self.RETURN,
                        returns,
                    ]

                    # transition probabilities
                    cars_after_return = cars_after_rent + returns
                    trans_prob_local[cars_before][cars_after_return] += (
                        request_prob * return_prob
                    )

        # build full state transition model
        self._expected_revenue = np.zeros((config.max_cars + 1, config.max_cars + 1))
        self._transition_prob = np.zeros(
            (
                max_cars + 1,
                max_cars + 1,
                max_cars + 1,
                max_cars + 1,
            )
        )
        for cars1, cars2 in self.state_space:
            self._expected_revenue[cars1, cars2] = (
                exp_rev_local[cars1][self.LOC1] + exp_rev_local[cars2][self.LOC2]
            )

            for cars1_after, cars2_after in self.state_space:
                self._transition_prob[cars1, cars2, cars1_after, cars2_after] = (
                    trans_prob_local[cars1][cars1_after][self.LOC1]
                    * trans_prob_local[cars2][cars2_after][self.LOC2]
                )

        self.trans_prob_local = trans_prob_local

    def _build_action_model(self):
        """
        For every (state, action), pre-compute:
          - after_state: how many cars end up at each location after moving
          - immediate_cost: cost of moving + parking penalty

        Note: illegal moves (more cars get moved then available) get clipped
        but move cost is calculated for full action to penalize illegal moves
        """
        config = self.config
        max_cars = config.max_cars
        self._move_model = {}

        for s in self.state_space:
            for a in self.action_space:
                cars1, cars2 = s
                move_amount = abs(a)

                # calculate move cost (unclipped) and after_state (clipped)
                move_cost = 0
                if a > 0:
                    move_cost = -config.move_cost * max(
                        0, move_amount - config.free_moves_from_1_to_2
                    )
                    actual_move = min(move_amount, cars1, max_cars - cars2)
                    after_state = (cars1 - actual_move, cars2 + actual_move)
                elif a < 0:
                    move_cost = -config.move_cost * move_amount
                    actual_move = min(move_amount, cars2, max_cars - cars1)
                    after_state = (cars1 + actual_move, cars2 - actual_move)
                else:
                    after_state = s

                # calculate parking penalty
                parking_cost = 0
                for cars_after in after_state:
                    if cars_after > config.max_free_parking:
                        parking_cost -= config.extra_parking_cost

                self._move_model[(s, a)] = (after_state, move_cost + parking_cost)

    def move(self, s: State, a: Action):
        """
        Returns (after_state, immediate_cost) after moving cars:
            - a>0: loc1→loc2
            - a<0: loc2→loc1
        """
        return self._move_model[(s, a)]

    def get_expected_revenue(self, s: State):
        """Expected rental revenue for after-state"""
        cars1, cars2 = s
        return self._expected_revenue[cars1, cars2]

    def get_transition_probability(self, s1: State, s2: State):
        """Probability of landing in s2 from after-state s1."""
        cars1, cars2 = s1
        cars1_new, cars2_new = s2
        return self._transition_prob[cars1, cars2, cars1_new, cars2_new]
