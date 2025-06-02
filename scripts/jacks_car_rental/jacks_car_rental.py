# | code-fold: false
from dataclasses import dataclass
from scipy.stats import poisson
from typing import Tuple
import numpy as np


@dataclass
class JacksCarRentalConfig:
    max_cars: int = 20
    λ1_rent: float = 3.0
    λ2_rent: float = 4.0
    λ1_return: float = 3.0
    λ2_return: float = 2.0
    max_move: int = 5
    move_cost: float = 2.0
    rental_revenue: float = 10.0

    # parameters for exercise 4.7
    free_moves_from_1_to_2: int = 0
    max_free_parking: int = 0
    extra_parking_cost: float = 0


State = Tuple[int, int]
Action = int  # >0: from 1 to 2, <0: from 2 to 1


class JacksCarRental:

    # constants for indexing
    LOC1, LOC2 = 0, 1
    REQ1, REQ2, RET1, RET2 = 0, 1, 2, 3

    def __init__(self, config):
        self.config = config
        self._build_state_space()
        self._build_action_space()
        self._build_poisson_cache()
        self._build_transition_model()

    def _build_state_space(self):
        """Build state space as all combinations of cars at both locations"""
        max_cars = self.config.max_cars
        self.state_space = [
            (l1, l2) for l1 in range(max_cars + 1) for l2 in range(max_cars + 1)
        ]

    def _build_action_space(self):
        """Build action space (technically this includes invalid moves)"""
        max_move = self.config.max_move
        self.action_space = [a for a in range(-max_move, max_move + 1)]

    def _build_poisson_cache(self):
        """Precompute Poisson probabilities for requests and returns"""
        max_cars = self.config.max_cars
        λ_values = [
            self.config.λ1_rent,
            self.config.λ2_rent,
            self.config.λ1_return,
            self.config.λ2_return,
        ]
        self._poisson_cache = np.zeros((4, max_cars + 1))
        for idx, λ in enumerate(λ_values):
            self._poisson_cache[idx] = poisson.pmf(np.arange(max_cars + 1), λ)

    def _build_transition_model(self):
        """Build expected rewards and transition probabilities"""
        config = self.config
        max_cars = config.max_cars

        er = np.zeros((max_cars + 1, 2))
        p = np.zeros((max_cars + 1, max_cars + 1, 2))
        # compute the location rewards and probabilities
        for l in range(max_cars + 1):
            p_req_total = np.zeros((2))
            for req in range(l + 1):
                # probabilities of requests
                if req == l:
                    p_req = 1 - p_req_total
                else:
                    p_req = np.array(
                        [
                            self._poisson_cache[self.REQ1, req],
                            self._poisson_cache[self.REQ2, req],
                        ]
                    )
                    p_req_total += p_req

                # expected rewards
                cars_left = l - req
                er[l] += req * config.rental_revenue * p_req

                # probabilities of returns
                p_ret_total = np.zeros((2))
                for ret in range(max_cars - cars_left + 1):
                    if ret == max_cars - cars_left:
                        p_ret = 1 - p_ret_total
                    else:
                        p_ret = np.array(
                            [
                                self._poisson_cache[self.RET1, ret],
                                self._poisson_cache[self.RET2, ret],
                            ]
                        )
                        p_ret_total += p_ret
                    # transition probabilities
                    p[l][cars_left + ret] += p_req * p_ret

        # now the state expected rewards and probabilities
        self._expected_revenue = np.zeros((config.max_cars + 1, config.max_cars + 1))
        self._transition_prob = np.zeros(
            (
                max_cars + 1,
                max_cars + 1,
                max_cars + 1,
                max_cars + 1,
            )
        )
        for l1, l2 in self.state_space:
            self._expected_revenue[l1, l2] = er[l1][self.LOC1] + er[l2][self.LOC2]

            for l1_new, l2_new in self.state_space:
                self._transition_prob[l1, l2, l1_new, l2_new] = (
                    p[l1][l1_new][self.LOC1] * p[l2][l2_new][self.LOC2]
                )

    def move(self, s: State, a: Action):
        """Compute state after moving cars from location 1 to 2 and corresponding costs"""
        config = self.config
        max_cars = self.config.max_cars
        s1, s2 = s
        move_amount = abs(a)
        move_cost = 0
        after_state = s

        # calculate move cost (unclipped) and clipped move amount
        if a > 0:
            move_cost = -config.move_cost * max(
                0, move_amount - config.free_moves_from_1_to_2
            )
            move_amount = min(move_amount, s1, max_cars - s2)
            after_state = (s1 - move_amount, s2 + move_amount)
        if a < 0:
            move_cost = -config.move_cost * move_amount
            move_amount = min(move_amount, s2, max_cars - s1)
            after_state = (s1 + move_amount, s2 - move_amount)

        # calculate rent penalty
        parking_cost = 0
        for cars in after_state:
            if cars > config.max_free_parking:
                parking_cost += -config.extra_parking_cost

        return after_state, move_cost + parking_cost

    def get_expected_revenue(self, s: State):
        s1, s2 = s
        return self._expected_revenue[s1, s2]

    def get_transition_probability(self, s1: State, s2: State):
        l1, l2 = s1
        l1_new, l2_new = s2
        return self._transition_prob[l1, l2, l1_new, l2_new]
