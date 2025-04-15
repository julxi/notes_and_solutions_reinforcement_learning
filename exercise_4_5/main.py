# | code-fold: true
import numpy as np
from collections import namedtuple
from itertools import product
from scipy.stats import poisson
from collections import defaultdict
import time
import numba


# ----------------------------------------
# Global Parameters
# ----------------------------------------
EXP_REQUEST_1 = 3
EXP_REQUEST_2 = 4
EXP_RETURN_1 = 3
EXP_RETURN_2 = 2

MAX_CARS_PER_SITE = 20
MAX_CUSTOMERS = MAX_CARS_PER_SITE
MAX_CARS_SWAP = 5
RENTAL_REWARD = 10
MOVE_COST = 2
FREE_MOVES_FROM_1_TO_2 = 1
MAX_PARKING_SPACE = 10
EXTRA_PARKING_FEE = 4

GAMMA = 0.9  # Discount factor
THETA = 1e-4  # Threshold for convergence
BOUND_ACTION_SIZE = 2 * MAX_CARS_SWAP + 1

ActionData = namedtuple(
    "ActionData",
    [
        "count",  # (s1, s2) -> number of valid actions
        "possible_actions",  # (s1, s2) -> all valid actions in (s1, s2)
    ],
)
TransitionData = namedtuple(
    "TransitionData",
    [
        "count",  # how many distinct next states we have for each (s1, s2, a)
        "p",  # probability sums leading to each next state
        "pr",  # probability-weighted reward sums
        "next_s1",  # next s1 values
        "next_s2",  # next s2 values
    ],
)


def build_poisson_cache():
    """
    Builds a cache of Poisson probabilities for:
            0 -> requests at site 1
            1 -> requests at site 2
            2 -> returns at site 1
            3 -> returns at site 2

    Each row in the cache holds Poisson pmf values for k in [0,MAX_CUSTOMERS-1],
    with the last entry storing the 'tail' probability (p(k>=MAX_CUSTOMERS)).
    """
    lam_values = [EXP_REQUEST_1, EXP_REQUEST_2, EXP_RETURN_1, EXP_RETURN_2]
    poisson_cache = np.zeros((4, MAX_CUSTOMERS + 1), dtype=np.float64)

    for i, lam in enumerate(lam_values):
        pmf_so_far = 0.0
        for k in range(MAX_CUSTOMERS):
            p = poisson.pmf(k, lam)
            poisson_cache[i, k] = p
            pmf_so_far += p
        # store tail at index = MAX_CUSTOMERS
        poisson_cache[i, MAX_CUSTOMERS] = 1.0 - pmf_so_far

    return poisson_cache


@numba.njit
def event_probability(r1, r2, ret1, ret2, pc):
    return pc[0, r1] * pc[1, r2] * pc[2, ret1] * pc[3, ret2]


@numba.njit
def calculate_possible_actions():
    """
    store valid actions in possible_actions[s1,s2,:],
    and how many are actually valid in num_actions[s1,s2].
    """
    num_actions = np.zeros(
        (MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1), dtype=np.int32
    )
    possible_actions = np.zeros(
        (MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1, BOUND_ACTION_SIZE),
        dtype=np.int32,
    )

    for s1 in range(MAX_CARS_PER_SITE + 1):
        for s2 in range(MAX_CARS_PER_SITE + 1):
            min_act = -min(s2, MAX_CARS_SWAP)
            max_act = min(s1, MAX_CARS_SWAP)

            valid_list = []
            for a in range(min_act, max_act + 1):
                valid_list.append(a)
            count = len(valid_list)

            num_actions[s1, s2] = count

            for i, act in enumerate(valid_list):
                possible_actions[s1, s2, i] = act

    return ActionData(
        count=num_actions,
        possible_actions=possible_actions,
    )


# --- TRANSITIONS
@numba.njit
def build_transition_data(action_data, pc):
    """
    Fills transitions_* arrays for each (s1, s2, action_index).

    For each (s1, s2, action_index) we group all (r1, r2, ret1, ret2) events that end up in the same (ns1, ns2).
    """
    # Unpack just once:
    shape_4d = (MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1, BOUND_ACTION_SIZE)
    shape_5d = (
        MAX_CARS_PER_SITE + 1,
        MAX_CARS_PER_SITE + 1,
        BOUND_ACTION_SIZE,
        (MAX_CARS_PER_SITE + 1) ** 2,
    )

    transitions_count = np.zeros(shape_4d, dtype=np.int32)
    transitions_p_sum = np.zeros(shape_5d, dtype=np.float64)
    transitions_pr_sum = np.zeros(shape_5d, dtype=np.float64)
    transitions_next_s1 = np.zeros(shape_5d, dtype=np.int32)
    transitions_next_s2 = np.zeros(shape_5d, dtype=np.int32)

    for s1 in range(MAX_CARS_PER_SITE + 1):
        for s2 in range(MAX_CARS_PER_SITE + 1):
            count_actions = action_data.count[s1, s2]

            for a_idx in range(count_actions):
                a = action_data.possible_actions[s1, s2, a_idx]

                # We'll do a local 2D array to accumulate p_sum/pr_sum
                grouped_p = np.zeros(
                    (MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1), dtype=np.float64
                )
                grouped_pr = np.zeros(
                    (MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1), dtype=np.float64
                )

                site1_after = s1 - a
                site2_after = s2 + a

                expenses = 0
                # move costs
                if a > 0:
                    expenses += max(0, a - FREE_MOVES_FROM_1_TO_2) * MOVE_COST
                else:
                    expenses += -a * MOVE_COST
                # parking costs
                if site1_after > MAX_PARKING_SPACE:
                    expenses += EXTRA_PARKING_FEE
                if site2_after > MAX_PARKING_SPACE:
                    expenses += EXTRA_PARKING_FEE

                # Accumulate probabilities
                for r1 in range(MAX_CUSTOMERS + 1):
                    for r2 in range(MAX_CUSTOMERS + 1):
                        for ret1 in range(MAX_CUSTOMERS + 1):
                            for ret2 in range(MAX_CUSTOMERS + 1):
                                p = event_probability(r1, r2, ret1, ret2, pc)
                                if p < 1e-12:
                                    continue  # skip negligible

                                # Fulfilled rentals
                                f1 = min(site1_after, r1)
                                f2 = min(site2_after, r2)
                                rent_income = (f1 + f2) * RENTAL_REWARD
                                immediate_reward = rent_income - expenses

                                # remain after rentals
                                rem1 = site1_after - f1
                                rem2 = site2_after - f2
                                ns1 = rem1 + ret1
                                if ns1 > MAX_CARS_PER_SITE:
                                    ns1 = MAX_CARS_PER_SITE
                                ns2 = rem2 + ret2
                                if ns2 > MAX_CARS_PER_SITE:
                                    ns2 = MAX_CARS_PER_SITE

                                grouped_p[ns1, ns2] += p
                                grouped_pr[ns1, ns2] += p * immediate_reward

                # Now flatten out unique (ns1, ns2) into the transitions_* arrays
                idx = 0
                for ns1 in range(MAX_CARS_PER_SITE + 1):
                    for ns2 in range(MAX_CARS_PER_SITE + 1):
                        p_sum = grouped_p[ns1, ns2]
                        if p_sum > 0.0:
                            transitions_p_sum[s1, s2, a_idx, idx] = p_sum
                            transitions_pr_sum[s1, s2, a_idx, idx] = grouped_pr[
                                ns1, ns2
                            ]
                            transitions_next_s1[s1, s2, a_idx, idx] = ns1
                            transitions_next_s2[s1, s2, a_idx, idx] = ns2
                            idx += 1

                # store how many distinct next states we found
                transitions_count[s1, s2, a_idx] = idx

    return TransitionData(
        count=transitions_count,
        p=transitions_p_sum,
        pr=transitions_pr_sum,
        next_s1=transitions_next_s1,
        next_s2=transitions_next_s2,
    )


@numba.njit
def q_value_of_action_index(s1, s2, a_idx, value_func, transition_data):
    """
    Returns Q(s1,s2, action_index) by summing over p*(r + gamma*V(ns)).
    """
    total = 0.0
    c = transition_data.count[s1, s2, a_idx]
    for n in range(c):
        p = transition_data.p[s1, s2, a_idx, n]
        pr = transition_data.pr[s1, s2, a_idx, n]
        ns1 = transition_data.next_s1[s1, s2, a_idx, n]
        ns2 = transition_data.next_s2[s1, s2, a_idx, n]
        total += pr + p * GAMMA * value_func[ns1, ns2]
    return total


@numba.njit
def policy_evaluation(policy, value_func, transition_data):
    sweep = 0
    while True:
        sweep += 1
        delta = 0.0
        for s1 in range(MAX_CARS_PER_SITE + 1):
            for s2 in range(MAX_CARS_PER_SITE + 1):
                old_val = value_func[s1, s2]
                a_idx = policy[s1, s2]
                value_func[s1, s2] = q_value_of_action_index(
                    s1, s2, a_idx, value_func, transition_data
                )
                delta = max(delta, abs(value_func[s1, s2] - old_val))
        if delta < THETA:
            break
    print("policy_evaluation needed sweeps:", sweep)


@numba.njit
def policy_improvement(policy, value_func, action_data, transition_data):
    """
    Updates 'policy' in-place. 'policy[s1,s2]' will store action_idx,
    Returns true if policy does not change (i.e. optimal)
    """
    policy_stable = True

    for s1 in range(MAX_CARS_PER_SITE + 1):
        for s2 in range(MAX_CARS_PER_SITE + 1):
            old_action_idx = policy[s1, s2]

            best_action_idx = 0
            best_q = -1e20
            n_act = action_data.count[s1, s2]
            for a_idx in range(n_act):
                q_val = q_value_of_action_index(
                    s1, s2, a_idx, value_func, transition_data
                )
                if q_val > best_q:
                    best_q = q_val
                    best_action_idx = a_idx

            policy[s1, s2] = best_action_idx
            if best_action_idx != old_action_idx:
                policy_stable = False

    return policy_stable


def policy_iteration(policy, value_func, action_data, transition_data):
    iteration = 0
    while True:
        print("\nPolicy #", iteration)
        iteration += 1
        print_policy(policy, action_data)
        policy_evaluation(policy, value_func, transition_data)
        done = policy_improvement(policy, value_func, action_data, transition_data)
        if done:
            break


def print_policy(policy, action_data):
    """
    Print the raw actions stored in 'possible_actions' at the indices in 'polSicy'.
    (Rows = s1, Cols = s2. The top row is s1=MAX_CARS_PER_SITE.)
    """
    max_cars = policy.shape[0] - 1  # e.g. 20 if shape is (21,21)
    print("Policy (rows=s1, cols=s2), top row is s1=MAX_CARS_PER_SITE:")
    for s1 in reversed(range(max_cars + 1)):
        row_str = []
        for s2 in range(max_cars + 1):
            a_idx = policy[s1, s2]  # This is the index of the action
            a = action_data.possible_actions[s1, s2, a_idx]  # The actual action value
            row_str.append(f"{a:+2d}")
        print(" ".join(row_str))


def print_values(value_func):
    print("Value Function (rows=s1, cols=s2):")
    for s1 in reversed(range(MAX_CARS_PER_SITE + 1)):
        row_str = " ".join(
            f"{int(value_func[s1, s2]):3d}" for s2 in range(MAX_CARS_PER_SITE + 1)
        )
        print(row_str)


# -----------------------------
# Run everything
# -----------------------------

# policy: (s1, s2) -> action index
policy = np.zeros((MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1), dtype=int)
value_func = np.zeros((MAX_CARS_PER_SITE + 1, MAX_CARS_PER_SITE + 1), dtype=np.float64)

# Build Poisson cache
start_time = time.perf_counter()
poisson_cache = build_poisson_cache()
elapsed_time = time.perf_counter() - start_time
print(f"Building poisson took {elapsed_time:.3f} seconds.")

# Build possible actions
start_time = time.perf_counter()
action_data = calculate_possible_actions()
elapsed_time = time.perf_counter() - start_time
print(f"Building possible actions took {elapsed_time:.3f} seconds.")

# Build transition data
start_time = time.perf_counter()
transition_data = build_transition_data(action_data, poisson_cache)
elapsed_time = time.perf_counter() - start_time
print(f"Building transitions took {elapsed_time:.3f} seconds.")

# Perform policy iteration
policy_iteration(policy, value_func, action_data, transition_data)
print("\nValue function of final policy:")
print_values(value_func)
