from race_track_environment import RaceTrackEnv
from race_track_environment import RaceTrack
from monte_carlo import MonteCarloESEveryVisit

import numpy as np

# init track
track_ascii = """
###-------------F
##--------------F
##--------------F
#---------------F
----------------F
----------------F
----------#
---------#
---------#
---------#
---------#
---------#
---------#
---------#
#--------#
#--------#
#--------#
#--------#
#--------#
#--------#
#--------#
#--------#
##-------#
##-------#
##-------#
##-------#
##-------#
##-------#
##-------#
###------#
###------#
###SSSSSS#
"""
race_track = RaceTrack(track_ascii)
env = RaceTrackEnv(race_track, max_velocity=4, max_acceleration=1)

# init policy
policy = np.array(
    [np.random.randint(a_count) for a_count in env.actions_per_state if a_count > 0]
)

# init evaluation
transitions = env.indexed_transitions
#print(transitions)
evaluation = np.where(
    transitions == RaceTrackEnv.DOES_NOT_EXIST,
    -np.inf,
    0.0,
)
evaluation = evaluation[: env.count_non_terminal_states]

np.set_printoptions(precision=1, suppress=True)

#print("before training:")
#print(policy)
#print(evaluation)

monte_carlo = MonteCarloESEveryVisit(env, policy, evaluation, 50)
monte_carlo.train(10_000_000)

#print("after training:")
#print(policy)
#print(evaluation)

print("evaluation at start:", evaluation[0])
example_states,_ = monte_carlo.generate_episode_from_state(1)
bla = []
for s in example_states:
    state = env.index_to_state(s)
    tuple_rep = (state.position[0], state.position[1], state.velocity[0], state.velocity[1]) 
    bla.append(tuple_rep)
race_track.print_ascii(bla)
