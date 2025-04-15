Exercise 2.4

Design and conduct an experiment to demonstrate the difficulties
that sample-average methods have for nonstationary problems.
Use a modified version of the 10-armed testbed in which all the q(a)
start out equal and then take independent random walks. Prepare plots
like Figure 2.1 for an action-value method using sample averages, incrementally computed by α = 1/k ,
and another action-value method using a constant step-size parameter,
α = 0.1. Use ε = 0.1 and, if necessary, runs longer than 1000 plays.

update formula:
NewEstimate ← OldEstimate + StepSize[Target − OldEstimate]
Sample average: StepSize = 1/k
constant step 