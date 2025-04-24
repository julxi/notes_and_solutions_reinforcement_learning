# Exercise 4.9 (programming)

Implement value iteration for the gambler’s
problem and solve it for p_h = 0.25 and p_h = 0.55. In programming, you may
find it convenient to introduce two dummy states corresponding to termination
with capital of 0 and 100, giving them values of 0 and 1 respectively. Show
your results graphically, as in Figure 4.6. Are your results stable as θ → 0?

## Policy Iteration
Once a policy, π, has been improved using v_π to yield a better policy, π',
we can then compute v_{π'} and improve it again to yield an even better π''.
We can thus obtain a sequence of monotonically improving policies and value functions:

π_0 −E→ v_{π_0} −I→ π_1 −E→ v_{π_1} −I→ π2 −E→ ... −I→ π* −E→ v*,

where −E→ denotes a policy evaluation and −I→ denotes a policy improvement.
Each policy is guaranteed to be a strict improvement over the previous one
(unless it is already optimal). Because a finite MDP has only a finite number
of policies, this process must converge to an optimal policy and optimal value
function in a finite number of iterations.


```
1. Initialization
    V(s) ∈ R and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
    Repeat
        ∆ ← 0
        For each s ∈ S:
            v ← V (s)
            V(s) ← Σ_{s',r} p(s',r|s,π(s)) [r + γV(s')]
            ∆ ← max(∆, |v − V (s)|)
    until ∆ < θ (a small positive number)

3. Policy Improvement
    policy-stable ← true
    For each s ∈ S:
        a ← π(s)
        π(s) ← argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        If a != π(s), then policy-stable ← false
    If policy-stable, then stop and return V and π; else go to 2
```

Policy iteration (using iterative policy evaluation) for v*.
This algorithm has a subtle bug, in that it may never terminate
if the policy continually switches between two or more policies
that are equally good.
The bug can be fixed by adding additional flags,
but it makes the pseudocode so ugly that it is not worth it.
