#######################################################################
# Copyright (C)                                                       #
# 2018 Sergii Bondariev (sergeybondarev@gmail.com)                    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#
# This is a reproduction of the plot shown in Example 13.1
# in Chapter 13, "Policy Gradient Methods". Book draft May 27, 2018.
# 

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def f(p):
    """ True value of the first state
    Args:
        p (float): probability of the action 'right'.
    Returns:
        True value of the first state.
        The expression is obtained by manually solving the easy linear system 
        of Bellman equations using known dynamics.
    """
    return (2 * p - 4) / (p * (1 - p))

epsilon = 0.05
fig, ax = plt.subplots(1, 1)

# Plot a graph 
p = np.linspace(0.01, 0.99, 100)
y = f(p)
ax.plot(p, y, color='red')

# Find a maximum point, can also be done analytically by taking a derivative
imax = np.argmax(y)
pmax = p[imax]
ymax = y[imax]
ax.plot(pmax, ymax, color='green', marker="*", label="optimal point: f({0:.2f}) = {1:.2f}".format(pmax, ymax))

# Plot points of two epsilon-greedy policies
ax.plot(epsilon, f(epsilon), color='magenta', marker="o", label="epsilon-greedy left")
ax.plot(1 - epsilon, f(1 - epsilon), color='blue', marker="o", label="epsilon-greedy right")

ax.set_ylabel("Value of the first state")
ax.set_xlabel("Probability of the action 'right'")
ax.set_title("Short corridor with switched actions")
ax.set_ylim(ymin=-105.0, ymax=5)
ax.legend()
fig.tight_layout()
plt.show()
