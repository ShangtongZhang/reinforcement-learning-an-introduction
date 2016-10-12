#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
def argmax(elements, unique=True):
    maxValue = np.max(elements)
    candidates = [i for i in range(0, len(elements)) if elements[i] == maxValue]
    if unique:
        return np.random.choice(candidates)
    return candidates
