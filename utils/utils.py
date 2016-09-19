import numpy as np
def argmax(elements, unique=True):
    maxValue = np.max(elements)
    candidates = [i for i in range(0, len(elements)) if elements[i] == maxValue]
    if unique:
        np.random.shuffle(candidates)
        return candidates[0]
    return candidates
