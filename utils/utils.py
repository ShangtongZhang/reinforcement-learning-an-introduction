#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import itertools
import heapq
def argmax(elements, unique=True):
    maxValue = np.max(elements)
    candidates = np.where(np.asarray(elements) == maxValue)[0]
    if unique:
        return np.random.choice(candidates)
    return list(candidates)

def pad(array, length, defaultValue=0.0):
    if len(array) > length:
        return array[0: length]
    else:
        return array + [defaultValue] * (length - len(array))

class PriorityQueue:

    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def addItem(self, item, priority=0):
        if item in self.entry_finder:
            self.removeItem(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def removeItem(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def popTask(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder
