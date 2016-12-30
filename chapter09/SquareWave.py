#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

# wrapper class for an interval
# readability is more important than efficiency, so I won't use many tricks
class Interval:
    # [@left, @right)
    def __init__(self, left, right):
        self.left = left
        self.right = right

    # whether a point is in this interval
    def contain(self, x):
        return self.left <= x < self.right

    # length of this interval
    def size(self):
        return self.right - self.left

# domain of the square wave, [0, 2)
domain = Interval(0.0, 2.0)

# square wave function
def squareWave(x):
    if 0.5 < x < 1.5:
        return 1
    return 0

# get @n samples randomly from the square wave
def sample(n):
    samples = []
    for i in range(0, n):
        x = np.random.uniform(domain.left, domain.right)
        y = squareWave(x)
        samples.append([x, y])
    return samples

# wrapper class for value function
class ValueFunction:
    # @domain: domain of this function, an instance of Interval
    # @alpha: basic step size for one update
    def __init__(self, featureWidth, domain=domain, alpha=0.2, numOfFeatures=50):
        self.featureWidth = featureWidth
        self.numOfFeatrues = numOfFeatures
        self.features = []
        self.alpha = alpha
        self.domain = domain

        # there are many ways to place those feature windows,
        # following is just one possible way
        step = (domain.size() - featureWidth) / (numOfFeatures - 1)
        left = domain.left
        for i in range(0, numOfFeatures - 1):
            self.features.append(Interval(left, left + featureWidth))
            left += step
        self.features.append(Interval(left, domain.right))

        # initialize weight for each feature
        self.weights = np.zeros(numOfFeatures)

    # for point @x, return the indices of corresponding feature windows
    def getActiveFeatures(self, x):
        activeFeatures = []
        for i in range(0, len(self.features)):
            if self.features[i].contain(x):
                activeFeatures.append(i)
        return activeFeatures

    # estimate the value for point @x
    def value(self, x):
        activeFeatures = self.getActiveFeatures(x)
        return np.sum(self.weights[activeFeatures])

    # update weights given sample of point @x
    # @delta: y - x
    def update(self, delta, x):
        activeFeatures = self.getActiveFeatures(x)
        delta *= self.alpha / len(activeFeatures)
        for index in activeFeatures:
            self.weights[index] += delta

# train @valueFunction with a set of samples @samples
def approximate(samples, valueFunction):
    for x, y in samples:
        delta = y - valueFunction.value(x)
        valueFunction.update(delta, x)

# Figure 9.8
def figure9_8():
    numOfSamples = [10, 40, 160, 2560, 10240]
    featureWidths = [0.2, 0.4, 1.0]
    axisX = np.arange(domain.left, domain.right, 0.02)
    for numOfSample, index in zip(numOfSamples, range(0, len(numOfSamples))):
        print(numOfSample, 'samples')
        samples = sample(numOfSample)
        valueFunctions = [ValueFunction(featureWidth) for featureWidth in featureWidths]
        plt.figure(index)
        plt.title(str(numOfSample) + ' samples')
        for valueFunction in valueFunctions:
            approximate(samples, valueFunction)
            values = [valueFunction.value(x) for x in axisX]
            plt.plot(axisX, values, label='feature width: ' + str(valueFunction.featureWidth))
        plt.legend()

figure9_8()
plt.show()