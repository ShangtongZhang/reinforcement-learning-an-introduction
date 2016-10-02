#######################################################################
# Copyright (C) 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1
actions = [ACTION_HIT, ACTION_STAND]

# policy for player
policyPlayer = np.zeros(22)
for i in range(12, 20):
    policyPlayer[i] = ACTION_HIT
policyPlayer[20] = ACTION_STAND
policyPlayer[21] = ACTION_STAND

# policy for dealer
policyDealer = np.zeros(22)
for i in range(12, 17):
    policyDealer[i] = ACTION_HIT
for i in range(17, 22):
    policyDealer[i] = ACTION_STAND

# get a new card
def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# play a game
def play():
    # sum of player
    playerSum = 0

    numOfAce = 0

    # whether player uses Ace as 11
    usableAce = False

    # initialize cards of player
    while playerSum < 12:
        # if sum of player is less than 12, always hit
        card = getCard()

        # if get an Ace, use it as 11
        if card == 1:
            numOfAce += 1
            card = 11
            usableAce = True
        playerSum += card

    # if player's sum is larger than 21, he must hold at least one Ace, two Aces are possible
    if playerSum > 21:
        # use the Ace as 1 rather than 11
        playerSum -= 10

        # if the player only has one Ace, then he doesn't have usable Ace any more
        if numOfAce == 1:
            usableAce = False

    # initialize cards of dealer, suppose dealer will show the first card ha gets
    dealerCard1 = getCard()
    dealerCard2 = getCard()

    # starting state of player
    state = [usableAce, playerSum, dealerCard1]

    # game starts!

    # player't turn
    while True:
        # get action based on current sum
        action = policyPlayer[playerSum]
        if action == ACTION_STAND:
            break
        # if hit, get new card
        playerSum += getCard()

        # player busts
        if playerSum > 21:
            # if player has a usable Ace, use it as 1 to avoid busting and continue
            if usableAce == True:
                playerSum -= 10
                usableAce = False
            else:
                # otherwise player loses
                return state, -1

    # initialize dealer's sum
    dealerSum = 0
    usableAce = False
    if dealerCard1 == 1 and dealerCard2 != 1:
        dealerSum += 11 + dealerCard2
        usableAce = True
    elif dealerCard1 != 1 and dealerCard2 == 1:
        dealerSum += dealerCard1 + 11
        usableAce = True
    elif dealerCard1 == 1 and dealerCard2 == 1:
        dealerSum += 1 + 11
        usableAce = True
    else:
        dealerSum += dealerCard1 + dealerCard2

    # dealer's turn
    while True:
        # get action based on current sum
        action = policyDealer[dealerSum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        dealerSum += getCard()
        # dealer busts
        if dealerSum > 21:
            if usableAce == True:
            # if dealer has a usable Ace, use it as 1 to avoid busting and continue
                dealerSum -= 10
                usableAce = False
            else:
            # otherwise dealer loses
                return state, 1

    # compare the sum between player and dealer
    if playerSum > dealerSum:
        return state, 1
    elif playerSum == dealerSum:
        return state, 0
    else:
        return state, -1

# Monte Carlo Sample
def monteCarlo(nEpisodes):
    statesUsableAce = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    statesUsableAceCount = np.ones((10, 10))
    statesNoUsableAce = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    statesNoUsableAceCount = np.ones((10, 10))
    for i in range(0, nEpisodes):
        state, reward = play()
        state[1] -= 12
        state[2] -= 1
        if state[0]:
            statesUsableAceCount[state[1], state[2]] += 1
            statesUsableAce[state[1], state[2]] += reward
        else:
            statesNoUsableAceCount[state[1], state[2]] += 1
            statesNoUsableAce[state[1], state[2]] += reward
    return statesUsableAce / statesUsableAceCount, statesNoUsableAce / statesNoUsableAceCount

statesUsableAce1, statesNoUsableAce1 = monteCarlo(10000)
statesUsableAce2, statesNoUsableAce2 = monteCarlo(500000)

# print the state value
figureIndex = 0
def prettyPrint(data, tile):
    global figureIndex
    fig = plt.figure(figureIndex)
    figureIndex += 1
    fig.suptitle(tile)
    ax = fig.add_subplot(111, projection='3d')
    axisX = []
    axisY = []
    axisZ = []
    for i in range(12, 22):
        for j in range(1, 11):
            axisX.append(i)
            axisY.append(j)
            axisZ.append(data[i - 12, j - 1])
    ax.scatter(axisX, axisY, axisZ)
    ax.set_xlabel('player sum')
    ax.set_ylabel('dealer showing')
    ax.set_zlabel('reward')

prettyPrint(statesUsableAce1, 'Usable Ace, 10000 Episodes')
prettyPrint(statesNoUsableAce1, 'No Usable Ace, 10000 Episodes')
prettyPrint(statesUsableAce2, 'Usable Ace, 500000 Episodes')
prettyPrint(statesNoUsableAce2, 'No Usable Ace, 500000 Episodes')
plt.show()


