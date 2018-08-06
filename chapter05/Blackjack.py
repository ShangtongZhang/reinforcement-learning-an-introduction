#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
ACTIONS = [ACTION_HIT, ACTION_STAND]

# policy for player
POLICY_PLAYER = np.zeros(22)
for i in range(12, 20):
    POLICY_PLAYER[i] = ACTION_HIT
POLICY_PLAYER[20] = ACTION_STAND
POLICY_PLAYER[21] = ACTION_STAND

# function form of target policy of player
def target_policy_player(usable_ace_player, player_sum, dealer_card):
    return POLICY_PLAYER[player_sum]

# function form of behavior policy of player
def behavior_policy_player(usable_ace_player, player_sum, dealer_card):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
POLICY_DEALER = np.zeros(22)
for i in range(12, 17):
    POLICY_DEALER[i] = ACTION_HIT
for i in range(17, 22):
    POLICY_DEALER[i] = ACTION_STAND

# get a new card
def get_card():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# play a game
# @policy_player: specify policy for player
# @initial_state: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initial_action: the initial action
def play(policy_player, initial_state=None, initial_action=None):
    # player status

    # sum of player
    player_sum = 0

    # trajectory of player
    player_trajectory = []

    # whether player uses Ace as 11
    usable_ace_player = False

    # dealer status
    dealer_card1 = 0
    dealer_card2 = 0
    usable_ace_dealer = False

    if initial_state is None:
        # generate a random initial state

        num_of_ace = 0

        # initialize cards of player
        while player_sum < 12:
            # if sum of player is less than 12, always hit
            card = get_card()

            # if get an Ace, use it as 11
            if card == 1:
                num_of_ace += 1
                card = 11
                usable_ace_player = True
            player_sum += card

        # if player's sum is larger than 21, he must hold at least one Ace, two Aces are possible
        if player_sum > 21:
            # use the Ace as 1 rather than 11
            player_sum -= 10

            # if the player only has one Ace, then he doesn't have usable Ace any more
            if num_of_ace == 1:
                usable_ace_player = False

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealer_card1 = get_card()
        dealer_card2 = get_card()

    else:
        # use specified initial state
        usable_ace_player, player_sum, dealer_card1 = initial_state
        dealer_card2 = get_card()

    # initial state of the game
    state = [usable_ace_player, player_sum, dealer_card1]

    # initialize dealer's sum
    dealer_sum = 0
    if dealer_card1 == 1 and dealer_card2 != 1:
        dealer_sum += 11 + dealer_card2
        usable_ace_dealer = True
    elif dealer_card1 != 1 and dealer_card2 == 1:
        dealer_sum += dealer_card1 + 11
        usable_ace_dealer = True
    elif dealer_card1 == 1 and dealer_card2 == 1:
        dealer_sum += 1 + 11
        usable_ace_dealer = True
    else:
        dealer_sum += dealer_card1 + dealer_card2

    # game starts!

    # player's turn
    while True:
        if initial_action is not None:
            action = initial_action
            initial_action = None
        else:
            # get action based on current sum
            action = policy_player(usable_ace_player, player_sum, dealer_card1)

        # track player's trajectory for importance sampling
        player_trajectory.append([(usable_ace_player, player_sum, dealer_card1), action])

        if action == ACTION_STAND:
            break
        # if hit, get new card
        player_sum += get_card()

        # player busts
        if player_sum > 21:
            # if player has a usable Ace, use it as 1 to avoid busting and continue
            if usable_ace_player == True:
                player_sum -= 10
                usable_ace_player = False
            else:
                # otherwise player loses
                return state, -1, player_trajectory

    # dealer's turn
    while True:
        # get action based on current sum
        action = POLICY_DEALER[dealer_sum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        new_card = get_card()
        if new_card == 1 and dealer_sum + 11 < 21:
            dealer_sum += 11
            usable_ace_dealer = True
        else:
            dealer_sum += new_card
        # dealer busts
        if dealer_sum > 21:
            if usable_ace_dealer == True:
            # if dealer has a usable Ace, use it as 1 to avoid busting and continue
                dealer_sum -= 10
                usable_ace_dealer = False
            else:
            # otherwise dealer loses
                return state, 1, player_trajectory

    # compare the sum between player and dealer
    if player_sum > dealer_sum:
        return state, 1, player_trajectory
    elif player_sum == dealer_sum:
        return state, 0, player_trajectory
    else:
        return state, -1, player_trajectory

# Monte Carlo Sample with On-Policy
def monte_carlo_on_policy(episodes):
    states_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_usable_ace_count = np.ones((10, 10))
    states_no_usable_ace = np.zeros((10, 10))
    # initialze counts to 1 to avoid 0 being divided
    states_no_usable_ace_count = np.ones((10, 10))
    for i in tqdm(range(0, episodes)):
        _, reward, player_trajectory = play(target_policy_player)
        for (usable_ace, player_sum, dealer_card), _ in player_trajectory:
            player_sum -= 12
            dealer_card -= 1
            if usable_ace:
                states_usable_ace_count[player_sum, dealer_card] += 1
                states_usable_ace[player_sum, dealer_card] += reward
            else:
                states_no_usable_ace_count[player_sum, dealer_card] += 1
                states_no_usable_ace[player_sum, dealer_card] += reward
    return states_usable_ace / states_usable_ace_count, states_no_usable_ace / states_no_usable_ace_count

# Monte Carlo with Exploring Starts
def monteCarloES(nEpisodes):
    # (playerSum, dealerCard, usableAce, action)
    stateActionValues = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    stateActionPairCount = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    def behaviorPolicy(usableAce, playerSum, dealerCard):
        usableAce = int(usableAce)
        playerSum -= 12
        dealerCard -= 1
        # get argmax of the average returns(s, a)
        values_ = stateActionValues[playerSum, dealerCard, usableAce, :] / stateActionPairCount[playerSum, dealerCard, usableAce, :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # play for several episodes
    for episode in range(nEpisodes):
        if episode % 1000 == 0:
            print('episode:', episode)
        # for each episode, use a randomly initialized state and action
        initialState = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initialAction = np.random.choice(ACTIONS)
        _, reward, trajectory = play(behaviorPolicy, initialState, initialAction)
        for (usableAce, playerSum, dealerCard), action in trajectory:
            usableAce = int(usableAce)
            playerSum -= 12
            dealerCard -= 1
            # update values of state-action pairs
            stateActionValues[playerSum, dealerCard, usableAce, action] += reward
            stateActionPairCount[playerSum, dealerCard, usableAce, action] += 1

    return stateActionValues / stateActionPairCount

# Monte Carlo Sample with Off-Policy
def monteCarloOffPolicy(nEpisodes):
    initialState = [True, 13, 2]
    sumOfImportanceRatio = [0]
    sumOfRewards = [0]
    for i in range(0, nEpisodes):
        _, reward, playerTrajectory = play(behavior_policy_player, initial_state=initialState)

        # get the importance ratio
        importanceRatioAbove = 1.0
        importanceRatioBelow = 1.0
        for (usableAce, playerSum, dealerCard), action in playerTrajectory:
            if action == target_policy_player(usableAce, playerSum, dealerCard):
                importanceRatioBelow *= 0.5
            else:
                importanceRatioAbove = 0.0
                break
        importanceRatio = importanceRatioAbove / importanceRatioBelow
        sumOfImportanceRatio.append(sumOfImportanceRatio[-1] + importanceRatio)
        sumOfRewards.append(sumOfRewards[-1] + reward * importanceRatio)
    del sumOfImportanceRatio[0]
    del sumOfRewards[0]

    sumOfRewards= np.asarray(sumOfRewards)
    sumOfImportanceRatio= np.asarray(sumOfImportanceRatio)
    ordinarySampling = sumOfRewards / np.arange(1, nEpisodes + 1)

    with np.errstate(divide='ignore',invalid='ignore'):
        weightedSampling = np.where(sumOfImportanceRatio != 0, sumOfRewards / sumOfImportanceRatio, 0)

    return ordinarySampling, weightedSampling

# print the state value
figureIndex = 0
def prettyPrint(data, tile, zlabel='reward'):
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
    ax.set_zlabel(zlabel)

def figure_5_1():
    states_usable_ace_1, states_no_usable_ace_1 = monte_carlo_on_policy(10000)
    states_usable_ace_2, states_no_usable_ace_2 = monte_carlo_on_policy(500000)

    states = [states_usable_ace_1, states_no_usable_ace_1,
              states_usable_ace_2, states_no_usable_ace_2]

    titles = ['Usable Ace, 10000 Episodes', 'No Usable Ace, 10000 Episodes',
              'Usable Ace, 500000 Episodes', 'No Usable Ace, 500000 Episodes']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for state, title, axis in zip(states, titles, axes):
        fig = sns.heatmap(state, cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=range(12, 22))
        fig.set_ylabel('player sum')
        fig.set_xlabel('dealer showing')
        fig.set_title(title)

    plt.savefig('../images/figure_5_1.png')
    plt.close()

# Figure 5.3
def figure5_3():
    stateActionValues = monteCarloES(500000)
    stateValueUsableAce = np.zeros((10, 10))
    stateValueNoUsableAce = np.zeros((10, 10))
    # get the optimal policy
    actionUsableAce = np.zeros((10, 10), dtype='int')
    actionNoUsableAce = np.zeros((10, 10), dtype='int')
    for i in range(10):
        for j in range(10):
            stateValueNoUsableAce[i, j] = np.max(stateActionValues[i, j, 0, :])
            stateValueUsableAce[i, j] = np.max(stateActionValues[i, j, 1, :])
            actionNoUsableAce[i, j] = np.argmax(stateActionValues[i, j, 0, :])
            actionUsableAce[i, j] = np.argmax(stateActionValues[i, j, 1, :])
    prettyPrint(stateValueUsableAce, 'Optimal state value with usable Ace')
    prettyPrint(stateValueNoUsableAce, 'Optimal state value with no usable Ace')
    prettyPrint(actionUsableAce, 'Optimal policy with usable Ace', 'Action (0 Hit, 1 Stick)')
    prettyPrint(actionNoUsableAce, 'Optimal policy with no usable Ace', 'Action (0 Hit, 1 Stick)')
    plt.show()

# Figure 5.4
def offPolicy():
    trueValue = -0.27726
    nEpisodes = 10000
    nRuns = 100
    ordinarySampling = np.zeros(nEpisodes)
    weightedSampling = np.zeros(nEpisodes)
    for i in range(0, nRuns):
        ordinarySampling_, weightedSampling_ = monteCarloOffPolicy(nEpisodes)
        # get the squared error
        ordinarySampling += np.power(ordinarySampling_ - trueValue, 2)
        weightedSampling += np.power(weightedSampling_ - trueValue, 2)
    ordinarySampling /= nRuns
    weightedSampling /= nRuns
    axisX = np.log10(np.arange(1, nEpisodes + 1))
    plt.plot(axisX, ordinarySampling, label='Ordinary Importance Sampling')
    plt.plot(axisX, weightedSampling, label='Weighted Importance Sampling')
    plt.xlabel('Episodes (10^x)')
    plt.ylabel('Mean square error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    figure_5_1()

