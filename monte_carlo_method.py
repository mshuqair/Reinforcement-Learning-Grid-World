import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Reinforcement Learning
# Grid World
# Monte Carlo Method


def generateEpisode():
    curState = initState
    episode = []
    while True:
        # check if agent reaches goal
        if list(curState) == terminationState:
            return episode
        # take action randomly
        action = random.choice(actions)
        finalState = np.array(curState) + np.array(action)
        # check if agent crosses grid world boundaries
        if -1 in finalState or gridSize in finalState:
            finalState = initState
        episode.append([list(curState), action, rewardSize, list(finalState)])
        curState = finalState


# Main Code
gamma = 0.5     # discount factor
rewardSize = -1 # reward for each state except termination state and initial state
gridSize = 5    # grid size
initState = [0, 0]  # initial state at top left
terminationState = [gridSize-1, gridSize-1]     # termination state at bottom right
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]    # actions: left, right, down, up
numIterations = 500                             # number of iterations
numSteps = np.zeros(numIterations)
valueTable = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}


for it in range(numIterations):
    episode = generateEpisode()
    numSteps[it] = len(episode) 
    G = 0
    for i, step in enumerate(episode[::-1]):
        G = gamma*G + step[2]
        # update value function
        if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
            idx = (step[0][0], step[0][1])
            returns[idx].append(G)
            newValue = np.average(returns[idx])
            deltas[idx[0], idx[1]].append(np.abs(valueTable[idx[0], idx[1]]-newValue))
            valueTable[idx[0], idx[1]] = newValue
    if it > 0:
        if numSteps[it] > numSteps[it-1]:
            numSteps[it] = numSteps[it-1]
valueTable = valueTable + 1


# Plotting
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.heatmap(data=valueTable, cmap='crest', linewidths=0.5, linecolor='white', annot=True)
ax.set_title('Final value table')
plt.savefig('figures/mc_final_value_table.png')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 2.5), layout='constrained')
ax.plot(np.arange(numIterations), numSteps)
ax.set_title('Steps to Goal')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps per episode')
plt.savefig('figures/mc_steps_to_goal.png')
plt.show()


all_series = [list(x)[:100] for x in deltas.values()]
SSE = np.zeros(shape=100)
for series in all_series:
    series = np.asarray(series)
    series = np.power(series, 2)
    if np.shape(series) != (0,):
        SSE = series + SSE
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 2.5), layout='constrained')
ax.plot(SSE)
ax.set_title('Sum of squared error (SSE)')
ax.set_xlabel('Episode')
ax.set_ylabel('SSE')
plt.savefig('figures/mc_sum_squared_error.png')
plt.show()
