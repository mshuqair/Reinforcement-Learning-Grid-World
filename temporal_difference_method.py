import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Reinforcement Learning
# Grid World
# Temporal Difference Learning Method


def takeAction(curState, action):
    if list(curState) == terminationState:
        return 0, None
    finalState = np.array(curState) + np.array(action)
    if -1 in finalState or gridSize in finalState:
        finalState = curState
    return rewardSize, list(finalState)


# main programs
gamma = 0.1 # discount factor
alpha = 0.2 # (0, 1]  step size
rewardSize = -1
gridSize = 5
initState = [0, 0]
terminationState = [gridSize-1, gridSize-1]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 100
numSteps = np.zeros(numIterations)
valueTable = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}


for it in range(numIterations):
    curState = initState
    while True:
        action = random.choice(actions)
        reward, finalState = takeAction(curState, action)
        if finalState is None:
            break
        before =  valueTable[curState[0], curState[1]]
        valueTable[curState[0], curState[1]] += alpha*(reward + gamma*valueTable[finalState[0], finalState[1]] - valueTable[curState[0], curState[1]])
        deltas[curState[0], curState[1]].append(float(np.abs(before-valueTable[curState[0], curState[1]])))        
        curState = finalState
        numSteps[it] += 1
    if it > 0:
        if numSteps[it] > numSteps[it-1]:
            numSteps[it] = numSteps[it-1]
valueTable = valueTable + 1


# Plotting
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.heatmap(data=valueTable, cmap='crest', linewidths=0.5, linecolor='white', annot=True)
ax.set_title('Final value table')
plt.savefig('figures/td_final_value_table.png')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 2.5), layout='constrained')
ax.plot(np.arange(numIterations), numSteps)
ax.set_title('Steps to Goal')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps per episode')
plt.savefig('figures/td_steps_to_goal.png')
plt.show()


all_series = [list(x)[:numIterations] for x in deltas.values()]
SSE = np.zeros(shape=numIterations)
for series in all_series:
    series = np.asarray(series)
    series = np.power(series, 2)
    if np.shape(series) != (0,):
        SSE = series + SSE
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 2.5), layout='constrained')
ax.plot(np.arange(numIterations), SSE)
ax.set_title('Sum of squared error (SSE)')
ax.set_xlabel('Episode')
ax.set_ylabel('SSE')
plt.savefig('figures/td_sum_squared_error.png')
plt.show()
