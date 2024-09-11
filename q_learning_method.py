import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.ops.numpy_ops import shape


# Reinforcement Learning
# Grid World
# Q-Learning Method



class State:
    def __init__(self, state):
        self.board = np.zeros([gridSize, gridSize])
        self.board[1, 1] = -1
        self.state = state
        self.isEnd = False

    def giveReward(self):
        if self.state == terminationState:
            reward = 1
        else:
            reward = 0
        return reward

    def isEndFunc(self):
        if self.state == terminationState:
            self.isEnd = True

    def nxtPosition(self, action):
        if action == "up":
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)

        # if next state is legal
        if (nxtState[0] >= 0) and (nxtState[0] <= (gridSize-1)):
            if (nxtState[1] >= 0) and (nxtState[1] <= (gridSize-1)):
                return nxtState
        return self.state


class Agent:
    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.state = State(state=initState)
        self.isEnd = self.state.isEnd
        self.lr = alpha
        self.exp_rate = epsilon
        self.decay_gamma = gamma
        self.rewards = []

        # initial Q values
        self.Q_values = {}
        for i in range(gridSize):
            for j in range(gridSize):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(low=0, high= 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.state.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.state.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.state = State(state=initState)
        self.isEnd = self.state.isEnd

    def play(self, iterations):
        iteration = 0
        while iteration < iterations:
            # to the end of game back propagate reward
            if self.state.isEnd:
                # back propagate
                reward = self.state.giveReward()
                self.rewards.append(reward/numSteps[iteration])
                for a in self.actions:
                    self.Q_values[self.state.state][a] = reward
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                iteration = iteration + 1
            else:
                action = self.chooseAction()
                # append trace
                self.states.append([self.state.state, action])
                self.state = self.takeAction(action)
                # mark is end
                self.state.isEndFunc()
                self.isEnd = self.state.isEnd
                numSteps[iteration] = numSteps[iteration] + 1
            if self.exp_rate > 0.01:
                self.exp_rate = self.exp_rate - (self.exp_rate/numIterations)


# Main Code
alpha = 0.5     # learning rate
epsilon = 1.0   # initial epsilon value (epsilon decay is applied at rate 1/numIterations)
gamma = 0.5     # discount factor
gridSize = 5    # square grid size
initState = (0, 0)  # starting position in the grid
terminationState = (gridSize-1, gridSize-1)     # end position in the grid (the goal)
numIterations = 100
numSteps = np.zeros(shape=numIterations)
valueTable = np.zeros(shape=(gridSize, gridSize))

agent = Agent()
agent.play(numIterations)
for row in range(0, gridSize):
    for column in range(0, gridSize):
        valueTable[row,column] = max(agent.Q_values[row,column]['up'],
                                     agent.Q_values[row,column]['down'],
                                     agent.Q_values[row,column]['left'],
                                     agent.Q_values[row,column]['right'])


# Plotting
sns.set_theme(style='darkgrid')
fig, ax = plt.subplots(nrows=1, ncols=1)
ax = sns.heatmap(data=valueTable, cmap='crest', linewidths=0.5, linecolor='white', annot=True)
ax.set_title('Final value table')
plt.savefig('figures/ql_final_value_table.png')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 2.5), layout='constrained')
ax.plot(np.arange(numIterations), numSteps)
ax.set_title('Steps to Goal')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps per episode')
plt.savefig('figures/ql_steps_to_goal.png')
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.5, 2.5), layout='constrained')
ax.plot(np.arange(numIterations), agent.rewards)
ax.set_title('Accumulated rewards')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward per episode')
plt.savefig('figures/ql_accumulated_rewards.png')
plt.show()