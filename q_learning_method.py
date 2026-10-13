import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --------------------------------------------------------------------------------------------------
# Reinforcement Learning
# Grid World
# Q-Learning
# --------------------------------------------------------------------------------------------------


ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class GridWorld:
    """Square Grid World environment."""

    def __init__(self, grid_size: int = 5, initial_state: Any = (0, 0), terminal_state=None):
        self.grid_size = grid_size
        self.initial_state = initial_state

        if terminal_state is None:
            terminal_state = (grid_size - 1, grid_size - 1)

        self.terminal_state = terminal_state

    def step(self, state: Any, action):
        """
        Move the agent according to the selected action.

        Returns
        -------
        next_state : tuple
            New grid position.
        reward : float
            Reward received after the action.
        done : bool
            Whether the terminal state was reached.
        """

        row_change, column_change = ACTIONS[action]

        next_row = state[0] + row_change
        next_column = state[1] + column_change

        # Check grid boundaries
        if 0 <= next_row < self.grid_size and 0 <= next_column < self.grid_size:
            next_state = (next_row, next_column)
        else:
            next_state = state

        done = next_state == self.terminal_state
        reward = 1 if done else -0.01

        return next_state, reward, done


class QLearningAgent:
    """Q-learning agent for the Grid World environment."""

    def __init__(self, environment: Any, learning_rate: float | int = 0.5, discount_factor: float | int = 0.9,
                 epsilon: float | int = 1.0, min_epsilon: float | int = 0.01):
        self.environment = environment

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.min_epsilon = min_epsilon

        self.actions = list[str](ACTIONS.keys())
        self.rng = np.random.default_rng()

        # Q-table dimensions:
        # row x column x action
        self.q_values = np.zeros(
            shape=(environment.grid_size, environment.grid_size, len(self.actions)),
            dtype=float,
        )

        # Training history
        self.steps_per_episode = []
        self.total_rewards_per_episode = []
        self.returns_per_episode = []

    def choose_action(self, state):
        """
        Select an action according to an epsilon-greedy policy.
        """

        # Exploration
        if self.rng.random() < self.epsilon:
            return self.rng.integers(len(self.actions))

        # Exploitation
        row, column = state

        return np.argmax(self.q_values[row, column])

    def update_q_value(self, state, action_index, reward, next_state, done):
        """
        Perform the standard Q-learning update.
        """

        row, column = state
        next_row, next_column = next_state

        current_q = self.q_values[row, column, action_index,]

        if done:
            target_q = reward
        else:
            best_next_q = np.max(self.q_values[next_row, next_column])

            target_q = (reward + self.discount_factor * best_next_q)

        # Q-learning update
        self.q_values[row, column, action_index] += self.learning_rate * (target_q - current_q)

    def train(self, n_episodes: int = 100):
        """
        Train the agent using Q-learning.
        """

        epsilon_decay = (self.initial_epsilon - self.min_epsilon) / n_episodes

        for _ in range(n_episodes):

            state = self.environment.initial_state

            done = False
            steps = 0
            total_reward = 0.0
            discounted_return = 0.0

            while not done:
                # Choose action
                action_index = self.choose_action(state)
                action = self.actions[action_index]

                # Execute action
                next_state, reward, done = (self.environment.step(state, action))

                # Update Q-table
                self.update_q_value(
                    state=state,
                    action_index=action_index,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )

                # Update state
                state = next_state

                steps += 1
                total_reward += reward
                discounted_return += (self.discount_factor ** steps) * reward

            # Store episode history
            self.steps_per_episode.append(steps)
            self.total_rewards_per_episode.append(total_reward)
            self.returns_per_episode.append(discounted_return)

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon - epsilon_decay)

    def get_value_table(self):
        """
        Return the maximum Q-value for each grid position.
        """

        value_table = np.max(self.q_values, axis=2)

        # Terminal state's defined value
        value_table[self.environment.terminal_state] = 1.0

        return value_table


def plot_results(value_table, steps_per_episode, total_rewards_per_episode,
                 output_directory="figures", save_figures:bool=False):
    """
    Plot and save Q-learning results.
    """

    sns.set_theme(style="whitegrid")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True, )

    # Final Value Table
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained", )
    fig.suptitle("Final Value Table")

    sns.heatmap(
        data=value_table,
        cmap="crest",
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
    )

    if save_figures:
        fig.savefig(output_directory / "ql_final_value_table.png", dpi=300)

    # Steps to Goal
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained", )
    fig.suptitle("Steps to Goal")

    sns.lineplot(
        x=np.arange(len(steps_per_episode)),
        y=steps_per_episode,
        ax=ax,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps per Episode")

    if save_figures:
        fig.savefig(output_directory / "ql_steps_to_goal.png", dpi=300)

    # Accumulated Rewards
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained", )
    fig.suptitle("Total Rewards per Episode")

    sns.lineplot(
        x=np.arange(len(total_rewards_per_episode)),
        y=total_rewards_per_episode,
        ax=ax,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Rewards")

    if save_figures:
        fig.savefig(output_directory / "ql_total_rewards_per_episode.png", dpi=300)

    plt.show()


def main():
    start_time = time.time()

    # Parameters
    grid_size = 5
    initial_state = (0, 0)
    terminal_state = (grid_size - 1, grid_size - 1)

    learning_rate = 0.5
    discount_factor = 0.95

    epsilon = 1.0
    min_epsilon = 0.01

    n_episodes = 500

    # Environment
    environment = GridWorld(
        grid_size=grid_size,
        initial_state=initial_state,
        terminal_state=terminal_state,
    )

    # Agent
    agent = QLearningAgent(
        environment=environment,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        min_epsilon=min_epsilon,
    )

    # Training
    agent.train(
        n_episodes=n_episodes,
    )

    # Value Table
    value_table = agent.get_value_table()
    print("\nFinal Value Table:")
    print(np.round(value_table, decimals=3, ))

    # Plot Results
    plot_results(
        value_table=value_table,
        steps_per_episode=agent.steps_per_episode,
        total_rewards_per_episode=agent.total_rewards_per_episode,
        save_figures=False,
    )

    execution_time = time.time() - start_time

    print(f"\nExecution time: {execution_time:.3f} seconds")


if __name__ == "__main__":
    main()
