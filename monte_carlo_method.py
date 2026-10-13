import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Reinforcement Learning
# Grid World
# Monte Carlo Method


ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class GridWorld:

    def __init__(self, grid_size: int = 5, initial_state: Any = (0, 0), terminal_state=None):
        self.grid_size = grid_size
        self.initial_state = initial_state

        if terminal_state is None:
            terminal_state = (grid_size - 1, grid_size - 1)

        self.terminal_state = terminal_state

    def step(self, state, action):

        row_change, column_change = ACTIONS[action]

        next_row = state[0] + row_change
        next_column = state[1] + column_change

        if 0 <= next_row < self.grid_size and 0 <= next_column < self.grid_size:
            next_state = (next_row, next_column)
        else:
            next_state = state

        done = next_state == self.terminal_state
        reward = 1.0 if done else -0.1

        return next_state, reward, done


class MonteCarloAgent:

    def __init__(self, environment: Any, discount_factor: float | int = 0.95):
        self.environment = environment
        self.discount_factor = discount_factor

        self.actions = list(ACTIONS.keys())
        self.rng = np.random.default_rng()

        self.value_table = np.zeros((environment.grid_size, environment.grid_size))

        self.returns = {
            (row, column): []
            for row in range(environment.grid_size)
            for column in range(environment.grid_size)
        }

    def choose_action(self):
        return self.rng.choice(self.actions)

    def generate_episode(self):

        state = self.environment.initial_state
        episode = []

        while state != self.environment.terminal_state:
            action = self.choose_action()
            next_state, reward, done = self.environment.step(state, action)
            episode.append((state, reward, next_state))
            state = next_state

        return episode

    def train(self, n_episodes: int = 500):

        for _ in range(n_episodes):

            episode = self.generate_episode()

            G = 0.0
            visited_states = set()

            for state, reward, _ in reversed(episode):

                G = reward + self.discount_factor * G

                if state not in visited_states:
                    visited_states.add(state)
                    self.returns[state].append(G)
                    self.value_table[state] = np.mean(self.returns[state])

    def get_value_table(self):

        value_table = self.value_table.copy()

        return value_table


def plot_results(value_table, output_directory: str = "figures"):
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

    fig.savefig(output_directory / "mc_final_value_table.png", dpi=300)

    plt.show()


def main():
    start_time = time.time()

    # Parameters
    grid_size = 5
    initial_state = (0, 0)
    terminal_state = (grid_size - 1, grid_size - 1)

    discount_factor = 0.95
    n_episodes = 500

    # Environment
    environment = GridWorld(
        grid_size=grid_size,
        initial_state=initial_state,
        terminal_state=terminal_state,
    )

    # Agent
    agent = MonteCarloAgent(
        environment=environment,
        discount_factor=discount_factor,
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
    )

    execution_time = time.time() - start_time

    print(f"\nExecution time: {execution_time:.3f} seconds")


if __name__ == '__main__':
    main()
