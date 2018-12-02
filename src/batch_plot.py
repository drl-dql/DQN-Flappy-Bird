"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Visualize loss and reward for all experiments in one run.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    """Plot."""
    logs_paths = ['../../../Project/DDQN-pytorch-master/logs/']
    for trial in range(1, 34):
        logs_paths.append(
            "../../../Project/DDQN-pytorch-master_{}/logs/".format(trial))

    for idx, logs_path in enumerate(logs_paths):

        # read in reward and loss
        episode, reward = zip(*np.load(os.path.join(logs_path, 'reward.npy')))
        _, loss = zip(*np.load(os.path.join(logs_path, 'loss.npy')))
        avg_reward = np.cumsum(reward) / np.arange(1, len(reward) + 1)

        # subplot
        fig, ax1 = plt.subplots(figsize=(20, 10))

        # subplot for loss
        color = 'tab:orange'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(episode, loss, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()
        ax1.set_ylim(0, max(avg_reward))

        # subplot for average reward
        color = 'tab:blue'
        ax2.set_ylabel('Average Reward', color=color)
        ax2.plot(episode, avg_reward, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # otherwise the right y-label is slightly clipped
        fig.tight_layout()

        if not os.path.isdir("../outs/"):
            os.mkdir("../outs/")
        plt.savefig("../outs/loss_reward_{}.png".format(idx), format='png')


if __name__ == '__main__':
    main()
