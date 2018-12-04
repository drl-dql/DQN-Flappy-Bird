"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Visualize loss and reward.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logs_path', dest='logs_path', type=str,
                        help='path of the checkpoint folder', default='./logs')
    args = parser.parse_args()

    return args


args = parse_args()


def main():
    """Plot."""
    args = parse_args()
    logs_path = args.logs_path

    # read in reward and loss
    episode, reward = zip(*np.load(os.path.join(logs_path, 'reward_23.npy')))
    _, loss = zip(*np.load(os.path.join(logs_path, 'loss_23.npy')))
    avg_reward = np.cumsum(reward) / np.arange(1, len(reward) + 1)

    # subplot
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # subplot for loss
    color = 'tab:orange'
    ax1.set_xlabel('Episode', fontsize=20)
    ax1.set_ylabel('Loss', color=color, fontsize=20)
    ax1.plot(episode, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()
    ax1.set_ylim(0, max(avg_reward))

    # subplot for average reward
    color = 'tab:blue'
    ax2.set_ylabel('Average Reward', color=color, fontsize=20)
    ax2.plot(episode, avg_reward, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # ax1.set_title('Adam optimizer, learning rate: 1e-3, Discount factor: 0.97, update target frequency: 13, batch size: 32, intial observe episode: 120', fontsize=20)
    ax1.set_title('Adam optimizer, learning rate: 1e-4, Discount factor: 0.99, update target frequency: 3, batch size: 16, intial observe episode: 120', fontsize=20)

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    if not os.path.isdir("./outs/"):
        os.mkdir("./outs/")
    plt.savefig("./outs/best.png", format='png')


if __name__ == '__main__':
    main()
