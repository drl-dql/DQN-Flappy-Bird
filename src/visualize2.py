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
    episode1, reward1 = zip(*np.load(os.path.join(logs_path, 'reward_20.npy')))
    episode2, reward2 = zip(*np.load(os.path.join(logs_path, 'reward_21.npy')))
    episode3, reward3 = zip(*np.load(os.path.join(logs_path, 'reward_22.npy')))
    episode4, reward4 = zip(*np.load(os.path.join(logs_path, 'reward_23.npy')))
    episode5, reward5 = zip(*np.load(os.path.join(logs_path, 'reward_24.npy')))

    _, loss1 = zip(*np.load(os.path.join(logs_path, 'loss_20.npy'), allow_pickle=True))
    _, loss2 = zip(*np.load(os.path.join(logs_path, 'loss_21.npy'), allow_pickle=True))
    _, loss3 = zip(*np.load(os.path.join(logs_path, 'loss_22.npy'), allow_pickle=True))
    _, loss4 = zip(*np.load(os.path.join(logs_path, 'loss_23.npy'), allow_pickle=True))
    _, loss5 = zip(*np.load(os.path.join(logs_path, 'loss_24.npy'), allow_pickle=True))

    # avg_reward1 = np.cumsum(reward1) / np.arange(1, len(reward1) + 1)
    # avg_reward2 = np.cumsum(reward2) / np.arange(1, len(reward2) + 1)
    # avg_reward3 = np.cumsum(reward3) / np.arange(1, len(reward3) + 1)
    # avg_reward4 = np.cumsum(reward4) / np.arange(1, len(reward4) + 1)
    # avg_reward5 = np.cumsum(reward5) / np.arange(1, len(reward5) + 1)

    # subplot
    fig, ax1 = plt.subplots(figsize=(20, 10))

    # subplot for loss
    color = 'tab:orange'
    ax1.set_xlabel('Episodes', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)
    ax1.plot(episode1, loss1, color=color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    # instantiate a second axes that shares the same x-axis
    # ax2 = ax1.twinx()
    # ax1.set_ylim(0, max(avg_reward1, avg_reward2, avg_reward3, avg_reward4, avg_reward5, avg_reward6, avg_reward7))

    # subplot for average reward
    color = 'tab:blue'
    # ax2.set_ylabel('Average Reward', color=color)
    ax1.plot(episode2, loss2, color=color, alpha=0.5)
    # ax2.tick_params(axis='y', labelcolor=color)

    # subplot for average reward
    color = 'tab:red'
    # ax2.set_ylabel('Average Reward', color=color)
    ax1.plot(episode3, loss3, color=color, alpha=0.5)
    # ax2.tick_params(axis='y', labelcolor=color)

    # subplot for average reward
    color = 'tab:pink'
    # ax2.set_ylabel('Average Reward', color=color)
    ax1.plot(episode4, loss4, color=color, alpha=0.5)
    # ax2.tick_params(axis='y', labelcolor=color)

    # subplot for average reward
    color = 'tab:green'
    # ax2.set_ylabel('Average Reward', color=color)
    ax1.plot(episode5, loss5, color=color, alpha=0.5)
    # ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend(['Disc. factor = 0.95', 'Disc. factor = 0.9', 'Disc. factor = 0.99', 'Disc. factor = 0.97', 'Disc. factor = 0.93'], prop = {'size' : 15})
    ax1.set_title('Discount factor [Optimizer : Adam, LR : 1e-3, Batch Size : 32]', fontsize=20)

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    if not os.path.isdir("./outs/"):
        os.mkdir("./outs/")
    plt.savefig("./outs/loss_loss.png", format='png')


if __name__ == '__main__':
    main()
