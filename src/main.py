"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

High Level Pipeline.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import os
import random
import argparse
import numpy as np

from ple import PLE
from ple.games.flappybird import FlappyBird

from replayBuffer import ReplayBuffer
from agent import Agent
from utils import *


def parse_args():
    """Parse hyper-parameters."""
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--logs_path', dest='logs_path', type=str,
                        help='path of the checkpoint folder', default='./logs')
    parser.add_argument('-v', '--video_path', dest='video_path', type=str,
                        help='path of the video folder', default='./video')
    parser.add_argument('-r', '--restore', dest='restore', type=str,
                        help='restore checkpoint', default=None)
    parser.add_argument('-t', '--train', dest='train', type=bool,
                        help='train policy or not', default=True)

    # directory
    parser.add_argument('--save_path', type=str,
                        default='results/', help='saving directory')
    parser.add_argument('--model_path', type=str,
                        default='model/', help='model directory')
    parser.add_argument('--hparam_path', type=str,
                        default='experiments/base_model/params.json',
                        help='hparam file')

    # parse the arguments
    args = parser.parse_args()

    return args


def parse_hparam(hparam_path):
    """Parse hyper-parameters."""
    params = Params(hparam_path)
    return params


def train(args, hparams):
    """Train."""
    logs_path = args.logs_path
    video_path = args.video_path
    restore = args.restore
    train = args.train

    # Initial PLE environment
    os.putenv('SDL_VIDEODRIVER', 'fbcon')
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Design reward
    reward_values = {
        "positive": 1,
        "tick": 0.1,
        "loss": -1,
    }

    # Create FlappyBird game env
    env = PLE(FlappyBird(), display_screen=False,
              reward_values=reward_values)

    # Gets the actions FlappyBird supports
    action_set = env.getActionSet()

    replay_buffer = ReplayBuffer(hparam.replay_buffer_size)
    agent = Agent(action_set, hparams)

    # restore model
    if restore:
        agent.restore(restore)

    reward_logs = []
    loss_logs = []

    for episode in range(1, hparam.total_episode + 1):
        # reset env
        env.reset_game()
        env.act(0)
        obs = convert(env.getScreenGrayscale())
        state = np.stack([[obs for _ in range(4)]], axis=0)
        t_alive = 0
        total_reward = 0

        if episode % hparam.save_video_frequency == 0 and episode > hparam.initial_observe_episode:
            agent.stop_epsilon()
            frames = [env.getScreenRGB()]

        while not env.game_over():
            action = agent.take_action(state)
            reward = env.act(action_set[action])

            if episode % hparam.save_video_frequency == 0 and episode > hparam.initial_observe_episode:
                frames.append(env.getScreenRGB())
            obs = convert(env.getScreenGrayscale())
            obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])

            state_new = np.append(state[:, 1:, ...], obs, axis=1)
            action_onehot = np.zeros(len(action_set))
            action_onehot[action] = 1

            t_alive += 1
            total_reward += reward
            replay_buffer.append(
                (state, action_onehot, reward, state_new, env.game_over()))
            state = state_new

        # save video
        if episode % hparam.save_video_frequency == 0 and episode > hparam.initial_observe_episode:
            os.makedirs(video_path, exist_ok=True)
            clip = make_video(frames, fps=60).rotate(-90)
            clip.write_videofile(os.path.join(
                video_path, 'env_{}.mp4'.format(episode)), fps=60)
            agent.restore_epsilon()
            print('Episode: {} t: {} Reward: {:.3f}'.format(
                episode, t_alive, total_reward))

        if episode > hparam.initial_observe_episode and train:
            # save model
            if episode % hparam.save_logs_frequency == 0:
                agent.save(episode, logs_path)
                np.save(os.path.join(logs_path, 'loss.npy'),
                        np.array(loss_logs))
                np.save(os.path.join(logs_path, 'reward.npy'),
                        np.array(reward_logs))

            # update target network
            if episode % hparam.update_target_frequency == 0:
                agent.update_target_network()

            # sample batch from replay buffer
            batch_state, batch_action, batch_reward, batch_state_new, batch_over = replay_buffer.sample(
                hparam.batch_size)

            # update policy network
            loss = agent.update_Q_network(
                batch_state, batch_action, batch_reward, batch_state_new, batch_over)

            loss_logs.extend([[episode, loss]])
            reward_logs.extend([[episode, total_reward]])

            # print reward and loss
            if episode % hparam.show_loss_frequency == 0:
                print('Episode: {} t: {} Reward: {:.3f} Loss: {:.3f}'.format(
                    episode, t_alive, total_reward, loss))

            agent.update_epsilon()


def main():
    """Main pipeline for Deep Reinforcement Learning with Double Q-learning."""
    args = parse_args()
    hparams = parse_hparam(args.hparam_path)

    # start training
    train(args, hparams)


if __name__ == '__main__':
    main()
