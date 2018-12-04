"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Trainer class.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import os
import numpy as np

from ple import PLE
from ple.games.flappybird import FlappyBird

from replayBuffer import ReplayBuffer
from agent import Agent
from utils import *


class Trainer(object):
    """Trainer."""

    def __init__(self, args, hparams):
        """Trainer initialization."""
        super(Trainer, self).__init__()
        self.args = args
        self.hparams = hparams

    def train(self):
        """Train."""
        logs_path = self.args.logs_path
        video_path = self.args.video_path
        restore = self.args.restore
        train = self.args.train

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
        env = PLE(FlappyBird(),
                  display_screen=False,
                  reward_values=reward_values)

        # Gets the actions FlappyBird supports
        action_set = env.getActionSet()

        replay_buffer = ReplayBuffer(self.hparams.replay_buffer_size)
        agent = Agent(action_set, self.hparams)

        # restore model
        if restore:
            agent.restore(restore)

        reward_logs = []
        loss_logs = []

        for episode in range(1, self.hparams.total_episode + 1):
            # reset env
            env.reset_game()
            env.act(0)
            obs = convert(env.getScreenGrayscale())
            state = np.stack([[obs for _ in range(4)]], axis=0)
            t_alive = 0
            total_reward = 0

            if episode % self.hparams.save_video_frequency == 0 and episode > self.hparams.initial_observe_episode:
                agent.stop_epsilon()
                frames = [env.getScreenRGB()]

            while not env.game_over():
                action = agent.take_action(state)
                reward = env.act(action_set[action])

                if episode % self.hparams.save_video_frequency == 0 and episode > self.hparams.initial_observe_episode:
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
            if episode % self.hparams.save_video_frequency == 0 and episode > self.hparams.initial_observe_episode:
                os.makedirs(video_path, exist_ok=True)
                clip = make_video(frames, fps=60).rotate(-90)
                clip.write_videofile(os.path.join(
                    video_path, 'env_{}.mp4'.format(episode)), fps=60)
                agent.restore_epsilon()
                print('Episode: {} t: {} Reward: {:.3f}'.format(
                    episode, t_alive, total_reward))

            if episode > self.hparams.initial_observe_episode and train:
                # save model
                if episode % self.hparams.save_logs_frequency == 0:
                    agent.save(episode, logs_path)
                    np.save(os.path.join(logs_path, 'loss.npy'),
                            np.array(loss_logs))
                    np.save(os.path.join(logs_path, 'reward.npy'),
                            np.array(reward_logs))

                # update target network
                if episode % self.hparams.update_target_frequency == 0:
                    agent.update_target_network()

                # sample batch from replay buffer
                batch_state, batch_action, batch_reward, batch_state_new, batch_over = replay_buffer.sample(
                    self.hparams.batch_size)

                # update policy network
                loss = agent.update_Q_network(batch_state,
                                              batch_action,
                                              batch_reward,
                                              batch_state_new,
                                              batch_over)

                loss_logs.extend([[episode, loss]])
                reward_logs.extend([[episode, total_reward]])

                # print reward and loss
                if episode % self.hparams.show_loss_frequency == 0:
                    print('Episode: {} t: {} Reward: {:.3f} Loss: {:.3f}'.format(
                        episode, t_alive, total_reward, loss))

                agent.update_epsilon()
