"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Agent Class.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import os
import glob
import random

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.functional import mse_loss

from model import Model


class Agent:
    """Agent for Flappy Bird."""

    def __init__(self, action_set, hparam):
        """Agent Initialization."""
        self.action_set = action_set
        self.action_number = len(action_set)

        self.hparam = hparam
        self.epsilon = self.hparam.initial_epsilon
        self._build_network()

    def _build_network(self):
        """Create Q Network and target network."""
        self.Q_network = Model(4, self.action_number).cuda()
        self.target_network = Model(4, self.action_number).cuda()
        self.optimizer = optim.RMSprop(self.Q_network.parameters(),
                                       lr=self.hparam.lr,
                                       momentum=self.hparam.momentum)

    def update_target_network(self):
        """Copy current network to update target network parameters."""
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def update_Q_network(self, state, action, reward, state_new, terminal):
        """Update Q Network parameters."""
        state = torch.from_numpy(state).float() / 255.0
        action = torch.from_numpy(action).float()
        state_new = torch.from_numpy(state_new).float() / 255.0
        terminal = torch.from_numpy(terminal).float()
        reward = torch.from_numpy(reward).float()

        # wrap them in `Variable` and cuda support
        state = Variable(state).cuda()
        action = Variable(action).cuda()
        state_new = Variable(state_new).cuda()
        terminal = Variable(terminal).cuda()
        reward = Variable(reward).cuda()

        # no parameters update
        self.Q_network.eval()
        self.target_network.eval()

        # use current network to evaluate action argmax_a' Q_current(s', a')_
        action_new = self.Q_network.forward(state_new).max(dim=1)[
            1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(
            self.hparam.batch_size, self.action_number)
        action_new_onehot = Variable(
            action_new_onehot.scatter_(1, action_new, 1.0)).cuda()

        # use target network to evaluate value
        # y = r + discount_factor * Q_tar(s', a')
        y = (reward + torch.mul(((self.target_network.forward(state_new) *
                                  action_new_onehot).sum(dim=1) * terminal),
                                self.hparam.discount_factor))

        # regression Q(s, a) -> y
        self.Q_network.train()
        Q = (self.Q_network.forward(state) * action).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())

        # optimizer perform back propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def take_action(self, state):
        """Perform an action."""
        state = torch.from_numpy(state).float() / 255.0
        state = Variable(state).cuda()

        self.Q_network.eval()
        estimate = self.Q_network.forward(state).max(dim=1)

        # with epsilon prob to choose random action else choose argmax Q estimate action
        if random.random() < self.epsilon:
            return random.randint(0, self.action_number - 1)
        else:
            return estimate[1].data[0]

    def update_epsilon(self):
        """Update parameter epsilon."""
        if self.epsilon > self.hparam.min_epsilon:
            self.epsilon -= self.hparam.epsilon_discount_rate

    def stop_epsilon(self):
        """Reset parameter epsilon."""
        self.epsilon_tmp = self.epsilon
        self.epsilon = 0

    def restore_epsilon(self):
        """Restore parameter epsilon."""
        self.epsilon = self.epsilon_tmp

    def save(self, step, logs_path):
        """Save model."""
        os.makedirs(logs_path, exist_ok=True)
        model_list = glob.glob(os.path.join(logs_path, '*.pth'))

        # remove oldest model and save the current model
        if len(model_list) > self.hparam.maximum_model - 1:
            min_step = min([int(li.split('/')[-1][6:-4]) for li in model_list])
            os.remove(os.path.join(logs_path, 'model-{}.pth' .format(min_step)))
        logs_path = os.path.join(logs_path, 'model-{}.pth' .format(step))

        self.Q_network.save(logs_path, step=step, optimizer=self.optimizer)
        print('==> Save model to {} successfully'.format(logs_path))

    def restore(self, logs_path):
        """Restore model."""
        self.Q_network.load(logs_path)
        self.target_network.load(logs_path)
        print('==> Restore model from {} successfully'.format(logs_path))
