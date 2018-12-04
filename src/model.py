"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Network Architecture.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import torch
import torch.nn as nn

from utils import save_checkpoint, load_checkpoint


class Model(nn.Module):
    """DQN-CNN model."""

    def __init__(self, action_num):
        """DQN-CNN Model Initialization."""
        super(Model, self).__init__()

        self.action_num = action_num
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),

            # conv2
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.ReLU(),

            # conv3
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64 * 7 * 7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.action_num))

    def forward(self, x):
        """Forward pass."""
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out

    def save(self, path, step, optimizer):
        torch.save({
            'step': step,
            'state_dict': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }, path)
            
    def load(self, checkpoint_path, optimizer=None):
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint['step']
        self.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])