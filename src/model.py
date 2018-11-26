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

    def __init__(self, input_size, output_size):
        """DQN-CNN Model Builder."""
        super(Model, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=self.input_size, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),

            # conv3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.output_size))

    def forward(self, observation):
        """Forward pass."""
        x = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out
