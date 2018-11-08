"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Helper Functions.

@author: Shubham Bansal,Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
"""

import os
import json
import shutil

import torch


class Params():
    """
    Load hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.lr)
    params.lr = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        """Initialization."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """Save parameters to json file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Load parameters from json file."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Give dict-like access to Params instance by `params.dict['learning_rate']."""
        return self.__dict__


def save_checkpoint(state, is_best, checkpoint):
    """
    Save model and training parameters at checkpoint + 'last.pth.tar'.

    If is_best==True, also saves checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Load model parameters (state_dict) from file_path.

    If optimizer is provided, loads state_dict of optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


#
