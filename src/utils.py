"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

Helper Functions.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import os
import cv2
import json
import shutil
import logging

import torch
import numpy as np


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


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the
    terminal is saved. In a permanent file. Here we save it to `model_dir/train.log`.

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """
    Save dict of floats in json file.

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """
    Save model and training parameters at checkpoint + 'last.pth.tar'.

    If is_best == True, also saves checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such
                      as epoch, optimizer state_dict
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

    If optimizer is provided, loads state_dict of optimizer assuming it is
    present in checkpoint.

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


def make_video(images, fps):
    """
    Make videos.

    Args:
        images:
        fps:
    """
    import moviepy.editor as mpy
    duration = len(images) / fps

    def make_frame(t):
        """A function `t-> frame at time t` where frame is a w*h*3 RGB array."""
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps

    return clip


def convert(image):
    """
    Convert images to 84 * 84.

    Args:
        image: PLE game screen.
    """
    image = cv2.resize(image, (84, 84))
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

    return image
