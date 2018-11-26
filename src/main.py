"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600

High Level Pipeline.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import argparse
from utils import *


def parse_args():
    """Parse hyper-parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--save_dir', default='results/', metavar='PATH', help='saving directory')
    parser.add_argument('--model_path', default='model/', metavar='PATH', help='model directory')
    parser.add_argument('--hparam_file', default='experiments/base_model/params.json', help='hparam file')

    # training settings
    parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
    parser.add_argument('--cuda', type=bool, default=True, help='whether training using cudatoolkit')

    # parse the arguments
    args = parser.parse_args()

    return args


def parse_hparam(hparam_file):
    """Parse hyper-parameters."""
    params = Params(hparam_file)
    return params


def main():
    """Main pipeline for Deep Reinforcement Learning with Double Q-learning."""
    # default parameters
    args = parse_args()
    print(args)
    # hyper-parameters
    hparam = parse_hparam(args.hparam_file)
    print(hparam)


if __name__ == '__main__':
    main()
