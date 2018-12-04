"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600.

High Level Pipeline.

@author: Shubham Bansal, Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
@references:
    [1] Hado van Hasselt, Arthur Guez and David Silver.
        Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461
"""

import argparse

from utils import Params
from train import Trainer


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


def main():
    """Main pipeline for Deep Reinforcement Learning with Double Q-learning."""
    args = parse_args()
    hparams = parse_hparam(args.hparam_path)
    print("==> hyper-parameters parsed successfully!")

    # start training
    print("==> Start training ...")
    trainer = Trainer(args, hparams)
    trainer.train()
    print("==> Training is done!")


if __name__ == '__main__':
    main()
