"""
Deep Reinforcement Learning with Double Q-learning on Atari 2600

High Level Pipeline.

@author: Shubham Bansal,Naman Shukla, Ziyu Zhou, Jianqiu Kong, Zhenye Na
"""


import argparse


def parse_args():
    """Parse hyper-parameters."""
    parser = argparse.ArgumentParser()

    # directory
    parser.add_argument('--dataroot', type=str, default="../data", help='path to dataset')
    parser.add_argument('--ckptroot', type=str, default="../model/", help='path to checkpoint')

    # hyperparameters settings
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.95, help='momentum parameter for RMSProp Optim')
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay (L2 penalty)')
    # parser.add_argument('--epochs', type=int, default=233, help='number of epochs to train')
    # parser.add_argument('--batch_size', type=int, default=128, help='training input batch size')

    # training settings
    parser.add_argument('--resume', type=bool, default=False, help='whether re-training from ckpt')
    parser.add_argument('--cuda', type=bool, default=True, help='whether training using cudatoolkit')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline for Deep Reinforcement Learning with Double Q-learning."""
    args = parse_args()
    pass


if __name__ == '__main__':
    main()
