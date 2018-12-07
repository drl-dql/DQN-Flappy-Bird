#!/bin/bash

echo "Running Setup ..."

utils/pytorch041_cuda92_colab.sh
pip install visdom
pip install pygame
pip install moviepy
git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
cd ..

