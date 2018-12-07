#!/bin/bash

TEXT_RESET='\e[0m'
TEXT_CYAN='\e[1;36m'
TEXT_PURPLE='\e[1;35m'

echo -e $TEXT_PURPLE
echo 'Setting up the DQN Flappy Bird Environment ...'
echo -e $TEXT_RESET

bash utils/pytorch041_cuda92_colab.sh
echo -e $TEXT_CYAN
echo 'pytorch041_cuda92_colab finished..'
echo -e $TEXT_RESET

pip install visdom
echo -e $TEXT_CYAN
echo 'Visdom python package setup done..'
echo -e $TEXT_RESET

pip install pygame
echo -e $TEXT_CYAN
echo 'Pygame python package setup done..'
echo -e $TEXT_RESET

pip install moviepy
echo -e $TEXT_CYAN
echo 'Moviepy python package setup done..'
echo -e $TEXT_RESET

git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .
cd ..
echo -e $TEXT_CYAN
echo 'Pygame Learning environment (PLE) is successfully configured..'
echo -e $TEXT_RESET

echo -e $TEXT_PURPLE
echo 'Environment configuration finished'
echo -e $TEXT_RESET
