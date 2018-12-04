#!/bin/bash
#PBS -l nodes=1:ppn=16:xk
#PBS -N ddqn-flappybird
#PBS -l walltime=48:00:00
#PBS -e $PBS_JOBNAME.$PBS_JOBID.err
#PBS -o $PBS_JOBNAME.$PBS_JOBID.out
#PBS -M netid@illinois.edu
source test2/bin/activate
cd ~/scratch/atari-game/src/
. /opt/modules/default/init/bash
module load bwpy/2.0.0-pre4
module load gcc/5.3.0
module load cudatoolkit
aprun -n 1 -N 1 python main.py --train=True