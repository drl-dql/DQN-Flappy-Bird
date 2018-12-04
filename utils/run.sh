#!/bin/bash
# This script runs cs598 mp5 image ranking


# MODIFY THESE
declare training_file="main.py"
declare walltime="48:00:00"
declare jobname="ddqn-flappybird"
declare netid="netid"
declare directory="~/scratch/atari-game/src/"

for job in "${directory[@]}"
do
    python gen_pbs.py $training_file $walltime $jobname $netid $directory $job > job.pbs
    echo "Submitting $job"
    qsub run.pbs -A bauh
done
