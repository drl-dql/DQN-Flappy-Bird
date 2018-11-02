#!/bin/bash
# This script runs cs598 mp5 image ranking


# MODIFY THESE
declare training_file="main.py"
declare walltime="06:00:00"
declare jobname="cs598-mp4"
declare netid="zna2"
declare directory="~/scratch/image-ranking/src/"

for job in "${directory[@]}"
do
    python gen_pbs.py $training_file $walltime $jobname $netid $directory $job > job.pbs
    echo "Submitting $job"
    qsub run.pbs -A bauh
done
