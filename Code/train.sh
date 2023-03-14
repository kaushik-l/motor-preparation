#!/bin/sh
#
# Script for training thalamocortical models.
#
#SBATCH --account=theory         # Replace ACCOUNT with your group account name
#SBATCH --job-name=rflo          # The job name.
#SBATCH -c 4                     # The number of cpu cores to use
#SBATCH -N 1                     # The number of nodes to use
#SBATCH -t 0-1:00               # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=5gb        # The memory the job will use per cpu core

module load anaconda
# pip install torchvision
# conda install pytorch torchvision torchaudio -c pytorch
# conda install -c conda-forge numpy
# conda install -c conda-forge scipy
# conda install -c conda-forge scikit-learn
# conda install -c conda-forge matplotlib

#Command to execute Python program

python main_batch.py 'sine_randinit' $SLURM_ARRAY_TASK_ID

#End of script
