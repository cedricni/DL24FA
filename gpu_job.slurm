#!/bin/sh
#SBATCH --job-name=JEPA
#SBATCH --account=csci_ga_2572-2024fa
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1

singularity exec --bind /scratch --nv --overlay /scratch/[netid]/overlay-25GB-500K.ext3:rw /scratch/[netid]/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /scratch/tn2354/DL24FA
python ./trainer.py