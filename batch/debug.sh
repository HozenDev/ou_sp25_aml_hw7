#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=results/debug/stdout.txt
#SBATCH --error=results/debug/stderr.txt
#SBATCH --time=00:30:00
#SBATCH --job-name=debug
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw7
#SBATCH --array=0

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

CODE_DIR=code
CONFIG_DIR=configs

## SHALLOW
python ${CODE_DIR}/main.py \
       @${CONFIG_DIR}/oscer.txt \
       @${CONFIG_DIR}/exp.txt \
       @${CONFIG_DIR}/net.txt --label NET \
       --exp_index $SLURM_ARRAY_TASK_ID \
       --cpus_per_task $SLURM_CPUS_PER_TASK \
       --save_model --render \
       --results_path "./results/exp/" \
       -vvv
