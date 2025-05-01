#!/bin/bash
#
# DEBUG CONF
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20G
#SBATCH --time=00:30:00

# PLOT CONF
##SBATCH --partition=gpu_a100
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=64
##SBATCH --mem=40G
##SBATCH --time=02:00:00

#SBATCH --output=results/plot/%j_stdout.txt
#SBATCH --error=results/plot/%j_stderr.txt
#SBATCH --job-name=plot
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw7

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

CODE_DIR=code
CONFIG_DIR=configs

python ${CODE_DIR}/plot.py \
       @${CONFIG_DIR}/oscer.txt \
       @${CONFIG_DIR}/exp.txt \
       @${CONFIG_DIR}/net.txt \
       --results_path "./results/plot/" \
