#!/bin/bash -l
#PBS -N rot_inv_echo
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB
#PBS -l walltime=24:00:00
#PBS -l gpu_type=v100
#PBS -A NAML0001
#PBS -q casper
#PBS -o out
#PBS -e out
source ~/.bashrc
conda activate storm-mode
echo-run hyperparameter.yaml conf.yaml -n $PBS_JOBID
