#!/bin/bash

#PBS -l walltime=14400
#PBS -l nodes=1:ppn=${snakemake[threads]}:gen8
#PBS -A cnm83860

#PBS -m ea

# change into the directory where qsub will be executed
cd $PBS_O_WORKDIR

module load cuda
module load gcc

export NUMBA_NUM_THREADS=$PBS_NP
export MKL_NUM_THREADS=$PBS_NP
export OMP_NUM_THREADS=$PBS_NP

export NUMBA_THREADING_LAYER=workqueue
envs/venv/bin/python ${snakemake_params[code]}
