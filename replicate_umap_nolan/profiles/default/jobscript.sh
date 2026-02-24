#!/bin/bash

#PBS -A cnm83860
#PBS -l walltime={resources.walltime}
#PBS -l nodes=1:ppn={threads}:{resources.arch}
#PBS -m ea

# change into the directory where qsub will be executed
cd $PBS_O_WORKDIR/workflow

module load cuda
module load gcc

export NUMBA_NUM_THREADS=$PBS_NP
export MKL_NUM_THREADS=$PBS_NP
export OMP_NUM_THREADS=$PBS_NP
export NUMBA_THREADING_LAYER=workqueue

{exec job}
