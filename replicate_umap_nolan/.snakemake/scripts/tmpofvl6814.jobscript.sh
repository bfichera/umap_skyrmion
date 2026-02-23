#!/bin/bash

declare -A snakemake_input=( )
declare -A snakemake_output=( [0]="output.pdf" )
declare -A snakemake_params=( )
declare -A snakemake_wildcards=( )
declare -A snakemake_resources=( [0]="1" [_cores]="1" [1]="1" [_nodes]="1" [2]="1000" [mem_mb]="1000" [3]="954" [mem_mib]="954" [4]="1000" [disk_mb]="1000" [5]="954" [disk_mib]="954" [6]="/tmp/3642233.sched5.carboncluster" [tmpdir]="/tmp/3642233.sched5.carboncluster" )
declare -A snakemake_log=( )
declare -A snakemake_config=( )
declare -A snakemake=( [threads]="1" [rule]="all" [bench_iteration]="None" [scriptdir]="/home/bfichera/data/projects/umap_skyrmion/replicate_umap_nolan/scripts" )

#PBS -l walltime=14400
#PBS -l nodes=1:ppn=32:gen8
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
venv/bin/python scripts/run.py
