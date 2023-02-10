#!/bin/bash
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --job-name=MatterEffect
#SBATCH --array=<number_of_array_jobs>
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:20
#SBATCH --output=slurm_outputs/MatterEffect_%A_%a.out
#SBATCH --error=slurm_outputs/MatterEffect_%A_%a.error

# load environment
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.8.10/site-packages

# get the initial and final values and number of array jobs as inputs to the script
initial_value=$1
final_value=$2
number_of_array_jobs=$3

# calculate the input for each job
input=$(echo "$initial_value + ($SLURM_ARRAY_TASK_ID * ($final_value - $initial_value)) / ($number_of_array_jobs - 1)" | bc -l)

# run the script. Inputs are E, npoints and savefile
python MatterEffect_cluster.py ${input} 1000 "../oscillationProbs/matterVanilla_${SLURM_ARRAY_TASK_ID}.txt"
