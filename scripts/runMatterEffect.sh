#!/bin/bash

# get the initial and final values and number of array jobs as inputs
initial_value=$1
final_value=$2
number_of_array_jobs=$3

# ensure the number of array jobs is a positive integer
if ! [[ "$number_of_array_jobs" =~ ^[0-9]+$ ]]
then
  echo "Error: The number of array jobs must be a positive integer."
  exit 1
fi

array_range="0-$((number_of_array_jobs - 1))"

# create a temp submission script 
cp MatterEffect_submission.sh MatterEffect_sub_temp.sh
# replace the number of array jobs in the submission script
sed -i "s/<number_of_array_jobs>/$array_range/g" MatterEffect_sub_temp.sh

# submit the submission script with the initial and final values as inputs
sbatch MatterEffect_sub_temp.sh $initial_value $final_value $number_of_array_jobs
