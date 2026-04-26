#!/bin/bash


# Define your permutations
heads=(16 32 64 128 256 512)
# heads=(256)
out_dirs=("output/run16" "output/run32" "output/run64" "output/128" "output/run256" "output/run512")
# out_dirs=("output/run256")

# Loop through and submit
for i in "${!heads[@]}"; do
    sbatch --export=SAVE_PATH=${out_dirs[$i]},HEAD_LAYER=${heads[$i]} train.slurm
done

# for i in "${!heads[@]}"; do
#     echo "Running iteration $i: Save=${out_dirs[$i]}, Head=${heads[$i]}"

#     # Set the variables for this specific execution
#     export SAVE_PATH=${out_dirs[$i]}
#     export HEAD_LAYER=${heads[$i]}

#     sh train.slurm
# done