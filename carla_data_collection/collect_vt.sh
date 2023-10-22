#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/nianyli/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/nianyli/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/nianyli/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/nianyli/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate interfuser

python --version

# set your dataset collection variables
export WORLD=town01
export NUM_FRAMES=30000
export DATASET_PATH=/home/nianyli/Desktop/code/thesis/DiffViewTrans/data/town01_vt_dataset

# restart the world, generate traffic in the world, begin the data collection script
# this is done because the world tends to crash if you collect too much data in one
# run of the script. this way, the carla server resets and there is less chance of
# crashing
for i in {1..10}
do
    python restart_carla_world.py --world=${WORLD} 
    python generate_traffic.py
    python collect_view_translation_data_demo.py --num_frames=${NUM_FRAMES} --dataset_path=${DATASET_PATH}
done

