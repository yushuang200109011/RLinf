## --------------- Prepare Environment ---------------
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

## --------------- LIBERO ---------------
# pi0
python toolkits/eval_scripts_openpi/libero_eval.py \
    --exp_name libero_spatial_pi0 \
    --config_name pi0_libero \
    --pretrained_path your_model_path/ \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10

# pi05
python toolkits/eval_scripts_openpi/libero_eval.py \
    --exp_name libero_spatial_pi05 \
    --config_name pi05_libero \
    --pretrained_path your_model_path/ \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10

