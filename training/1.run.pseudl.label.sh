#########################################################################
# File Name: 1.run.pseudl.label.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed 12 Feb 2025 07:32:42 AM UTC
#########################################################################
#!/bin/bash

#!/usr/bin/env bash

#--wandb_project "distil-whisper-labelling" \
#--report_to "wandb" \

lang="ja"
cache="/workspace/asr/distil-whisper/training/"

#accelerate launch run_pseudo_labelling.py \
#python -m ipdb run_pseudo_labelling.py \
#python run_pseudo_labelling.py \

#--concatenate_audio \
#accelerate launch run_pseudo_labelling.py \
#python -m ipdb run_pseudo_labelling.py \

accelerate launch run_pseudo_labelling.py \
  --model_name_or_path "openai/whisper-large-v3" \
  --dataset_name "mozilla-foundation/common_voice_16_1" \
  --dataset_config_name $lang \
  --dataset_split_name "train+validation+test" \
  --dataset_cache_dir $cache \
  --cache_dir $cache \
  --text_column_name "sentence" \
  --id_column_name "path" \
  --output_dir "./common_voice_16_1_${lang}_pseudo_labelled" \
  --per_device_eval_batch_size 64 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --logging_steps 500 \
  --max_label_length 256 \
  --preprocessing_batch_size 256 \
  --preprocessing_num_workers 8 \
  --dataloader_num_workers 8 \
  --language $lang \
  --task "transcribe" \
  --concatenate_audio \
  --return_timestamps \
  --streaming False \
  --report_to "none" \
  --generation_num_beams 1 
  #--push_to_hub
