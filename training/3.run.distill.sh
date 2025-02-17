#########################################################################
# File Name: 3.run.distill.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Feb 14 09:09:07 2025
#########################################################################
#!/bin/bash

cache="/workspace/asr/distil-whisper/training/"
eval_dataset_name="./common_voice_16_1_ja_pseudo_labelled_good" 

outdir=cache+"/distill_output"

save_steps=200

#--overwrite_output_dir \

#python -m ipdb run_distillation.py \
accelerate launch run_distillation.py \
  --model_name_or_path "./distil-large-v3-init" \
  --cache_dir $cache \
  --dataset_cache_dir $cache \
  --teacher_model_name_or_path "openai/whisper-large-v3" \
  --train_dataset_name "${eval_dataset_name}+${eval_dataset_name}" \
  --train_split_name "train+validation" \
  --text_column_name "sentence+sentence" \
  --train_dataset_samples "7+4" \
  --eval_dataset_name ${eval_dataset_name} \
  --eval_split_name "test" \
  --eval_text_column_name "sentence" \
  --eval_steps ${save_steps} \
  --save_steps ${save_steps} \
  --warmup_steps 50 \
  --learning_rate 0.0001 \
  --lr_scheduler_type "constant_with_warmup" \
  --timestamp_probability 0.2 \
  --condition_on_prev_probability 0.2 \
  --language "ja" \
  --task "transcribe" \
  --logging_steps 25 \
  --save_total_limit 1 \
  --max_steps 150000 \
  --wer_threshold 20 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --dataloader_num_workers 8 \
  --preprocessing_num_workers 8 \
  --ddp_timeout 7200 \
  --dtype "bfloat16" \
  --attn_implementation "sdpa" \
  --output_dir ${outdir} \
  --do_train \
  --do_eval \
  --report_to "none" \
  --gradient_checkpointing \
  --predict_with_generate \
  --freeze_encoder \
  --freeze_embed_positions \
  --streaming False #\
  #--push_to_hub

