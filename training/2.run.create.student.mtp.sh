#########################################################################
# File Name: 2.run.create.student.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Feb 13 14:36:07 2025
#########################################################################
#!/bin/bash

#!/usr/bin/env bash

cache="/workspace/asr/distil-whisper/training/"

python -m ipdb create_student_model_mtp.py \
  --teacher_checkpoint "openai/whisper-large-v3" \
  --encoder_layers 4 \
  --decoder_layers 2 \
  --cache_dir $cache \
  --decoder_mtp_n 3 \
  --save_dir "./distil-large-v3-init-debug-mtp-parallel"
