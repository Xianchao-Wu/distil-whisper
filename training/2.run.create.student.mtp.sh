#########################################################################
# File Name: 2.run.create.student.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Feb 13 14:36:07 2025
#########################################################################
#!/bin/bash

#!/usr/bin/env bash

cache="/workspace/asr/distil-whisper/training/"

enlayers=32
delayers=2
mtp_type="parallel" # "causal"
mtp_n=2 # next n token prediction

outdir="./distil-large-v3-init-debug-mtp-en${enlayers}-de${delayers}-${mtp_type}-n${mtp_n}"

python -m ipdb create_student_model_mtp.py \
  --teacher_checkpoint "openai/whisper-large-v3" \
  --encoder_layers ${enlayers} \
  --decoder_layers ${delayers} \
  --cache_dir $cache \
  --decoder_mtp_n ${mtp_n} \
  --decoder_mtp_type ${mtp_type} \
  --save_dir ${outdir} 

