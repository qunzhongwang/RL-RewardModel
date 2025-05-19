#!/usr/bin/env zsh

source /m2v_intern/wangqunzhong/miniconda3/etc/profile.d/conda.sh
cd /m2v_intern/wangqunzhong/research/ddpo-pytorch
conda activate cklst
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 rl_scripts/cklst_answer_hpd_debug.py