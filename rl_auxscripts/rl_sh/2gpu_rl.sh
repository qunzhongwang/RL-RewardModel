#!/bin/bash

cd /m2v_intern/wangqunzhong/research/ddpo-pytorch
conda activate cklst
source rl_auxscripts/export_sh.sh
export CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --config_file config/acc_config/2gpu_ds.json  rl_scripts/cklst_answer_hpd_3.py
