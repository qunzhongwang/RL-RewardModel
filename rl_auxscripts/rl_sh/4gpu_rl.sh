#!/usr/bin/env zsh

MAINSCRIP="rl_scripts/rl_pipeline.py"
BASEJSON="config/acc_config/"
BASECONF="config/"
# source /m2v_intern/wangqunzhong/miniconda3/etc/profile.d/conda.sh
cd /m2v_intern/wangqunzhong/research/ddpo-pytorch
conda activate cklst
source rl_auxscripts/export_sh.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file "${BASEJSON}4gpu_ds.json" $MAINSCRIP --config ${BASECONF}4gpu_z2.py "$@"