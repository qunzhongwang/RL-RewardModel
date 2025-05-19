#!/usr/bin/env zsh


DEBUG_FLAG=""

for arg in "$@"; do
  case $arg in
    --debug)
      DEBUG_FLAG="--debug_ver=True"
      shift
      ;;
    *)
      ;;
  esac
done

MAINSCRIP="rl_scripts/rl_pipeline.py"
BASEJSON="config/acc_config/"
BASECONF="config/"
# source /m2v_intern/wangqunzhong/miniconda3/etc/profile.d/conda.sh
cd /m2v_intern/wangqunzhong/research/ddpo-pytorch
conda activate cklst
source rl_auxscripts/export_sh.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch --config_file "${BASEJSON}6gpu_ds.json" $MAINSCRIP --config ${BASECONF}6gpu_z2.py $DEBUG_FLAG
