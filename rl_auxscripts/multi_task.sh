#!/bin/bash

CUDA_VISIBLE_DEVICES=0 accelerate launch rl_scripts/cklst_answer_hpd_2.py --task_id 0 &
CUDA_VISIBLE_DEVICES=1 accelerate launch rl_scripts/cklst_answer_hpd_2.py --task_id 1 &
CUDA_VISIBLE_DEVICES=2 accelerate launch rl_scripts/cklst_answer_hpd_2.py --task_id 2 &
CUDA_VISIBLE_DEVICES=3 accelerate launch rl_scripts/cklst_answer_hpd_2.py --task_id 3 &