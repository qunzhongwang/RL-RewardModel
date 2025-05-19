#!/bin/bash

for i in {0..3}; do
    CUDA_VISIBLE_DEVICES=$i accelerate launch rl_scripts/cklst_answer_hpd_2.py --task_id $i &
done

wait