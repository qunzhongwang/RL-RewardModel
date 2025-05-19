# 标准库导入
import os
import sys
import time
import random
import datetime
import tempfile
import io
from functools import partial
from collections import defaultdict

# 第三方库导入
import numpy as np
import torch
import wandb
import requests
import openai
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW

import tqdm
from datasets import Dataset, load_dataset, load_from_disk
from absl import app, flags
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, ProjectConfiguration, FP8RecipeKwargs
from accelerate.logging import get_logger
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
from undecorated import undecorated
from ml_collections import config_flags

# 自定义模块导入
import ddpo_pytorch.rewards_Qwen
from rl_utils import make_collate_fn, _global_grad_norm, encodeAsPIL, process_to_IMG

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_bool("WithTie", None, "Override 'WithTie' in config.")
flags.DEFINE_string("dataset_name", None, "Override 'dataset_name' in config.")
flags.DEFINE_string("Project_name", None, "Override 'Project_name' in config.")
flags.DEFINE_string("pmtcnt", None, "Override 'pmtcnt' in config.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logger = get_logger(__name__)

FP8_RECIPE_KWARGS = {"fp8_format": "HYBRID", "amax_history_len": 32, "amax_compute_algo": "max"}
kwargs = [FP8RecipeKwargs(backend="TE", **FP8_RECIPE_KWARGS)]


def main(_):
    config = FLAGS.config
    if FLAGS.WithTie is not None:
        config.WithTie = FLAGS.WithTie
    if FLAGS.dataset_name is not None:
        config.dataset_name = FLAGS.dataset_name
    if FLAGS.Project_name is not None:
        config.Project_name = FLAGS.Project_name
    if FLAGS.pmtcnt is not None:
        config.pmtcnt = FLAGS.pmtcnt
    

    # basic Accelerate and logging setup
    set_seed(config.seed)
    if config.deepspeed_stage != 3:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=config.deepspeed_stage,  # ZeRO Stage 选择
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            #bf16=True,
        )
    else:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=3,  # 使用 ZeRO Stage 3
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            # offload_optimizer_device="none",  # 将优化器分区到 CPU，可选值为 "cpu" 或 "nvme"
            # offload_param_device="none",      # 将参数分区到 CPU，可选值为 "cpu" 或 "nvme"

        )
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        # kwargs_handlers=kwargs,
    )
    if config.deepspeed_stage != 3:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map= accelerator.device, 
            cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=False,
            cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
        )
    

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    lora_config = LoraConfig(
        r=8,              # 低秩矩阵的秩
        lora_alpha=32,    # LoRA 的缩放因子
        target_modules=["q_proj", "v_proj"],  # 指定 LoRA 作用的目标模块（如注意力层）
        lora_dropout=0.1, # LoRA 的 dropout 概率
        bias="none",      # 是否调整模型的偏置
        task_type="CAUSAL_LM"  # LoRA 的任务类型（序列到序列生成任务）
    )

    model = get_peft_model(model, lora_config)

    optimizer = AdamW8bit(model.parameters(), lr=1e-4)

    toolbox = (model, processor)
    
    # dataset_name = config.dataset_name
    dataset_name = "/m2v_intern/wangqunzhong/research/huggingface/dataset/ymhao/HPD_v2"
    if dataset_name == "/m2v_intern/liujie/research/huggingface/dataset/yuvalkirstain/pickapic_v1_unique":
        data_name = "pickscore"
    elif dataset_name == "/m2v_intern/liujie/research/huggingface/dataset/imagereward/fidelity_rating_dataset":
        data_name = "imagereward"
    elif dataset_name == "/m2v_intern/liujie/research/huggingface/dataset/yuvalkirstain/pickapic_v1":
        data_name = "pickscore_normal"
    elif dataset_name == "/m2v_intern/wangqunzhong/research/huggingface/dataset/ymhao/HPD_v2":
        data_name = "HPD_v2"
    if data_name == "pickscore_normal" or data_name == "HPD_v2":
        dataset = load_dataset(dataset_name, split="train", streaming = True)
        start_idx = random.randint(200, 2000)
        chunk_size = 1800  
        buffer = []
        cnt = random.randint(2, 5)
        size = 0
        for idx, example in enumerate(dataset):
            if idx < start_idx:
                continue 
            if size < chunk_size:
                if cnt == 0:
                    buffer.append(example)
                    cnt = random.randint(2, 5)
                    size += 1
                else:
                    cnt -= 1
            else:
                break
        dataset = Dataset.from_list(buffer) 
    else:
        dataset = load_dataset(dataset_name, split="validation", num_proc=64)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=make_collate_fn(processor, data_name))
    batch_size = 2
    if (config.pmtcnt == "0") or (config.pmtcnt == "3") or (config.pmtcnt == "5") or (config.pmtcnt == "9") or (config.pmtcnt == "13"):
        cklstpmt = "4-22-"+config.pmtcnt+"question"
    else:
        cklstpmt = "4-22-3question"

    now = datetime.datetime.now()

    formatted_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    unique_id = formatted_time
    config.WithTie=True

    appendinfo = "woT_" + config.pmtcnt + "_"
    if config.WithTie:
        appendinfo = "wT_" + config.pmtcnt + "_"
    outputfile_name = "output" + appendinfo + unique_id
    config.run_name = "Answer_" + appendinfo + config.run_name

    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + data_name + unique_id

    #TODO
    outputfile_name = "rl_cklst"
    config.run_name = "rl"+"_baseline_"+unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    
    #model.gradient_checkpointing_enable()

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)


    PROJECT_NAME = "Qwen_RL_train"#config.Project_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=PROJECT_NAME,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )

    logger.info(f"\n{config}")
    # if config.WithTie:
    #     config.reward_pw_fn = "evaluate_promptImagePairDirectAnswerWithTie_GPT4"
    # else:
    #     config.reward_pw_fn = "evaluate_promptImagePairDirectAnswer_GPT4"
    
    config.reward_pw_fn = "evaluate_QwenVL2_7B_BL"

    reward_fn = getattr(ddpo_pytorch.rewards_Qwen, config.reward_pw_fn)()
    
    basedir = "logfile"
    os.makedirs(basedir, exist_ok=True)
    outf = outputfile_name+".txt"
    errorf = outputfile_name+"_error.txt"
    cachef = outputfile_name+"_cahce.txt"
    basefiledir = "answer_outfile"
    baseerrordir = "answer_errorfile"
    basecachedir = "answer_cahefile"
    os.makedirs(os.path.join(basedir,basefiledir), exist_ok=True)
    os.makedirs(os.path.join(basedir,baseerrordir), exist_ok=True)
    os.makedirs(os.path.join(basedir,basecachedir), exist_ok=True)

    # f = open(os.path.join(basedir, basefiledir, outf), "w")
    # error = open(os.path.join(basedir, baseerrordir, errorf), "w")
    # cache = open(os.path.join(basedir, basecachedir, cachef), "w")

    ppoDict = {
        "gamma": 0.99,  # 折扣因子
        "clip_epsilon": 0.2,  # PPO clipping epsilon
        "entropy_coef": 0.01,  # 熵正则化系数
        "learning_rate": 1e-5,  # 学习率
        "value_loss_coef": 0.5,  # 值函数损失系数
    }

    def reward_func(batch,accelerator):
        
        RECORD = True

        loss,retInfo = reward_fn(batch, cklth=cklstpmt, toolbox=toolbox,ppoDict=ppoDict, accelerator=accelerator)

        return loss, retInfo

    cumulative_sums = {
        "fomat correctness": 0,
        "choose correctness": 0,
        "avg lth": 0, 
        "avg reward": 0,
        "avg reasoning": 0,
    #"avg check point count": 0,
    }
    cumulative_counts = {
        "fomat correctness": 0,
        "choose correctness": 0,
        "avg lth": 0,
        "avg reward": 0,
        "avg reasoning": 0,
    #"avg check point count": 0,
        }
    
    for epoch in tqdm(range(config.num_epochs)):
        model.train()
        tmp_grad = []
        iter_lth = int(len(loader)/batch_size)
        for idx, batch in tqdm(enumerate(loader)):
            loss,retInfo = reward_func(batch[0],accelerator)

            for key in retInfo:
                cumulative_sums[key] += retInfo[key]
                cumulative_counts[key] += 1
            
            if config.inference:
                optimizer.zero_grad()
                #print(f"[ok] begin")
                #print(torch.cuda.memory_summary())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (p for p in model.parameters() if p.requires_grad)
                    accelerator.clip_grad_norm_(params_to_clip, config.train.max_grad_norm)
                
                tmp_grad.append(_global_grad_norm(model.parameters()))
                optimizer.step()

            if idx %5 != 0:
                accelerator.log({
                    "loss": loss.item(),
                },
                step=iter_lth * epoch + idx,
                )
            
            # 每隔 10 个 batch 记录一次日志（仅主进程）
            if accelerator.is_main_process and idx % 5 == 0:
                cumulative_avg_info = {
                    key: cumulative_sums[key] / cumulative_counts[key] for key in cumulative_sums
                }
                accelerator.log({
                    **cumulative_avg_info,
                    "loss": loss.item(),  # 记录当前 loss
                    # "grad norm":sum(tmp_grad) / len(tmp_grad) if tmp_grad else 0,
                    #"step": idx           # 当前 step
                },
                step=iter_lth * epoch + idx,#global_step,
                )
                tmp_grad.clear()

    return 0


if __name__ == "__main__":
    app.run(main)
