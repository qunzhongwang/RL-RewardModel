# 标准库导入
import os
import sys
import time
import random
import datetime
import tempfile
import io
from collections import deque
from functools import partial
from collections import defaultdict


# 第三方库导入
import numpy as np
import torch
import wandb
import requests
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
from peft import LoraConfig, get_peft_model, PeftModel
from bitsandbytes.optim import AdamW8bit
from undecorated import undecorated
from ml_collections import config_flags

# 自定义模块导入
import ddpo_pytorch.vlm_as_rm.rewards_Qwen
from rl_utils import make_collate_fn, _global_grad_norm, encodeAsPIL, process_to_IMG, get_uid, get_dataname, _get_streamed_dataset, get_streamed_dataset,find_ckpt

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
config_flags.DEFINE_config_file("config", "config/1gpu_bm.py", "benchmark configuration.")
flags.DEFINE_string("Project_name", None, "Override 'Project_name' in config.")
flags.DEFINE_bool("debug_ver", None, "Override 'DEBUG' in config.")
FLAGS = flags.FLAGS
FLAGS(sys.argv)

logger = get_logger(__name__)

def main(_):
    config = FLAGS.config
    if FLAGS.Project_name is not None:
        config.Project_name = FLAGS.Project_name
    if FLAGS.debug_ver is not None:
        config.debug_ver = FLAGS.debug_ver
    if config.debug_ver:
        config.Project_name = "DEBUG_NULL"
    if not config.log_cot:
        config.log_cot = True
    if not config.inference:
        config.inference = True

    #basic Accelerate and logging setup
    set_seed(config.seed)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16, 
        device_map= accelerator.device, 
        attn_implementation=config.pretrained.attn_implementation,
        cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
    )
    

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    processor.tokenizer.padding_side = "left"


    dataset_name = config.data_conf.dataset_url
    data_name = get_dataname(dataset_name) 

    if data_name == "pickscore_normal" or data_name == "HPD_v2":
        dataset = _get_streamed_dataset(dataset_name,config.data_conf.chunk_size)
    else:
        dataset = load_dataset(dataset_name, split="validation", num_proc=64)
    
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=make_collate_fn(processor, data_name))
    unique_id,ckptuid = get_uid()
    config.run_name += "_" + data_name + unique_id



    
    cumulative_sums = {
        "fomat correctness": 0.,
        "choose correctness": 0.,
        "avg lth": 0.,
        "avg reward": 0.,
        "avg reasoning": 0.,
        "global avg correctness":0.,
        "global avg reward":0.,
        "global reward std":0.,
        "pure loss":0.,
        "kl loss":0.,

    }

    cumulative_counts = {
        "fomat correctness": 0.,
        "choose correctness": 0.,
        "avg lth": 0.,
        "avg reward": 0.,
        "avg reasoning": 0.,
        "global avg correctness":0.,
        "global avg reward":0.,
        "global reward std":0.,
        "pure loss":0.,
        "kl loss":0.,
        }
        
    if config.resume_ckpt_id or config.resume_ckpt:
        gpus = config.get("resume_gpus", None) if config.get("resume_gpus", None) else accelerator.num_processes
        zeRO = config.get("resume_zeRO", None) if config.get("resume_zeRO", None) else config.deepspeed_stage
        lora_path = os.path.join("lora_log", "lora_" + config.resume_ckpt) if config.resume_ckpt else find_ckpt(config.loradir, config.resume_ckpt_id,mode="lora",gpus=gpus,zeRO=zeRO)
        if lora_path and os.path.exists(lora_path):
            model = PeftModel.from_pretrained(
                model, 
                lora_path,
                is_trainable=False
                )
        else:
            logger.info("No lora found. inference base model")

    #     checkpoint_path = os.path.join("ckpt_log", "checkpoint_" + config.resume_ckpt + ".pt") if config.resume_ckpt else find_ckpt(config.logdir, config.resume_ckpt, mode="ckpt",gpus=accelerator.num_processes,zeRO=config.deepspeed_stage)
    #     if checkpoint_path and os.path.exists(checkpoint_path):
    #         checkpoint = torch.load(checkpoint_path, map_location="cpu")
    #         logger.info(f"inference using checkpoint: {checkpoint_path}")
    #     else:
    #         logger.error("No ckpt found. You need to do inference with this code")
    # else:
    #     logger.error("No ckpt name given. You need to do inference with this code")

    model, loader = accelerator.prepare(model, loader)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.Project_name,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")
    os.makedirs(config.inference_basedir, exist_ok=True)
    inf_cache_dir = os.path.join(config.inference_basedir, config.run_name)
    os.makedirs(inf_cache_dir, exist_ok=True)
    reward_fn = getattr(ddpo_pytorch.vlm_as_rm.rewards_Qwen, config.reward.reward_pw_fn)()
    
    def reward_func(batch,accelerator):
        loss,retInfo = reward_fn(batch, toolbox= (model, processor,logger), accelerator=accelerator,config=config)
        return loss, retInfo

    correctness_queue = deque(maxlen=5)
    reward_queue = deque(maxlen=5)
    std_queue = deque(maxlen=5)


    model.eval()
    for idx, batch in tqdm(enumerate(loader)):
        inf_batch_dir = os.path.join(inf_cache_dir, f"{idx}_batch")
        os.makedirs(inf_batch_dir, exist_ok=True)
        config.curr_batch_dir = inf_batch_dir

        _,retInfo = reward_func(batch[0], accelerator)

        for key in retInfo:
            if isinstance(retInfo[key], torch.Tensor):
                retInfo[key] = retInfo[key].item()
            cumulative_sums[key] += retInfo[key]
            cumulative_counts[key] += 1
        correctness_queue.append(retInfo["global avg correctness"])
        reward_queue.append(retInfo["global avg reward"])
        std_queue.append(retInfo["global reward std"])


        if accelerator.is_main_process and idx % config.log_freq == 0:
            cumulative_avg_info = {
                key: cumulative_sums[key] / cumulative_counts[key]
                for key in cumulative_sums
                if cumulative_counts[key] != 0
            }
            cumulative_avg_info["global reward std"] = cumulative_avg_info["global reward std"]**(1/2)

            accelerator.log({
                "recent reward": sum(reward_queue) / len(reward_queue) if reward_queue else 0,
                "recent correctness": sum(correctness_queue) / len(correctness_queue) if correctness_queue else 0,
                "recent std": (sum(std_queue) / len(std_queue))**(1/2) if std_queue else 0,
                **cumulative_avg_info,
            },
            step=idx)
        
    if accelerator.is_main_process:
        cumulative_avg_info = {
                "final/"+key: cumulative_sums[key] / cumulative_counts[key]
                for key in cumulative_sums
                if cumulative_counts[key] != 0
            }
        cumulative_avg_info["final/global reward std"] = cumulative_avg_info["final/global reward std"]**(1/2)
        wandb.summary.update(cumulative_avg_info)


    return 0


if __name__ == "__main__":
    app.run(main)
