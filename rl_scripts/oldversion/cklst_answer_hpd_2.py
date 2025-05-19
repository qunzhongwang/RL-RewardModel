from collections import defaultdict
import contextlib
import os
import sys
import datetime
import concurrent
from concurrent import futures
import time
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

from functools import partial
import tqdm
import tempfile
from PIL import Image

import numpy as np
import torch
import wandb
import ddpo_pytorch.rewards_Qwen
from ddpo_pytorch.rewards_Qwen import FullPmt

# from qwen_vl_utils import process_vision_info
import io
import base64


from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW


from trl import PPOTrainer, PPOConfig
import copy

# from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
# from diffusers.loaders import AttnProcsLayers
# from diffusers.models.attention_processor import LoRAAttnProcessor
# from src.datasets import CLIPBTDatasetConfig, CLIPBTDataset

# import ddpo_pytorch.prompts
# import ddpo_pytorch.rewards

# from ddpo_pytorch.stat_tracking import PerPromptStatTracker
# from ddpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
# from ddpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob


import openai
import requests

import random
from torchvision import transforms
from datasets import load_from_disk, load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from threading import Lock
import base64
from io import BytesIO


from undecorated import undecorated
from types import MethodType
from bitsandbytes.optim import AdamW8bit

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")
flags.DEFINE_bool("WithTie", None, "Override 'WithTie' in config.")
flags.DEFINE_string("dataset_name", None, "Override 'dataset_name' in config.")
flags.DEFINE_string("Project_name", None, "Override 'Project_name' in config.")
flags.DEFINE_string("pmtcnt", None, "Override 'pmtcnt' in config.")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logger = get_logger(__name__)

def process_to_IMG(imgbytes):
    image_file = BytesIO(imgbytes)
    img = Image.open(image_file)
    return img


def encodeAsPIL(imagelist, target_size=512,  as_base64=False):
        images = []
        total_width = 0
        max_height = 0

        for image_data in imagelist:
            img = image_data.convert("RGB")
            original_width, original_height = img.size

            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
            img = img.resize((new_width, new_height))  # Resize the image
            images.append(img)

            total_width += new_width
            max_height = max(max_height, new_height)

        # Create a blank combined image
        combined_image = Image.new('RGB', (total_width + 5 * (len(imagelist) - 1), max_height))

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 5

            # If as_base64 is True, return Base64-encoded string
        if as_base64:
            img_byte_arr = io.BytesIO()
            combined_image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)  # Move to the beginning of the BytesIO stream
            img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
            return img_base64

        # Otherwise, return the PIL Image object
        return combined_image


def make_collate_fn(processor, data_name):
    def collate(batch):    
        #保证胜负 随机inverse
        for sample in batch:
            if data_name == "HPD_v2":
                sample["caption"] = sample["prompt"]
                sample['label_0'] = sample["human_preference"][0]
                sample['label_1'] = sample["human_preference"][1]
                sample["jpg_0"] = sample["image"][0]["bytes"]
                sample["jpg_1"] = sample["image"][1]["bytes"]
            
            if sample['label_0'] > sample['label_1']:
                sample["jpg_0"], sample["jpg_1"] = sample["jpg_0"], sample["jpg_1"]
                
            else:
                sample["jpg_0"], sample["jpg_1"] = sample["jpg_1"],sample["jpg_0"]

            if random.random() > 0.5:
                lpic, rpic, inv = "jpg_0", "jpg_1", 0
            else:
                lpic, rpic, inv = "jpg_1", "jpg_0", 1

            sample["lpic"] = sample[lpic]
            sample["rpic"] = sample[rpic]
            sample["inv"] = inv

            if data_name == "pickscore" or data_name == "pickscore_normal" or data_name =="HPD_v2":
                sample["lpic"] = process_to_IMG(sample["lpic"])
                sample["rpic"] = process_to_IMG(sample["rpic"])
            
        imgs   = []
        prompts= []
        invs = []
        for sample in batch:
            gross_img  = encodeAsPIL([sample["lpic"],sample["rpic"]])
            imgs.append(gross_img)
            # 同一段文字 prompt
            msg = [{
                "role":"user",
                "content":[
                    {"type":"image"},
                    {"type":"text","text":FullPmt.format(locPrompt=sample["caption"])}
                ]
            }]
            prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True))
            invs.append(sample["inv"])
        # tokenizer 会把 <image> placeholder 插进去
        model_inputs = processor(text=prompts,
                                images=imgs,
                                padding=True,
                                return_tensors="pt")
        # for key, value in model_inputs.items():
        #     if isinstance(value, torch.Tensor) and torch.is_floating_point(value):  # 检查是否为浮点张量
        #         model_inputs[key] = value.requires_grad_()
        #         print(f"[ok] {key} is now requires_grad=True")
        #         print(f"[ok] {key} is now requires_grad=True")
        #         break  # 如果只需要处理第一个满足条件的张量，则可以跳出循环
        model_inputs["invs"] = invs
        return model_inputs, batch          # 附带原始样本用于 reward
    return collate


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
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto", 
        cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
    )
    

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-7B-Instruct", 
    #     torch_dtype="auto", 
    #     device_map="auto", 
    #     cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen2.5-VL-7B-Instruct",
    # )
    # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    lora_config = LoraConfig(
        r=8,              # 低秩矩阵的秩
        lora_alpha=32,    # LoRA 的缩放因子
        target_modules=["q_proj", "v_proj"],  # 指定 LoRA 作用的目标模块（如注意力层）
        lora_dropout=0.1, # LoRA 的 dropout 概率
        bias="none",      # 是否调整模型的偏置
        task_type="CAUSAL_LM"  # LoRA 的任务类型（序列到序列生成任务）
    )

    model = get_peft_model(model, lora_config)

    optimizer = AdamW(model.parameters(), lr=1e-4)

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
        start_idx = random.randint(1000, 10000)
        chunk_size = 500  
        buffer = []
        cnt = random.randint(5, 10)
        size = 0
        for idx, example in enumerate(dataset):
            if idx < start_idx:
                continue 
            if size < chunk_size:
                if cnt == 0:
                    buffer.append(example)
                    cnt = random.randint(5, 10)
                    size += 1
                else:
                    cnt -= 1
            else:
                break
            
        dataset = Dataset.from_list(buffer) 
    else:
        dataset = load_dataset(dataset_name, split="validation", num_proc=64)
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=make_collate_fn(processor, data_name))

    # number of timesteps within each trajectory to train on
    #num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
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
    config.run_name = "rl"+data_name+unique_id

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

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    model.gradient_checkpointing_enable()

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)


    PROJECT_NAME = "Qwen_RL"#config.Project_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=PROJECT_NAME,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )

    logger.info(f"\n{config}")
    random.seed(config.seed)
    

    # if config.WithTie:
    #     config.reward_pw_fn = "evaluate_promptImagePairDirectAnswerWithTie_GPT4"
    # else:
    #     config.reward_pw_fn = "evaluate_promptImagePairDirectAnswer_GPT4"
    
    config.reward_pw_fn = "evaluate_QwenVL2_7B_P"

    reward_fn = getattr(ddpo_pytorch.rewards_Qwen, config.reward_pw_fn)()

    columns = ["ANS",]
    table = wandb.Table(columns=columns)
    

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

    f = open(os.path.join(basedir, basefiledir, outf), "w")
    error = open(os.path.join(basedir, baseerrordir, errorf), "w")
    cache = open(os.path.join(basedir, basecachedir, cachef), "w")

    ppoDict = {
        "gamma": 0.99,  # 折扣因子
        "clip_epsilon": 0.2,  # PPO clipping epsilon
        "entropy_coef": 0.01,  # 熵正则化系数
        "learning_rate": 1e-5,  # 学习率
        "value_loss_coef": 0.5,  # 值函数损失系数
    }

    def reward_func(batch,accelerator):
        
        RECORD = True

        loss,retInfo = reward_fn(batch, Error=error, ANALYSIS=cache, RECORD=RECORD, cklth=cklstpmt, toolbox=toolbox,ppoDict=ppoDict, accelerator=accelerator)

        return loss, retInfo

    for epoch in tqdm(range(config.num_epochs)):
        model.train()

        cumulative_sums = {
            "fomat correctness": 0,
            "choose correctness": 0,
            "avg lth": 0, 
            "avg reward": 0,
            "avg reasoning": 0,
#            "avg check point count": 0,
        }
        cumulative_counts = {
            "fomat correctness": 0,
            "choose correctness": 0,
            "avg lth": 0,
            "avg reward": 0,
            "avg reasoning": 0,
#            "avg check point count": 0,
            }

        for batch in tqdm(loader):
            with accelerator.accumulate(model):
                loss,retInfo = reward_func(batch[0],accelerator)

                for key in retInfo:
                    cumulative_sums[key] += retInfo[key]
                    cumulative_counts[key] += 1

                optimizer.zero_grad()
                
                accelerator.backward(loss)
                #print(torch.cuda.memory_summary())
                optimizer.step()

                cumulative_avg_info = {
                    key: cumulative_sums[key] / cumulative_counts[key] for key in cumulative_sums
                }

            wandb.log({
                **cumulative_avg_info, 
                })
    return 0


if __name__ == "__main__":
    app.run(main)
