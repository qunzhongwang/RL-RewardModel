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

from qwen_vl_utils import process_vision_info
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



def main(_):
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
    # generate_with_grad = undecorated(model.__class__.generate)  # 通过实例的类获取
    # model.generate_with_grad = MethodType(generate_with_grad, model)

    # 递归解除装饰器（彻底移除 @torch.no_grad）
    # def undecorate_method(method):
    #     while hasattr(method, '__wrapped__'):
    #         method = method.__wrapped__
    #     return method

    # # 获取原始 generate 方法并解除装饰
    # original_generate = undecorate_method(model.generate.__func__)
    # model.generate_with_grad = MethodType(original_generate, model)

    # def custom_generate(self, input_ids, **kwargs):
    #     # 强制启用梯度计算环境
    #     with torch.enable_grad():
    #         # 禁用缓存系统（缓存会导致计算图断裂）
    #         kwargs.update({"use_cache": False})
            
    #         # 执行原始生成逻辑
    #         return self.generate_with_grad(
    #             input_ids=input_ids,
    #             **kwargs
    #         )

    # # 绑定自定义生成方法
    # model.custom_generate = MethodType(custom_generate, model)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    toolbox = (model, processor)

    config = FLAGS.config
    if FLAGS.WithTie is not None:
        config.WithTie = FLAGS.WithTie
    if FLAGS.dataset_name is not None:
        config.dataset_name = FLAGS.dataset_name
    if FLAGS.Project_name is not None:
        config.Project_name = FLAGS.Project_name
    if FLAGS.pmtcnt is not None:
        config.pmtcnt = FLAGS.pmtcnt
    
    if (config.pmtcnt == "0") or (config.pmtcnt == "3") or (config.pmtcnt == "5") or (config.pmtcnt == "9") or (config.pmtcnt == "13"):
        cklstpmt = "4-22-"+config.pmtcnt+"question"
    else:
        cklstpmt = "4-22-3question"

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
    
    config.run_name = "RL_CKLST"

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

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

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
        gradient_accumulation_steps=config.train.gradient_accumulation_steps
        * num_train_timesteps,
    )

    PROJECT_NAME = "DEBUG"#config.Project_name
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=PROJECT_NAME,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )

    logger.info(f"\n{config}")


    random.seed(config.seed)
    
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
    
    dataset_length = len(dataset)
    index_list = list(range(0, dataset_length))

    random.shuffle(index_list)
    
    score_chosen = "label_0"
    score_rejected = "label_1"

    chosen = "jpg_0"
    rejected = "jpg_1"

    # if config.WithTie:
    #     config.reward_pw_fn = "evaluate_promptImagePairDirectAnswerWithTie_GPT4"
    # else:
    #     config.reward_pw_fn = "evaluate_promptImagePairDirectAnswer_GPT4"
    config.reward_pw_fn = "evaluate_QwenVL2_7B"

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
    
    def rwd_calc(tup):
        
        sample,_ = tup
        sample["_"] = _

        if data_name == "HPD_v2":
            sample["caption"] = sample["prompt"]
            sample['label_0'] = sample["human_preference"][0]
            sample['label_1'] = sample["human_preference"][1]
            sample["jpg_0"] = sample["image"][0]["bytes"]
            sample["jpg_1"] = sample["image"][1]["bytes"]


        if sample['label_0'] > sample['label_1']:
            chosen, rejected = "jpg_0", "jpg_1"
        else:
            chosen, rejected = "jpg_1", "jpg_0"

        sample["jpg_0"], sample["jpg_1"] = sample[chosen], sample[rejected]

        if random.random() > 0.5:
            left, right = "jpg_0", "jpg_1"
        else:
            left, right = "jpg_1", "jpg_0"

        sample["left"] = left
        sample["right"] = right

        prompt = sample["caption"]

        if data_name == "pickscore" or data_name == "pickscore_normal" or data_name =="HPD_v2":
            sample[left] = process_to_IMG(sample[left])
            sample[right] = process_to_IMG(sample[right])
        
        RECORD = (_ %5==0)
        # RECORD = True
        ANS_DICT = reward_fn(sample[left], sample[right], prompt, Error=error, ANALYSIS=cache, RECORD=RECORD, cklth=cklstpmt, toolbox=toolbox)

        reward = ANS_DICT["model chose"]

        fmt = ANS_DICT["fomat correctness"]

        lth = ANS_DICT["lth"]

        chosen = sample["left"]

        rejected = sample["right"]

        images = [sample[chosen], sample[rejected]]
        
        _ = sample["_"]
        
        if reward is None:
            flag = None
        else:
            flag = reward > 0 if chosen == "jpg_0" else reward < 0

        with tempfile.TemporaryDirectory() as tmpdir:
            
            for i, pil in enumerate(images):
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))

            wandb.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.100} | {((-1)**i)*reward:.2f}",
                        )
                        for i in range(2)  # 仅记录两个图像
                    ],
                },
                commit=False,
            )
        
        wandb.log(
            {
                "correct": int(flag) + (reward == 0) / 2,
                "Total inference Lenth": lth,
            },
            commit=False,
        )
        

        if fmt == 0:
            return -0.5, ANS_DICT["logp"], fmt
        # -1.5 格式错+判断错 -1是格式对判断错了 -0.5 格式错 判断一样 0格式对判断一样 0.5是格式错答案对 1是全对
        return -0.5+ 0.5 * fmt + 2*flag + (reward == 0) -1, ANS_DICT["logp"],fmt

    def process(tup):
        _,now = tup
        #try:
        subprocess(tup)
        return True
        # except Exception:
        #     error.write(f"pic {now} is not run successfully")
        #     return False

    # ppo_config = PPOConfig(
    #     batch_size=4,
    #     learning_rate=1e-5,
    #     gradient_accumulation_steps=8,
    # )

    # ppo_trainer = PPOTrainer(
    #     model=model,
    #     args=ppo_config,
    #     train_dataset=dataset,
    #     processing_class=processor,
    #     ref_model= None,
    #     reward_model=rwd_calc,
    # )
    
    # 训练循环
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)

    num_epochs = 3  # 训练轮数

    #dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for epoch in range(num_epochs):

        model.train()
        epoch_loss = 0
        cnt = 0
        cum_rwd = 0
        for sample in dataset:

            rwd,logp,fmt = rwd_calc((sample, 0))
            loss = -rwd* logp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cum_rwd += rwd
            cnt +=1

            # breakpoint()
            wandb.log(
            {
                "avg_loss": epoch_loss/cnt,
                "reward": rwd,
                "fmt":int(fmt),
                "avg_rwd":cum_rwd/cnt,
            },
        )
    

    return 0


if __name__ == "__main__":
    app.run(main)
