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
from rl_utils import make_collate_fn, _global_grad_norm, encodeAsPIL, process_to_IMG, get_uid, get_dataname, get_streamed_dataset,find_ckpt

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

config_flags.DEFINE_config_file("config", "config/base_rl.py", "Training configuration.")
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
    
    # basic Accelerate and logging setup
    set_seed(config.seed)
    if config.deepspeed_stage in [1,2]:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=config.deepspeed_stage,  # ZeRO Stage 选择
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    elif config.deepspeed_stage == 3:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=config.deepspeed_stage,  # 可以拓展额外配置
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        )
    
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    if config.deepspeed_stage in [1,2,3]:
        accelerator = Accelerator(
            log_with="wandb",
            mixed_precision=config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
            deepspeed_plugin=deepspeed_plugin,
        )
    else:
        accelerator = Accelerator( 
            log_with="wandb",
            mixed_precision=config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        )
    
    if config.deepspeed_stage != 3:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map= accelerator.device, 
            attn_implementation=config.pretrained.attn_implementation,
            cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.pretrained.model, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=False,
            attn_implementation=config.pretrained.attn_implementation,
            cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
        )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    processor.tokenizer.padding_side = "left"


    for name, module in model.named_modules():
        if "attention" in name.lower():
            logger.info(f"{name}: {type(module)}")

    dataset_name = config.data_conf.dataset_url
    data_name = get_dataname(dataset_name) 

    if data_name == "pickscore_normal" or data_name == "HPD_v2":
        #config.data_conf.chunk_size = 10
        train_dataset,val_dataset = get_streamed_dataset(dataset_name, config.data_conf.chunk_size, config.data_conf.verify_chunk_size)
    else:
        dataset = load_dataset(dataset_name, split="validation", num_proc=64)
    
    loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=make_collate_fn(processor, data_name))
    validation_loader =  DataLoader(val_dataset, batch_size=config.val.val_batch_size, shuffle=True, collate_fn=make_collate_fn(processor, data_name))
    unique_id,ckptuid = get_uid()

    config.run_name += "_" + data_name + unique_id
    outputfile_name = "outfile_" + config.run_name


    # if config.resume_from:
    #     config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
    #     if "checkpoint_" not in os.path.basename(config.resume_from):
    #         checkpoints = list(
    #             filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
    #         )
    #         if len(checkpoints) == 0:
    #             raise ValueError(f"No checkpoints found in {config.resume_from}")
    #         config.resume_from = os.path.join(
    #             config.resume_from,
    #             sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
    #         )
    
    cumulative_sums = {
    #     "fomat correctness": 0.,
    #     "choose correctness": 0.,
    #     "avg lth": 0.,
    #     "avg reward": 0.,
    #     "avg reasoning": 0.,
    #     "global avg correctness":0.,
    #     "global avg reward":0.,
    #     "global reward std":0.,
    #     "pure loss":0.,
    #     "kl loss":0.,
    # #"avg check point count": 0,
    }

    cumulative_counts = {
    #     "fomat correctness": 0.,
    #     "choose correctness": 0.,
    #     "avg lth": 0.,
    #     "avg reward": 0.,
    #     "avg reasoning": 0.,
    #     "global avg correctness":0.,
    #     "global avg reward":0.,
    #     "global reward std":0.,
    #     "pure loss":0.,
    #     "kl loss":0.,
    # #"avg check point count": 0,
        }

    # 恢复 checkpoint
    if config.resume_ckpt:
        lora_path = find_ckpt(config.loradir, config.resume_ckpt,mode="lora",gpus=accelerator.num_processes,zeRO=config.deepspeed_stage)
        if lora_path and os.path.exists(lora_path) and not config.inference:
            model = PeftModel.from_pretrained(
                model, 
                lora_path,
                is_trainable=True
                )
        elif lora_path and os.path.exists(lora_path) and config.inference:
            model = PeftModel.from_pretrained(
                model, 
                lora_path,
                is_trainable=False
                )
        else:
            logger.info("No lora found.")
            lora_config = LoraConfig(
                r=config.train.lora_r,              
                lora_alpha=config.train.lora_alpha,    
                target_modules=config.train.lora_target_modules,
                lora_dropout=config.train.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
        checkpoint_path = find_ckpt(config.logdir, config.resume_ckpt, mode="ckpt",gpus=accelerator.num_processes,zeRO=config.deepspeed_stage)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if config.train.use_8bit_adam and not config.inference:
                optimizer = AdamW8bit(model.parameters(), lr=config.train.learning_rate)
            elif not config.inference:
                optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
            if not config.inference:
                optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
            init_epoch = checkpoint.get("epoch", 0)
            cumulative_sums = checkpoint.get("cumulative_sums", {})
            cumulative_counts = checkpoint.get("cumulative_counts", {})
            logger.info(f"Resumed training from epoch {init_epoch} using checkpoint: {checkpoint_path}")
        else:
            logger.info("No checkpoint found.")
            init_epoch = 0
            if config.train.use_8bit_adam and not config.inference:
                optimizer = AdamW8bit(model.parameters(), lr=config.train.learning_rate)
            elif not config.inference:
                optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
    else:
        lora_config = LoraConfig(
            r=config.train.lora_r,              
            lora_alpha=config.train.lora_alpha,    
            target_modules=config.train.lora_target_modules,
            lora_dropout=config.train.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        logger.info("Not going to use checkpoint, starting from scratch.")
        init_epoch = 0
        if config.train.use_8bit_adam and not config.inference:
            optimizer = AdamW8bit(model.parameters(), lr=config.train.learning_rate)
        elif not config.inference:
            optimizer = AdamW(model.parameters(), lr=config.train.learning_rate)
    

    
    #model.gradient_checkpointing_enable()
    #,cumulative_sums, cumulative_counts
    if config.inference:
        model, loader, validation_loader = accelerator.prepare(model, loader, validation_loader)
    else:
        model, optimizer, loader, validation_loader = accelerator.prepare(model, optimizer, loader, validation_loader)
    
    for key1,key2 in zip(cumulative_sums, cumulative_counts):
        if isinstance(cumulative_sums[key1], torch.Tensor):
            cumulative_sums[key1] = cumulative_sums[key1].item()
        if cumulative_sums[key1] is None:
            cumulative_sums[key1] = 0.
        if isinstance(cumulative_counts[key2], torch.Tensor):
            cumulative_counts[key2] = cumulative_counts[key2].item()
        if cumulative_counts[key2] is None:
            cumulative_counts[key2] = 0.
    
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.Project_name,
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")


    reward_fn = getattr(ddpo_pytorch.vlm_as_rm.rewards_Qwen, config.reward.reward_pw_fn)()
    
    def reward_func(batch,accelerator):
        loss,retInfo = reward_fn(batch, toolbox= (model, processor,logger), accelerator=accelerator,config=config)
        return loss, retInfo

    correctness_queue = deque(maxlen=5)
    reward_queue = deque(maxlen=5)
    std_queue = deque(maxlen=5)
    #global_step = 0
    for epoch in tqdm(range(init_epoch, config.num_epochs)):
        #logger.info("next epoch")
        model.train()
        tmp_grad = []
        iter_lth = len(loader)
        for idx, batch in tqdm(enumerate(loader)):
            loss,retInfo = reward_func(batch[0],accelerator)
            for key in retInfo:
                if isinstance(retInfo[key], torch.Tensor):
                    retInfo[key] = retInfo[key].item()
                if key not in cumulative_sums:
                    cumulative_sums[key] = 0.0
                    cumulative_counts[key] = 0
                cumulative_sums[key] += retInfo[key]
                cumulative_counts[key] += 1
            correctness_queue.append(retInfo["global avg correctness"])
            reward_queue.append(retInfo["global avg reward"])
            std_queue.append(retInfo["global reward std"])

            if not config.inference:
                optimizer.zero_grad()
                #print(torch.cuda.memory_summary())
                accelerator.backward(loss)
            
                if accelerator.sync_gradients:
                    params_to_clip = (p for p in model.parameters() if p.requires_grad)
                    tmp_grad.append(_global_grad_norm(model.parameters()))
                    accelerator.print(_global_grad_norm(model.parameters()))
                    accelerator.print(tmp_grad)
                    accelerator.clip_grad_norm_(params_to_clip, config.train.max_grad_norm)
                optimizer.step()
                loss = loss.item() if loss else 0.
            else:
                loss = 0.

            if idx % config.log_freq!= 0:
                accelerator.log(
                    {"loss": loss,},
                    step=(iter_lth * epoch + idx) #accelerator.num_processes,
                )
            
            # 每隔 10 个 batch 记录一次日志（仅主进程）
            if accelerator.is_main_process and idx % config.log_freq == 0:
                cumulative_avg_info = {
                    key: cumulative_sums[key] / cumulative_counts[key] 
                    for key in cumulative_sums
                    if cumulative_counts[key] != 0
                }

                accelerator.log({
                    "loss": loss,  # 记录当前 loss
                    "grad norm":sum(tmp_grad) / len(tmp_grad) if tmp_grad else 0,
                    "recent reward": sum(reward_queue) / len(reward_queue) if reward_queue else 0,
                    "recent correctness": sum(correctness_queue) / len(correctness_queue) if correctness_queue else 0,
                    "recent std": (sum(std_queue) / len(std_queue))**(1/2) if std_queue else 0,
                    **cumulative_avg_info,
                    #"step": idx           # 当前 step
                },
                step= (iter_lth * epoch + idx)#accelerator.num_processes,#global_step,
                )
                tmp_grad.clear()
                #增加 save ckpoint的历程
        if not config.inference:
            model.eval()
            config.inference = True
            validation_sums = {}
            validation_counts = {}

            with torch.no_grad():
                for val_idx, val_batch in tqdm(enumerate(validation_loader), desc="Validation"):
                    val_loss, val_retInfo = reward_func(val_batch[0], accelerator)
                    for key in val_retInfo:
                        if isinstance(val_retInfo[key], torch.Tensor):
                            val_retInfo[key] = val_retInfo[key].item()
                        if key not in validation_sums:
                            validation_sums[key] = 0.0
                            validation_counts[key] = 0
                        validation_sums[key] += val_retInfo[key]
                        validation_counts[key] += 1

            validation_avg_info = {
                "validation/"+key: validation_sums[key] / validation_counts[key] for key in validation_sums
            }
            validation_avg_info["validation/global reward std"] = validation_avg_info["validation/global reward std"]**(1/2)

            accelerator.log({
                    **validation_avg_info,
                    #"step": idx           # 当前 step
                },
                step= (iter_lth * epoch + idx),#accelerator.num_processes,#global_step,
                )
            config.inference = False
        #增加 save ckpoint的历程
        if accelerator.is_main_process and not config.inference:  # 仅主进程保存
            os.makedirs(config.logdir, exist_ok=True) 
            checkpoint_path = os.path.join(config.logdir, f"checkpoint_{accelerator.num_processes}gpus_z{config.deepspeed_stage}_{ckptuid}_{epoch}.pt")
            accelerator.save({
                "epoch": epoch + 1,
                "optimizer_state_dict": optimizer.state_dict(),
                "cumulative_sums": cumulative_sums,
                "cumulative_counts": cumulative_counts,
            }, checkpoint_path)
            lora_pth = os.path.join(config.loradir,f"lora_{accelerator.num_processes}gpus_z{config.deepspeed_stage}_{ckptuid}_{epoch}")
            model.save_pretrained(lora_pth)
            logger.info(f"Checkpoint saved to {checkpoint_path}\nlora saved to {lora_pth}")

    return 0


if __name__ == "__main__":
    app.run(main)
