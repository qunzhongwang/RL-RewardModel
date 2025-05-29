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
from accelerate.state import AcceleratorState

from accelerate.utils import set_seed, ProjectConfiguration, FP8RecipeKwargs
from accelerate.logging import get_logger
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

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
flags.DEFINE_bool("entropy_loss", False, "override entropy loss switch")
flags.DEFINE_bool("reward_cot", False, "override reward cot")

FLAGS = flags.FLAGS
FLAGS(sys.argv)

logger = get_logger(__name__)

def main(_):
    config = FLAGS.config
    if FLAGS.Project_name is not None:
        config.Project_name = FLAGS.Project_name
    if FLAGS.debug_ver is not None:
        config.debug_ver = FLAGS.debug_ver
    if FLAGS.entropy_loss is not None:
        config.rl_conf.entropy_loss = FLAGS.entropy_loss
    if FLAGS.reward_cot is not None:
        config.reward.reward_long_cot = FLAGS.reward_cot
    
    if config.debug_ver:
        config.Project_name = "DEBUG_NULL"

    
    # basic Accelerate and logging setup
    # set_seed(config.seed)
    # 获取当前 GPU 的进程 ID
    
    if config.deepspeed_stage in [1,2]:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=config.deepspeed_stage,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    elif config.deepspeed_stage == 3:
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=config.deepspeed_stage,
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
        AcceleratorState().deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1
    else:
        accelerator = Accelerator( 
            log_with="wandb",
            mixed_precision=config.mixed_precision,
            project_config=accelerator_config,
            gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        )
    
    rank = accelerator.process_index 

    seed = 42 + rank 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Rank {rank}: Random seed set to {seed}")

    if config.deepspeed_stage != 3:
        if config.pretrained.model == "Qwen/Qwen2-VL-7B-Instruct":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct", 
                torch_dtype=torch.bfloat16, 
                device_map= accelerator.device, 
                attn_implementation=config.pretrained.attn_implementation,
                cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct", 
                    torch_dtype=torch.bfloat16, 
                    device_map= accelerator.device, 
                    attn_implementation=config.pretrained.attn_implementation,
                    cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen2.5-VL-7B-Chat",
                )
    else:
        if config.pretrained.model == "Qwen/Qwen2-VL-7B-Instruct":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.pretrained.model, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=False,
                attn_implementation=config.pretrained.attn_implementation,
                cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
            )
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.pretrained.model, 
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=False,
                attn_implementation=config.pretrained.attn_implementation,
                cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen2.5-VL-7B-Chat",
            )
    if config.pretrained.model == "Qwen/Qwen2-VL-7B-Instruct":
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    else:
        min_pixels = 16*14*14
        max_pixels = 120*14*14
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    
    rank = accelerator.local_process_index

    # if rank == 0:

    #     print("Main process before breakpoint")
    #     import pdb
    #     pdb.set_trace() 
    #     print("Main process after breakpoint")
    # else:
    #     # 子进程逻辑
    #     print(f"Sub process {rank} before wait")
    

    accelerator.wait_for_everyone()

    processor.tokenizer.padding_side = "left"


    for name, module in model.named_modules():
        if "attention" in name.lower():
            logger.info(f"{name}: {type(module)}")

    dataset_name = config.data_conf.dataset_url
    data_name = get_dataname(dataset_name) 

    if data_name == "pickscore_normal" or data_name == "HPD_v2":
        #config.data_conf.chunk_size = 10
        train_dataset,val_dataset = get_streamed_dataset(dataset_name, config.data_conf.chunk_size, config.data_conf.verify_chunk_size)
    elif data_name == "human_video":
        dataset = load_dataset(
        "csv", 
        data_files="/m2v_intern/wangqunzhong/research/kwai_data/dataset/data.csv"
        )["train"]
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]
        train_dataset = train_dataset.select_columns(["chosen_video_path", "rejected_video_path", "caption"])
        val_dataset = val_dataset.select_columns(["chosen_video_path", "rejected_video_path", "caption"])
    else:
        dataset = load_dataset(dataset_name, split="validation", num_proc=64)
    
    my_counter_closure = [0]
    loader = DataLoader(
        train_dataset, 
        batch_size=config.train.batch_size, 
        shuffle=False, 
        collate_fn=make_collate_fn(
            processor, 
            data_name, 
            accelerator=accelerator, 
            parser_type=config.data_type, 
            video_fps=config.video_fps, 
            config=config,
            counter_closure=my_counter_closure
        ))
    
    validation_loader =  DataLoader(val_dataset, batch_size=config.val.val_batch_size, shuffle=True, collate_fn=make_collate_fn(processor, data_name))
    unique_id,ckptuid = get_uid()

    config.run_name += "_" + data_name + unique_id
    outputfile_name = "outfile_" + config.run_name

    
    cumulative_sums = {}; cumulative_counts = {}

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
    

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})

    if config.inference:
        model, loader, validation_loader = accelerator.prepare(model, loader, validation_loader)
    else:
        model, optimizer = accelerator.prepare(model, optimizer)
        # loader = accelerator.prepare_data_loader(loader, prepare_data_loader=False)
        # validation_loader = accelerator.prepare_data_loader(validation_loader, prepare_data_loader=False)
    
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
            init_kwargs={"wandb": {
                "name": config.run_name,
                "entity":"KwaiAiTraining",
            }},
        )
    logger.info(f"\n{config}")


    reward_fn = getattr(ddpo_pytorch.vlm_as_rm.rewards_Qwen, config.reward.reward_pw_fn)(select=config.select)
    
    def reward_func(batch,accelerator):
        loss,retInfo = reward_fn(batch, toolbox= (model, processor,logger), accelerator=accelerator,config=config)
        return loss, retInfo

    correctness_queue = deque(maxlen=5)
    reward_queue = deque(maxlen=5)
    std_queue = deque(maxlen=5)

    cur_table = wandb.Table(columns=["step", "Video Left", "Video Right","correct or not", "reasoning"],data=[])

    for epoch in tqdm(range(init_epoch, config.num_epochs)):
        #logger.info("next epoch")
        model.train()
        tmp_grad = []
        iter_lth = len(loader)
        for idx, batch in tqdm(enumerate(loader)):
            try:
                if batch[0] is None:
                    doc = ""
                    chz = 0.5
                    continue
                #print(batch[1][0]['caption'])
                loss,retInfo = reward_func(batch[0],accelerator)
                doc = retInfo.pop("doc to record", None)
                chz = retInfo.pop("chz to record", None)
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
            except Exception as exp:
                print(f"{exp}")
                pass
            my_counter_closure = (iter_lth * epoch + idx) + 1
            if accelerator.is_main_process and idx % config.log_freq == 0:
                cumulative_avg_info = {
                    key: cumulative_sums[key] / cumulative_counts[key] 
                    for key in cumulative_sums
                    if cumulative_counts[key] != 0
                }
                
                cur_table = wandb.Table(cur_table.columns, cur_table.data.append([
                    (iter_lth * epoch + idx),
                    wandb.Video(batch[1][-1]["chosen_video_path"]),
                    wandb.Video(batch[1][-1]["rejected_video_path"]),
                    doc,
                    chz,
                ]))
                accelerator.log({
                    "Case_Study_Table":cur_table,
                    "loss": loss,
                    "grad norm":sum(tmp_grad) / len(tmp_grad) if tmp_grad else 0,
                    "recent reward": sum(reward_queue) / len(reward_queue) if reward_queue else 0,
                    "recent correctness": sum(correctness_queue) / len(correctness_queue) if correctness_queue else 0,
                    "recent std": (sum(std_queue) / len(std_queue))**(1/2) if std_queue else 0,
                    **cumulative_avg_info,

                },
                step= (iter_lth * epoch + idx)
                )
                tmp_grad.clear()

            #增加 save ckpoint的历程
            if accelerator.is_main_process and not config.inference and (iter_lth * epoch + idx) % config.ckpt_freq :
                os.makedirs(config.logdir, exist_ok=True) 
                checkpoint_path = os.path.join(config.logdir, f"checkpoint_{accelerator.num_processes}gpus_z{config.deepspeed_stage}_{ckptuid}_{(iter_lth * epoch + idx)}.pt")
                accelerator.save({
                    "epoch": epoch + 1,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cumulative_sums": cumulative_sums,
                    "cumulative_counts": cumulative_counts,
                }, checkpoint_path)
                lora_pth = os.path.join(config.loradir,f"lora_{accelerator.num_processes}gpus_z{config.deepspeed_stage}_{ckptuid}_{epoch}")
                model.save_pretrained(lora_pth)
                logger.info(f"Checkpoint saved to {checkpoint_path}\nlora saved to {lora_pth}")
        
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
                },
                step= (iter_lth * epoch + idx),
                )
            config.inference = False
        
    return 0


if __name__ == "__main__":
    app.run(main)
