# 标准库导入
import re
import os
import random
import base64
import io
from io import BytesIO
import datetime

# 第三方库导入
import torch
import numpy as np
from PIL import Image
from datasets import Dataset, load_dataset, load_from_disk
from qwen_vl_utils import process_vision_info
import wandb

# 自定义模块导入
from ddpo_pytorch.vlm_as_rm.rewards_Qwen import my_prompt
from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state

def get_uid():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y.%m.%d_%H.%M.%S")
    formatted_date = now.strftime("%m%d%H")
    
    return formatted_time,formatted_date

def get_dataname(dataset_name=None):    
    if dataset_name == "/m2v_intern/liujie/research/huggingface/dataset/yuvalkirstain/pickapic_v1_unique":
        data_name = "pickscore"
    elif dataset_name == "/m2v_intern/liujie/research/huggingface/dataset/imagereward/fidelity_rating_dataset":
        data_name = "imagereward"
    elif dataset_name == "/m2v_intern/liujie/research/huggingface/dataset/yuvalkirstain/pickapic_v1":
        data_name = "pickscore_normal"
    elif dataset_name == "/m2v_intern/wangqunzhong/research/kwai_data/dataset/data":
        data_name = "human_video"
    else:
        data_name = "HPD_v2"
    return data_name

def _get_streamed_dataset(dataset_url,chunk_size=100,verify_chunk_size=100):
    dataset = load_dataset(dataset_url, split="train", streaming = True)
    start_idx = random.randint(200, 2000)
    chunk_size = chunk_size  
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
    return dataset

def get_streamed_dataset(dataset_url, chunk_size=100, val_chunk_size=20, seed=42):
    if seed is not None:
        random.seed(seed)
    dataset = load_dataset(dataset_url, split="train", streaming=True)
    buffer = []
    skip_counter = 0
    for idx, example in enumerate(dataset):
        if skip_counter < 2:
            skip_counter += 1
            continue
        buffer.append(example)
        skip_counter = 0
        if len(buffer) >= chunk_size + val_chunk_size:
            break
    train_buffer = buffer[:chunk_size]
    val_buffer = buffer[chunk_size:chunk_size + val_chunk_size]
    train_dataset = Dataset.from_list(train_buffer)
    val_dataset = Dataset.from_list(val_buffer)

    return train_dataset, val_dataset

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
        img = img.resize((new_width, new_height))
        images.append(img)

        total_width += new_width
        max_height = max(max_height, new_height)

    # Create a blank combined image
    combined_image = Image.new('RGB', (total_width + 5 * (len(imagelist) - 1), max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + 5

    if as_base64:
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)  # Move to the beginning of the BytesIO stream
        img_base64 = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        return img_base64
    return combined_image

def make_collate_fn(processor, data_name,parser_type="image",accelerator=None,video_fps=8., config=None,counter_closure=None):
    
    cache_mapping = {
        "human_video" : "video",

    }
    if data_name in cache_mapping:
        parser_type = cache_mapping[data_name]

    def collate(batch):  
        
        #胜负随机inverse
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
            imgs.append([sample["lpic"], sample["rpic"]])
            # 同一段文字 prompt
            msg = [{
                "role":"user",
                "content":[
                    {"type":"image"},
                    {"type":"image"},
                    {"type":"text","text":my_prompt.format(locPrompt=sample["caption"])}
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
        # 如果传入了 accelerator，将数据移动到其 device
        if accelerator is not None:
            for key, value in model_inputs.items():
                if isinstance(value, torch.Tensor):
                    model_inputs[key] = value.to(accelerator.device)
            # 如果需要，处理 batch 中的其他数据
            for sample in batch:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.to(accelerator.device)

        return model_inputs, batch          # 附带原始样本用于 reward
    
    def video_collate(batch):
        my_prompt = """\
            Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each video by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: 'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...\n2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...\n3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...\n[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...\nTotal score:\nVideo 1: 9+8+7+6=30\nVideo 2: 7+6+5+8=26\n</think>\n<answer>Video 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]\
        """
        fps=video_fps
        use_frames = False
        videos   = []
        prompts= []
        msgs = []
        invs = []
        for sample in batch:
            if random.random() > 0.5:
                lvd, rvd, inv = "chosen_video_path", "rejected_video_path", 0
            else:
                lvd, rvd, inv = "rejected_video_path", "chosen_video_path", 1

            invs.append(inv)
            if use_frames:
                pass
            else:

                msg = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"{sample[lvd]}",
                                "max_pixels": config.input_conf.input_pixel_conf,  # 14 * 14 *80,
                                "total_pixels": config.input_conf.total_pixel,  # 1024 * 28 * 28,
                                "fps": fps,
                            },
                            {
                                "type": "video",
                                "video": f"{sample[rvd]}",
                                "max_pixels": config.input_conf.input_pixel_conf,  # 14 * 14 * 80,
                                "total_pixels": config.input_conf.total_pixel,  # 1024 * 28 * 28,
                                "fps":fps,
                            },
                            {"type": "text", "text": my_prompt.format(prompt=sample["caption"])},
                        ],
                    }
                ]

            prompts.append(
                processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
                )
            )
            msgs.append(msg)

        # tokenizer 会把 <image> placeholder 插进去
        # breakpoint()
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(msgs, return_video_kwargs=True)
            model_inputs = processor(text=prompts,images=image_inputs,videos=video_inputs, padding=True,return_tensors="pt",**video_kwargs)
            breakpoint()
            accelerator.print(model_inputs["pixel_values_videos"].shape, model_inputs["image_thw"])
            rank = accelerator.local_process_index
        
            # if rank == 0:
            #     print("Main process before breakpoint")
            #     import pdb
            #     pdb.set_trace() 
            #     print("Main process after breakpoint")
            # else:
            #     # 子进程逻辑
            #     print(f"Sub process {rank} before wait")
    
            model_inputs["invs"] = invs
            if counter_closure and accelerator.is_main_process and counter_closure[0] % config.log_freq == 0 and False:
                
                wandb.log(
                    {
                        "video_case":[
                            wandb.Video(sample[lvd],caption=f"ground_truth:{int(invs[-1] == 0)}"),
                            wandb.Video(sample[rvd],caption=f"ground_truth:{int(invs[-1] == 0)}")
                        ],
                    },
                    commit=False,
                )

            return model_inputs, batch
        except Exception as exp:
            print(f"{exp}")
            return None, None
    
    collate_dict = {
        "image": collate,
        "video": video_collate
    }

    return collate_dict[parser_type]

def _global_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = 0.0
    for p in parameters:
        param_norm = safe_get_full_grad(p).norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
def save_checkpoint(epoch, model, optimizer, cumulative_sums, cumulative_counts, checkpoint_dir="ckpt_log", uid="0515"):
    os.makedirs(checkpoint_dir, exist_ok=True) 
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{uid}_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cumulative_sums": cumulative_sums,
        "cumulative_counts": cumulative_counts,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def find_ckpt(logdir, epoch, mode="ckpt",gpus=1,zeRO=0):
    if mode == "ckpt":
        pattern = re.compile(rf"checkpoint_{gpus}gpus_z{zeRO}_"r"(\d{6})"rf"_{epoch}")
    elif mode == "lora":
        pattern = re.compile(rf"lora_{gpus}gpus_z{zeRO}_"r"(\d{6})"rf"_{epoch}")

    latest_uid = None
    latest_ckpt_path = None
    for filename in os.listdir(logdir):
        
        match = pattern.match(filename)
        if match:
            uid = match.group(1)
            ckpt_path = os.path.join(logdir, filename)
            if latest_uid is None or uid > latest_uid:
                latest_uid = uid
                latest_ckpt_path = ckpt_path
    return latest_ckpt_path