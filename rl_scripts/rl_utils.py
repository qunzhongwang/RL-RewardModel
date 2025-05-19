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

# 自定义模块导入
from ddpo_pytorch.vlm_as_rm.rewards_Qwen import FullPmt
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

def _global_grad_norm(parameters, norm_type=2):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = 0.0
    for p in parameters:
        param_norm = safe_get_full_grad(p).norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


# 定义保存 checkpoint 的函数
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