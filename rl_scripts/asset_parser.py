import os
import gdown
import zipfile
# from IPython.display import HTML
from base64 import b64encode
import torchvision
import transformers
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

# def download_file_from_drive(file_id, output_path="./video_asset"):
#     # Create the output directory if it doesn't exist
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     # Download the file from Google Drive using the file ID
#     download_path = os.path.join(output_path, "working_video.mp4")
#     gdown.download(f"https://drive.google.com/uc?id={file_id}", download_path, quiet=False)

#     print(f"Files extracted to {output_path}")
# os.makedirs("./video_asset",exist_ok=True)
# download_file_from_drive("1GB8FwkwRtMfeFMwfvTregFd-pWD6IXlv", output_path="./video_asset/working_video.mp4")
# # Path to your video file
# video_path = "./video_asset/working_video.mp4"

# # Read the video file and encode it in base64



# # Display the video
# # display(HTML(video_tag))

# model = Qwen2VLForConditionalGeneration.from_pretrained(
#             "Qwen/Qwen2-VL-7B-Instruct", 
#             torch_dtype=torch.bfloat16, 
#             device_map= "auto", 
#             cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
#         )
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# def query_video(prompt, use_frames=True, frames_path="/home/qwen2_vl/content/frames", video_path=None):
#     if use_frames:
#         # Get the frames
#         selected_frames = get_frame_list(output_path)

#         # Create messages structure for frames
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": selected_frames,
#                         "fps": 1.0,
#                     },
#                     {"type": "text", "text": prompt},
#                 ],
#             }]
#     else:
#         # Create messages structure for the entire video
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": f"file://{video_path}",
#                         "max_pixels": 360 * 420,
#                         "fps": 1.0,
#                     },
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]

#     print(f"Using {'frames' if use_frames else 'entire video'} for inference.")

#     # Preparation for inference
#     text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
#     breakpoint()
#     image_inputs, video_inputs = process_vision_info([messages, messages])
#     breakpoint()
#     inputs = processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
#     #inputs = processor(text=[text,text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt",)
#     inputs = inputs.to("cuda")
#     breakpoint()

#     # Inference
#     with torch.no_grad():  # Use no_grad to save memory during inference
#         generated_ids = model.generate(**inputs, max_new_tokens=128)

#     # Trim the generated output to remove the input prompt
#     generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

#     # Decode the generated text
#     output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     breakpoint()

#     print(output_text)
#     torch.cuda.empty_cache()
# video_path = "./video_asset/working_video.mp4/working_video.mp4"
# absolute_path = os.path.abspath(video_path)
# query_video("What is the specific name of the sports discipline in this video?",
#             use_frames=False, video_path=absolute_path)

class VideoDataset(Dataset):
    default_csv = "/m2v_intern/shuangji/data"
    def __init__(self, csv_file=VideoDataset.default_csv, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data.iloc[idx]
        video_path = row["video_path"]
        video_length = row["video_length"]

        vae_512_path = row["vae_512_path"]
        vae_256_path = row["vae_256_path"]
        kps_style2_path = row["kps_style2_path"]
        kps_style2_length = row["kps_style2_length"]

        caption = row["caption"]
        have_text = row["have_text"]
        ocr_bbox = eval(row["ocr_bbox"])  # 转换为列表
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(vae_512_path):
            raise FileNotFoundError(f"VAE 512 file not found: {vae_512_path}")
        if not os.path.exists(vae_256_path):
            raise FileNotFoundError(f"VAE 256 file not found: {vae_256_path}")
        sample = {
            "video_path": video_path,
            "video_length": video_length,
            "caption": caption,
            "vae_512_path": vae_512_path,
            "vae_256_path": vae_256_path,
            "kps_style2_path": kps_style2_path,
            "kps_style2_length": kps_style2_length,
            "have_text": have_text,
            "ocr_bbox": ocr_bbox,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# 测试 Dataset 类
if __name__ == "__main__":
    # CSV 文件路径
    csv_file = "data.csv"  # 替换为你的 CSV 文件路径

    # 创建 Dataset
    dataset = VideoDataset(csv_file)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # 测试读取数据
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}:")
        for key, value in batch.items():
            print(f"  {key}: {value}")
        break