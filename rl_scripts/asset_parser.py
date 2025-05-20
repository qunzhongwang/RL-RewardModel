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

def download_file_from_drive(file_id, output_path="./video_asset"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download the file from Google Drive using the file ID
    download_path = os.path.join(output_path, "working_video.mp4")
    gdown.download(f"https://drive.google.com/uc?id={file_id}", download_path, quiet=False)

    print(f"Files extracted to {output_path}")
os.makedirs("./video_asset",exist_ok=True)
download_file_from_drive("1GB8FwkwRtMfeFMwfvTregFd-pWD6IXlv", output_path="./video_asset/working_video.mp4")
# Path to your video file
video_path = "./video_asset/working_video.mp4"

# Read the video file and encode it in base64



# Display the video
# display(HTML(video_tag))

model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16, 
            device_map= "auto", 
            cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
        )
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def query_video(prompt, use_frames=True, frames_path="/home/qwen2_vl/content/frames", video_path=None):
    if use_frames:
        # Get the frames
        selected_frames = get_frame_list(output_path)

        # Create messages structure for frames
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": selected_frames,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        # Create messages structure for the entire video
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    breakpoint()
    print(f"Using {'frames' if use_frames else 'entire video'} for inference.")

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    breakpoint()

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    breakpoint()

    # Inference
    with torch.no_grad():  # Use no_grad to save memory during inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim the generated output to remove the input prompt
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    breakpoint()

    print(output_text)
    torch.cuda.empty_cache()

query_video("What is the specific name of the sports discipline in this video?",
            use_frames=False, video_path="./video_asset/working_video.mp4")