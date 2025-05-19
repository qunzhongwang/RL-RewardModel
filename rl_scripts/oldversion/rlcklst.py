from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import io
import base64
from PIL import Image

def encodeAsBase64(imagelist, target_size=512,  as_base64=False):
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

# 加载两张待比较的图片
image1 = Image.open("tmp.png")
image2 = Image.open("tmp1.png")

# 合并图片并生成base64
image = encodeAsBase64([image1, image2])

# 初始化模型和处理器
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    torch_dtype="auto", 
    device_map="auto", 
    cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen-VL-7B-Chat",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": "What are the differences between the two images?"},
        ],
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")



# 生成比较结果
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("Image Comparison Result:", output_text)