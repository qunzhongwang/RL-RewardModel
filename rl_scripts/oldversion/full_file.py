# train_qwen2_vl_ppo_full.py
# ======================================================================
#  1. 依赖
# ======================================================================
from __future__ import annotations
import os, io, random, datetime, tempfile, sys, json, re, base64
from dataclasses import dataclass
from typing import Dict, List, Any

from datasets import load_from_disk, load_dataset, Dataset

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer
from accelerate import Accelerator

import yaml
import wandb

# -----------------------------------------------------------------------------
#  2. 配置区 – 路径 & 超参 (改成你自己的)
# -----------------------------------------------------------------------------
DATASET_PATH = "/m2v_intern/wangqunzhong/research/huggingface/dataset/ymhao/HPD_v2"
RUN_NAME     = "qwen2_vl_ppo_lora"
WANDB_PROJ   = "PPO_DEBUG"

MAX_NEW_TOK  = 256          # generate 时 token 上限
BATCH_SIZE   = 2            # rollout batch
PPO_EPOCHS   = 4            # 每个 batch 的 PPO epoch
LR           = 1e-5         # LoRA 学习率
LORA_RANK    = 8

SEED         = 42

# -----------------------------------------------------------------------------
#  3. 一些共用工具函数
# -----------------------------------------------------------------------------
def seed_everything(seed:int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_to_IMG(img_bytes: bytes) -> Image.Image:
    """
    数据集中存的是 jpeg bytes;
    转成 PIL, 做最小 resize, 同时保持三通道.
    """
    img = Image.open(io.BytesIO(img_bytes))#.convert("RGB")
    return img


def encodeAsPIL(imagelist, target_size=512, as_base64=False):
    """
    把两张图拼成一张左右拼接的大图 Qwen-VL 只吃一张 image token 
    """
    images, total_width, max_height = [], 0, 0
    for image_data in imagelist:
        img = image_data.convert("RGB")
        orig_w, orig_h = img.size
        new_h  = target_size
        new_w  = int((target_size / orig_h) * orig_w)
        img    = img.resize((new_w, new_h))
        images.append(img)
        total_width += new_w
        max_height   = max(max_height, new_h)

    canvas = Image.new("RGB", (total_width + 5*(len(images)-1), max_height))
    x_off = 0
    for img in images:
        canvas.paste(img, (x_off, 0))
        x_off += img.width + 5

    if as_base64:
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode()
    return canvas

# -----------------------------------------------------------------------------
#  4. prompt、解析、校验函数 (摘自 evaluate_QwenVL2_7B )
# -----------------------------------------------------------------------------
with open("ddpo_pytorch/prompt_relative.yaml") as f:
    PROMPT_YAML = yaml.safe_load(f)

CKLST_PROMPT = PROMPT_YAML["cklst-prompt"]
EVAL_PROMPT  = PROMPT_YAML["eval-prompt"]

FullPmt = FullPmt = """
        Conduct image comparison which is generated by a prompt through these phases:

        1. **Checklist Generation**:
        - Identify critical comparison dimensions 
        - Assign weights (sum=100%) based on task importance

        2. **Structured Comparison**:
        - Tabular format: "Dimension | Image1 | Image2 | Preference | Weight"
        - Preference codes:  
            - 1 = Image1 superior
            - -1 = Image2 superior
            - 0 = Equivalent

        3. **Deep Analysis**:
        - Detailed reasoning for each dimension comparison
        - Explain judgment rationale and relative importance
        - Consider contextual factors and edge cases

        4. **Holistic Verdict**:
        - Synthesize analysis beyond numerical scores
        - Final declaration format: "Final Verdict: image1/tie/image2"

        Output MUST follow EXACTLY:
        ====BEGIN CHECKLIST====
        1. Dimension1: Weight%
        2. Dimension2: Weight%
        ...
        ====END CHECKLIST====

        ====BEGIN COMPARISON====
        Dimension | Image1 | Image2 | Preference | Weight
        Dimension1 | desc | desc | code | X%
        ...
        ====END COMPARISON====

        ====BEGIN ANALYSIS====
        [Free-form analytical reasoning demonstrating critical evaluation of 
        comparison results and their contextual implications]
        ====END ANALYSIS====

        ====BEGIN VERDICT====
        Final Verdict: Image1/Tie/Image2
        ====END VERDICT====
    Now, the prompt of generated images is {locPrompt}. The provided image is structured such that the **left part represents Image 1** and the **right part represents Image 2**. 
    """

def validate_format(text):
    required_blocks = {
        'CHECKLIST': r'(\d+\.\s.+?:\s\d+%)',
        'COMPARISON': r'([^|]+\|[^|]+\|[^|]+\|(-1|0|1)\|\s?\d+%)',
        'ANALYSIS': r'\S+',  # At least some non-whitespace content
        'VERDICT': r'Final Verdict:\s*(Image1|Tie|Image2)'
    }
    
    try:
        # Block existence check
        for block in required_blocks:
            if not re.search(f'====BEGIN {block}====.*====END {block}====', text, re.DOTALL):
                return False

        # Checklist validation
        checklist = re.findall(r'\d+\.\s(.+?):\s(\d+)%', 
                            re.search(r'====BEGIN CHECKLIST====(.*?)====END CHECKLIST====', text, re.DOTALL).group(1))
        if sum(int(w) for _,w in checklist) != 100 or len(checklist) < 2:
            return False

        # Comparison table validation
        comparison = re.search(r'====BEGIN COMPARISON====(.*?)====END COMPARISON====', text, re.DOTALL).group(1)
        lines = [line.strip() for line in comparison.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
            
        # Verify comparison line format
        pattern = re.compile(r'^[^|]+\|[^|]+\|[^|]+\|(-1|0|1)\|\s?\d+%$')
        if not all(pattern.match(line) for line in lines[1:]):
            return False

        # Analysis content check
        analysis_content = re.search(r'====BEGIN ANALYSIS====(.*?)====END ANALYSIS====', text, re.DOTALL).group(1).strip()
        if len(analysis_content) < 50:  # Minimum analysis length
            return False

        return True
        
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False


def parse_output(text:str):
    """
    把模型输出解析成 dict, 并给出 verdict_code (1/-1/0).
    """
    verdict_code = None
    m = re.search(r'Final Verdict:\s*(Image1|Tie|Image2)', text)
    if m:
        if m.group(1).lower().startswith("image1"):
            verdict_code = 1
        elif m.group(1).lower().startswith("image2"):
            verdict_code = -1
        else:
            verdict_code = 0
    return {"verdict_code": verdict_code, "text": text}

# -----------------------------------------------------------------------------
#  5. 数据集实现
# -----------------------------------------------------------------------------
class HpdPairDataset(Dataset):
    """
    直接读 jsonl 其中每行形如
    {
      "prompt": "...",
      "human_preference": [score0, score1],
      "image": [{"bytes": ...}, {"bytes": ...}]
    }
    """
    def __init__(self, path:str):
        self.samples = []
        with open(os.path.join(path, "data.jsonl"), "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# -----------------------------------------------------------------------------
#  6. 构建 reward_fn (完全可微要求不高, 只返回标量 reward)
# -----------------------------------------------------------------------------
def build_reward_fn(toolbox):
    model, processor = toolbox                   # 解包

    def _inner(image1:Image.Image,
               image2:Image.Image,
               prompt:str) -> Dict[str,Any]:
        """
        复刻你之前 _fn 的核心, 但做了删减:
        1. 把 image1,image2 拼成一张送入模型;
        2. 用 generate 得到答案 text;
        3. 给出 reward & debug 信息.
        """
        # -- 拼图
        big_img = encodeAsPIL([image1, image2])

        # -- 构造 chat-template
        msg = [{
            "role":"user",
            "content":[
                {"type":"image"},
                {"type":"text","text":FullPmt.format(locPrompt=prompt)}
            ]
        }]
        text_prompt = processor.apply_chat_template(msg, add_generation_prompt=True)

        # processor() 里 images 要有占位，但我们会自己塞 pixel_values
        model_inputs = processor(text=[text_prompt],
                                 images=[big_img],
                                 padding=True,
                                 return_tensors="pt").to(model.device)

        # -- 推理
        gen = model.generate(**model_inputs,
                             max_new_tokens=1024,
                             do_sample=True,
                             temperature=0.8)
        ans = processor.batch_decode(gen, skip_special_tokens=True)[0]

        # -- 解析 + 打分
        fmt_ok = validate_format(ans)
        parsed  = parse_output(ans)
        verdict = parsed["verdict_code"]

        # 非常简单的 reward 公式：格式正确 + verdict
        reward = 1.0 * fmt_ok
        if verdict is not None:
            reward += verdict          # Image1优则+1, 劣则-1, Tie 0

        return {"reward": reward, "text": ans}

    # ================
    # 返回一个 Callable
    # ================
    def reward_fn(batch_text:List[str], raw_batch:List[Dict[str,Any]]) -> torch.Tensor:
        rewards = []
        for sample in raw_batch:
            img0, img1 = map(process_to_IMG, [sample["image"][0]["bytes"], sample["image"][1]["bytes"]])
            out = _inner(img0, img1, sample["prompt"])
            rewards.append(out["reward"])
            # 在 reward_fn 里直接 wandb 打日志也行，这里略
        return torch.tensor(rewards, dtype=torch.float32, device="cuda")

    return reward_fn

# -----------------------------------------------------------------------------
#  7. collate_fn – 负责把 prompt + pixel_values 打包
# -----------------------------------------------------------------------------
def make_collate_fn(processor:AutoProcessor):
    def collate(batch:List[Dict[str,Any]]):
        # 把两张图都拼好
        imgs   = []
        prompts= []
        for sample in batch:
            img0,img1 = map(process_to_IMG,
                            [sample["image"][0]["bytes"], sample["image"][1]["bytes"]])
            big_img   = encodeAsPIL([img0,img1])
            imgs.append(big_img)

            # 同一段文字 prompt
            msg = [{
                "role":"user",
                "content":[
                    {"type":"image"},
                    {"type":"text","text":FullPmt.format(locPrompt=sample["prompt"])}
                ]
            }]
            prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True))

        # tokenizer 会把 <image> placeholder 插进去
        model_inputs = processor(text=prompts,
                                images=imgs,
                                padding=True,
                                return_tensors="pt")
        return model_inputs, batch          # 附带原始样本用于 reward
    return collate

# -----------------------------------------------------------------------------
#  8. LoRA-PPO Trainer 子类 – 让 generate 能吃 pixel_values
# -----------------------------------------------------------------------------
class VLPPOTrainer(PPOTrainer):
    def generate(self, model_inputs:Dict[str,torch.Tensor], **gen_kwargs):
        pixel_values = model_inputs.pop("pixel_values")
        with torch.no_grad():
            seq = self.model.generate(
                **model_inputs,
                pixel_values=pixel_values,
                max_new_tokens=MAX_NEW_TOK,
                **gen_kwargs,
            )
        return seq

# -----------------------------------------------------------------------------
#  9. 主函数
# -----------------------------------------------------------------------------
def main():
    seed_everything(SEED)

    # ------- accelerator & wandb --------
    accelerator = Accelerator(log_with="wandb")
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=WANDB_PROJ,
            init_kwargs={"wandb":{"name":RUN_NAME}},
        )

    # ------- model + LoRA ---------------
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", 
        torch_dtype="auto", 
        device_map="auto", 
        cache_dir="/m2v_intern/wangqunzhong/research/huggingface/model/Qwen/Qwen-VL-7B-Chat",
    )
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=32,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_cfg)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    toolbox   = (model, processor)

    # ------- dataset --------------------
    # dataset  = HpdPairDataset(DATASET_PATH)
    dataset = load_dataset(DATASET_PATH, split="train", streaming = True)
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
    

    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=make_collate_fn(processor))
    
    class RewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = base_model  # 预训练模型
            self.regression_head = nn.Linear(base_model.config.hidden_size, 1)
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids, attention_mask)
            last_hidden_states = outputs.last_hidden_state[:, 0, :]  # 取[CLS]向量
            reward = self.regression_head(last_hidden_states)
            return reward
    # ------- PPO trainer ---------------
    ppo_cfg = PPOConfig(
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        mini_batch_size=1,
        num_ppo_epochs=PPO_EPOCHS,
        #log_with=None,      
        #accelerator=accelerator,          
    )
    trainer = VLPPOTrainer(
        args=ppo_cfg,
        model=model,
        ref_model=None,
        processing_class=processor.tokenizer,
        
    )

    reward_fn = build_reward_fn(toolbox)

    # ==============================================================
    #  RL 训练主循环
    # ==============================================================
    for step,(model_inputs,raw_batch) in enumerate(dataloader):
        # 把张量搬到 device
        breakpoint()
        model_inputs = {k:v.to(accelerator.device) for k,v in model_inputs.items()}

        # 1. rollout
        responses_ids = trainer.generate(model_inputs)
        responses_txt = processor.batch_decode(
            responses_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 2. reward
        rewards = reward_fn(responses_txt, raw_batch)  # tensor[bs]

        # 3. PPO 更新
        trainer.step(model_inputs["input_ids"], responses_ids, rewards)

        # 4. 日志
        if accelerator.is_main_process and step % 10 == 0:
            wandb.log({
                "step": step,
                "reward/mean": rewards.mean().item(),
                "reward/std":  rewards.std().item(),
                "lr": trainer.optimizer.param_groups[0]["lr"],
            })

    accelerator.end_training()


if __name__ == "__main__":
    main()