from PIL import Image
import io
import numpy as np
import torch
import requests
import json
import base64
import re
import yaml
import sys
import os
import os.path as osp
import random

import concurrent.futures

from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

import cv2
from decord import VideoReader
import pandas as pd
from tqdm import tqdm


sys.path.append("/video/wangjiahao08/workspace/data_engine")
sys.path.append("/video/wangjiahao08/workspace/data_engine/gpt4_utils")

from openai import OpenAI
from gpt4_utils.gpt4o_api import GPT4o_Service
from mmu.media_common_pb2 import ImgUnit



random.seed(42)

bizs = ['liangjiajun_939ed227_gpt-4o-2024-08-06']

def evaluate_promptImagePair_Openrouter():
    
    with open("ddpo_pytorch/prompt_relative.yaml") as f:
        data = yaml.safe_load(f)

    CKLST_PROMPT = data["cklst-prompt"]
    EVAL_PROMPT = data["eval-prompt"]
    FLITER_PROMPT = data['filter-prompt']

    COMP_MAPPING = {
    "1 >> 2": 2,
    "1 > 2": 1,
    "1 = 2": 0,
    "1 < 2": -1,
    "1 << 2": -2,
    "2 >> 1": -2,
    "2 > 1": -1,
    "2 = 1": 0,
    "2 < 1": 1,
    "2 << 1": 2
    }

    MODEL= "qwen/qwen2.5-vl-32b-instruct"

    def resize_image_tensor(image_tensor, target_height, target_width):
        #
        image = Image.fromarray((image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
        
        #
        resized_image = image.resize((target_width, target_height))
        resized_image = np.array(resized_image)

        resized_image = resized_image.astype('float32')
        
        #
        resized_tensor = (
            torch.tensor(resized_image).clone().detach().permute(2, 0, 1) / 255
        ).float()
    
        return resized_tensor

    def concatenate_images(image_tensor1, image_tensor2, dim=1):
        h1, w1 = image_tensor1.shape[1], image_tensor1.shape[2]
        h2, w2 = image_tensor2.shape[1], image_tensor2.shape[2]
        
        #
        target_height = max(h1, h2)
        target_width = max(w1, w2)

        #
        image_tensor1_resized = resize_image_tensor(image_tensor1, target_height, target_width)
        image_tensor2_resized = resize_image_tensor(image_tensor2, target_height, target_width)

        #
        concatenated_tensor = torch.cat((image_tensor1_resized, image_tensor2_resized), dim=dim)
        
        return concatenated_tensor

    def get_image_base64(image):
        """
        Accepts an image (NumPy array or PyTorch tensor), processes it, and 
        returns the Base64-encoded data URL string.
        """

        # Ensure the image is in the correct format and data type
        if isinstance(image, torch.Tensor):
            # 如果输入是 torch.Tensor，将其转换为 uint8 格式
            image = (image * 255).round().clamp(0, 255).to(torch.uint8)
        elif isinstance(image, np.ndarray):
            # 如果输入是 NumPy 数组，确保其为 NHWC 格式
            if image.shape[0] == 3:  # 假设是 CHW 格式
                image = image.transpose(1, 2, 0)  # 转换为 HWC 格式
            image = torch.tensor(image, dtype=torch.uint8)
        elif isinstance(image, Image.Image):
            # 如果输入是 PIL.Image.Image，将其转换为 NumPy 数组
            image = np.array(image)  # 转换为 NumPy 数组
            if image.ndim == 2:  # 如果是灰度图（单通道）
                image = np.expand_dims(image, axis=-1)  # 添加通道维度
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image, dtype=torch.uint8)
        else:
            # 如果输入类型不支持，抛出异常
            raise TypeError("Unsupported image type. Must be a NumPy array, PyTorch tensor, or PIL image.")
        
        # Encode the image to Base64
        image = image.permute(1, 2, 0).cpu().numpy()
        # Convert the image to a PIL Image
        pil_image = Image.fromarray(image)
        # Save the image to a bytes buffer in JPEG format
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)
        # Encode the image bytes to Base64
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        # Return the Base64-encoded data URL
        return f"data:image/jpeg;base64,{image_base64}"
    


    def _fn(image1,image2, prompt=""):
        """
        判断函数
        """

        DEBUG = True
        import random

        # 参数
        mean = 1          # 均值
        std_dev = 2       # 标准差（方差 = std_dev^2 = 4）

        # 生成随机数
        random_value = random.gauss(mean, std_dev)

        return int(random_value)
        
        # Initialize the OpenAI client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",  # Replace with your OpenRouter API key
        )

        # Path to the image
        to_tensor = transforms.ToTensor()

        image1 = to_tensor(image1)  
        image2 = to_tensor(image2)
        image = concatenate_images(image1, image2)
        to_pil = transforms.ToPILImage()
        image = to_pil(image)
        #image.save(f"log_pic/image_{prompt}.png")

        base64_image_url = get_image_base64(image)

        # generate CKLST
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": CKLST_PROMPT.format(locPrompt=prompt)
                        },
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                                "url": base64_image_url,
                            }
                        },
                    ]
                }
            ]
        )

        CKLST = completion.choices[0].message.content
        QSLST = re.findall(r'\d+\.\s\*\*Question:\*\*\s(.*?)\s\s', CKLST)
        QSLST = list(dict.fromkeys(QSLST))
        locQSLST = " ".join(QSLST)

        if False:
            print(prompt)
            print("[Begin CKLST]")
            print(CKLST)
            print("[QSLST]")
            print(locQSLST)

        
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt)
                        },
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                                "url": base64_image_url,
                            }
                        },
                    ]
                }
            ]
        )
        EVAL = completion.choices[0].message.content 

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": FLITER_PROMPT.format(locQslst=locQSLST)
                        },
                        {
                            "type": "image_url",
                            "image_url": 
                            {
                                "url": base64_image_url,
                            }
                        },
                    ]
                }
            ]
        )
        SCORE =  completion.choices[0].message.content

        question_pattern = r"\*{0,2}Question\*{0,2}: (.*?\?)"
        importance_pattern = r"\*{0,2}Importance Score\*{0,2}: (\d)"
        questions = re.findall(question_pattern, SCORE, re.S)
        importance_scores = re.findall(importance_pattern, SCORE)

        # 构造字典
        #Filter_Dict = {q.strip(): int(score.strip()) for q, score in zip(questions, importance_scores)}

        

        principle_pattern = r"\*{0,2}Principle\*{0,2}: (.*)"
        comparison_levels = r"\*{0,2}Comparison Level\*{0,2}:\*{0,2} (.*?)\n"
        principles = re.findall(principle_pattern, EVAL)
        comparison_levels = re.findall(comparison_levels, EVAL)

        #构造字典并映射评分
        Eval_Dict = {

        }
        for principle, comparison in zip(principles, comparison_levels):
            princ = principle.strip("* ")
            compri = comparison.strip("* ")
            try:
                Eval_Dict[princ] = COMP_MAPPING[compri]
            except Exception:
                print("can't fine key")
            

        Eval_Dict = {
            principle.strip("* "): COMP_MAPPING[comparison.strip("* ")]
            for principle, comparison in zip(principles, comparison_levels)
        }


        final_score = 0
        rwr = 0

        for eva, score in Eval_Dict.items():
            # eva_importance = Filter_Dict[eva]
            final_score += score
            # rwr += eva_importance

        if DEBUG:
            #print("[Score]")
            #print(SCORE)
            #print(Filter_Dict)
            print("[JUDGE]")
            print(EVAL)
            #print(Eval_Dict)
            #print("[Score]")
            #print(final_score)
        
        print(final_score, Eval_Dict)
        return final_score, Eval_Dict
    
    return _fn

def evaluate_promptImagePair_GPT4():
    
    with open("ddpo_pytorch/prompt_relative.yaml") as f:
        data = yaml.safe_load(f)

    CKLST_PROMPT = data["cklst-prompt"]
    EVAL_PROMPT = data["eval-prompt"]
    FLITER_PROMPT = data['filter-prompt']

    COMP_MAPPING = {
    "1 >> 2": 2,
    "1 > 2": 1,
    "1 = 2": 0,
    "1 < 2": -1,
    "1 << 2": -2,
    "2 >> 1": -2,
    "2 > 1": -1,
    "2 = 1": 0,
    "2 < 1": 1,
    "2 << 1": 2
    }

    def read_frame(imagelist, target_size=512):
        images = []
        total_width = 0
        max_height = 0

        for image_data in imagelist:
            img = image_data.convert("RGB")
            original_width, original_height = img.size

            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
            img = img.resize((new_width, new_height) )#, Image.ANTIALIAS)
            images.append(img)

            total_width += new_width
            max_height = max(max_height, new_height)

        combined_image = Image.new('RGB', (total_width + 5*(len(imagelist) - 1), max_height))
        

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 5
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
    
        img_byte_arr = img_byte_arr.getvalue()

        img_unit = ImgUnit(image=img_byte_arr)

        return img_unit
    


    def _fn(image1,image2, prompt="", Error=None, DEBUG=False, ANALYSIS=None, RECORD=False):
        """
        打分
        """
        
        imagelist = [image1, image2]
        img_unit = read_frame(imagelist)

        # Initialize the OpenAI client
        biz = bizs[0 % len(bizs)]
        client = GPT4o_Service(biz)
        
        lth = 0

        #CKLST
        ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
        cnt_1 = 1
        while ans is None:
            print("[WAITING]")
            ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
            cnt_1 += 1

        lth += len(ans)
        CKLST = ans
        QSLST = re.findall(r'\d+\.\s\*\*Question:\*\*\s(.*?)\s\s', CKLST)
        QSLST = list(dict.fromkeys(QSLST))
        locQSLST = " ".join(QSLST)

        ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
        cnt_2 = 1
        while ans2 is None:
            print("[WAITING]")
            ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
            cnt_2 += 1

        lth += len(ans2)
        EVAL = ans2
        principle_pattern = r"\*{0,2}Principle\*{0,2}: (.*)"
        comparison_levels = r"\*{0,2}Comparison Level\*{0,2}:\*{0,2} (.*?)\n"
        principles = re.findall(principle_pattern, EVAL)
        comparison_levels = re.findall(comparison_levels, EVAL)

        Eval_Dict = {}
        
        NumberOfWrongCompRES = 0
        WrongCompRES = []
            
        for principle, comparison in zip(principles, comparison_levels):
            try:
                Eval_Dict[principle.strip("* ")] = COMP_MAPPING[comparison.strip("* ")]
            except Exception:
                NumberOfWrongCompRES += 1
                WrongCompRES.append(comparison.strip("* "))

        final_score = 0

        for eva, score in Eval_Dict.items():
            final_score += score
        
        isNULLANS = False
        if len(Eval_Dict)==0:
            isNULLANS = True

        if not isNULLANS:
            posANS, negANS = sum(k > 0 for k in Eval_Dict.values()), sum(k < 0 for k in Eval_Dict.values())
            posRATE, negRATE = posANS/len(Eval_Dict), negANS/len(Eval_Dict)
        else:
            posANS, negANS, posRATE, negRATE = 0,0,0,0
        
        if not principles:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()
        elif not Eval_Dict:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()

        ERROR_DICT = {
            "NumberOfWrongCompRES":NumberOfWrongCompRES,
            "WrongCompRES": WrongCompRES,
            "FIRST RESPOND TIME DELAY":cnt_1,
            "SECOND RESPOND TIME DELAY":cnt_2,
            "Is Null Answer":isNULLANS
        }

        ANS_DICT = {
            "Positive Rate":posRATE,
            "Negative Rate":negRATE,
            "Null Rate": 1 - posRATE - negRATE,
            "TOTAL": len(Eval_Dict),
            "Lenth":lth,
        }

        if DEBUG:
            print("[Score]")
            print(final_score)
            print("[DICT]")
            print(Eval_Dict)

        if RECORD:
            ANALYSIS.write(ans2)
            ANALYSIS.flush()

        return final_score, ERROR_DICT, ANS_DICT
    
    return _fn

def evaluate_promptImagePairDirectAnswer_GPT4():
    
    with open("ddpo_pytorch/prompt_directanswer.yaml") as f:
        data = yaml.safe_load(f)

    CKLST_PROMPT = data["cklst-prompt"]
    EVAL_PROMPT = data["eval-prompt"]

    COMP_MAPPING = {
        "1 >> 2": 2,
        "1 > 2": 1,
        "1 = 2": 0,
        "1 < 2": -1,
        "1 << 2": -2,
        "2 >> 1": -2,
        "2 > 1": -1,
        "2 = 1": 0,
        "2 < 1": 1,
        "2 << 1": 2
    }

    ANSWER_MAPPING = {
        "Image 1": 1,
        "Image 2": -1,
    #   "Tie": 0
    }

    def read_frame(imagelist, target_size=512):
        images = []
        total_width = 0
        max_height = 0

        for image_data in imagelist:
            img = image_data.convert("RGB")
            original_width, original_height = img.size

            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
            img = img.resize((new_width, new_height) )#, Image.ANTIALIAS)
            images.append(img)

            total_width += new_width
            max_height = max(max_height, new_height)

        combined_image = Image.new('RGB', (total_width + 5*(len(imagelist) - 1), max_height))
        

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 5
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
    
        img_byte_arr = img_byte_arr.getvalue()

        img_unit = ImgUnit(image=img_byte_arr)

        return img_unit
    
    def _fn(image1,image2, prompt="", Error=None, DEBUG=False, ANALYSIS=None, RECORD=False):
        """
        打分
        """
        imagelist = [image1, image2]
        img_unit = read_frame(imagelist)

        # Initialize the OpenAI client
        biz = bizs[0 % len(bizs)]
        client = GPT4o_Service(biz)
        
        lth = 0

        #CKLST
        ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
        cnt_1 = 1
        while ans is None:
            print("[WAITING]")
            ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
            cnt_1 += 1

        lth += len(ans)
        CKLST = ans
        QSLST = re.findall(r'\d+\.\s\*\*Question:\*\*\s(.*?)\s\s', CKLST)
        QSLST = list(dict.fromkeys(QSLST))
        locQSLST = " ".join(QSLST)

        ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
        cnt_2 = 1
        while ans2 is None:
            print("[WAITING]")
            ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
            cnt_2 += 1

        lth += len(ans2)
        EVAL = ans2
        principle_pattern = r"\*{0,2}Principle\*{0,2}: (.*)"
        comparison_levels = r"\*{0,2}Comparison Level\*{0,2}:\*{0,2} (.*?)\n"
        principles = re.findall(principle_pattern, EVAL)
        comparison_levels = re.findall(comparison_levels, EVAL)

        final_answer_pattern = r"<Final answer>:\s*\*\*Better Image\*\*:\s*(Image 1|Image 2)"

        Final_answer = re.search(final_answer_pattern, ans2)
        isNULLANS = False
        if Final_answer is None:
            isNULLANS = True
            reward = None
        else:
            Final_answer = Final_answer.group(1)
            reward = ANSWER_MAPPING.get(Final_answer, None)
        if reward is None:
            isNULLANS = True

        Eval_Dict = {}
        NumberOfWrongCompRES = 0
        WrongCompRES = []
        for principle, comparison in zip(principles, comparison_levels):
            try:
                Eval_Dict[principle.strip("* ")] = COMP_MAPPING[comparison.strip("* ")]
            except Exception:
                NumberOfWrongCompRES += 1
                WrongCompRES.append(comparison.strip("* "))

        final_score = 0
        for eva, score in Eval_Dict.items():
            final_score += score
        
        isNULLCOMP = False
        if len(Eval_Dict)==0:
            isNULLCOMP = True
        if not isNULLCOMP:
            posANS, negANS = sum(k > 0 for k in Eval_Dict.values()), sum(k < 0 for k in Eval_Dict.values())
            posRATE, negRATE = posANS/len(Eval_Dict), negANS/len(Eval_Dict)
        else:
            posANS, negANS, posRATE, negRATE = 0,0,0,0
        if not principles:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()
        elif not Eval_Dict:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()
        elif not reward:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()

        ERROR_DICT = {
            "NumberOfWrongCompRES":NumberOfWrongCompRES,
            "WrongCompRES": WrongCompRES,
            "FIRST RESPOND TIME DELAY":cnt_1,
            "SECOND RESPOND TIME DELAY":cnt_2,
            "Is Null Answer":isNULLANS,
            "Is Null Compare": isNULLCOMP,
        }

        ANS_DICT = {
            "Positive Rate":posRATE,
            "Negative Rate":negRATE,
            "Null Rate": 1 - posRATE - negRATE,
            "TOTAL": len(Eval_Dict),
            "Lenth":lth,
            "Comp Score": final_score
        }

        if DEBUG:
            print("[Score]")
            print(final_score)
            print("[DICT]")
            print(Eval_Dict)

        if RECORD:
            ANALYSIS.write(ans2+"\n")
            ANALYSIS.flush()

        return reward, ERROR_DICT, ANS_DICT
    
    return _fn

def evaluate_promptImagePairDirectAnswerWithTie_GPT4():
    
    with open("ddpo_pytorch/prompt_directanswer.yaml") as f:
        data = yaml.safe_load(f)
    with open("ddpo_pytorch/cklst_prompt.yaml") as f:
        cklstpmt = yaml.safe_load(f)
    with open("ddpo_pytorch/eval_prompt.yaml") as f:
        evalpmt = yaml.safe_load(f)

    COMP_MAPPING = {
        "1 >> 2": 2,
        "1 > 2": 1,
        "1 = 2": 0,
        "1 < 2": -1,
        "1 << 2": -2,
        "2 >> 1": -2,
        "2 > 1": -1,
        "2 = 1": 0,
        "2 < 1": 1,
        "2 << 1": 2
    }
    ANSWER_MAPPING = {
        "Image 1": 1,
        "Image 2": -1,
        "Tie": 0
    }

    def read_frame(imagelist, target_size=512):
        images = []
        total_width = 0
        max_height = 0

        for image_data in imagelist:
            img = image_data.convert("RGB")
            original_width, original_height = img.size

            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
            img = img.resize((new_width, new_height) )#, Image.ANTIALIAS)
            images.append(img)

            total_width += new_width
            max_height = max(max_height, new_height)

        combined_image = Image.new('RGB', (total_width + 5*(len(imagelist) - 1), max_height))
        

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 5
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
    
        img_byte_arr = img_byte_arr.getvalue()

        img_unit = ImgUnit(image=img_byte_arr)

        return img_unit
    
    def _fn(image1,image2, prompt="", Error=None, DEBUG=False, ANALYSIS=None, RECORD=False,CKLST="4-22-3question"):
        """打分"""
        DirInfer = False
        if CKLST == "4-22-0question":
            DirInfer = True 
        if not DirInfer:
            CKLST_PROMPT = cklstpmt[CKLST]
        else:
            CKLST_PROMPT = ""
        EVAL_PROMPT = evalpmt["4-22-1555"]
        imagelist = [image1, image2]
        img_unit = read_frame(imagelist)

        # Initialize the OpenAI client
        biz = bizs[0 % len(bizs)]
        client = GPT4o_Service(biz)
        
        lth = 0
        if not DirInfer:
            #CKLST
            ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
            cnt_1 = 1
            while ans is None:
                print("[WAITING]")
                ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
                cnt_1 += 1

            lth += len(ans)
            CKLST = ans
            QSLST = re.findall(r'\d+\.\s\*\*Question:\*\*\s(.*?)\s\s', CKLST)
            QSLST = list(dict.fromkeys(QSLST))
            locQSLST = " ".join(QSLST)
        else:
            cnt_1 = 0
            locQSLST = "Not Available, please direct compare Image1 & Image2"
        
        ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
        cnt_2 = 1
        while ans2 is None:
            print("[WAITING]")
            ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
            cnt_2 += 1

        lth += len(ans2)
        EVAL = ans2
        principle_pattern = r"\*{0,2}Principle\*{0,2}: (.*)"
        comparison_levels = r"\*{0,2}Comparison Level\*{0,2}:\*{0,2} (.*?)\n"
        principles = re.findall(principle_pattern, EVAL)
        comparison_levels = re.findall(comparison_levels, EVAL)

        final_answer_pattern = r"<Final answer>:\s*\*\*Better Image\*\*:\s*(Image 1|Image 2|Tie)\.?"
        Final_answer = re.search(final_answer_pattern, ans2)

        isNULLANS = False
        if Final_answer is None:
            isNULLANS = True
            reward = None
        else:
            Final_answer = Final_answer.group(1)
            reward = ANSWER_MAPPING.get(Final_answer, None)
        if reward is None:
            isNULLANS = True

        Eval_Dict = {}
        NumberOfWrongCompRES = 0
        WrongCompRES = []
        for principle, comparison in zip(principles, comparison_levels):
            try:
                Eval_Dict[principle.strip("* ")] = COMP_MAPPING[comparison.strip("* ")]
            except Exception:
                NumberOfWrongCompRES += 1
                WrongCompRES.append(comparison.strip("* "))

        final_score = 0
        for eva, score in Eval_Dict.items():
            final_score += score
        
        isNULLCOMP = False
        if len(Eval_Dict)==0:
            isNULLCOMP = True
        if not isNULLCOMP:
            posANS, negANS = sum(k > 0 for k in Eval_Dict.values()), sum(k < 0 for k in Eval_Dict.values())
            posRATE, negRATE = posANS/len(Eval_Dict), negANS/len(Eval_Dict)
        else:
            posANS, negANS, posRATE, negRATE = 0,0,0,0
        if not principles:
            Error.write("RESPOND\n"+ EVAL + "\n")
            Error.flush()
        elif not Eval_Dict:
            Error.write("RESPOND\n"+ EVAL + "\n")
            Error.flush()
        elif not reward:
            Error.write("RESPOND\n"+ EVAL + "\n")
            Error.flush()

        ERROR_DICT = {
            "NumberOfWrongCompRES":NumberOfWrongCompRES,
            "WrongCompRES": WrongCompRES,
            "FIRST RESPOND TIME DELAY":cnt_1,
            "SECOND RESPOND TIME DELAY":cnt_2,
            "Is Null Answer":isNULLANS,
            "Is Null Compare": isNULLCOMP,
        }

        ANS_DICT = {
            "Positive Rate":posRATE,
            "Negative Rate":negRATE,
            "Null Rate": 1 - posRATE - negRATE,
            "TOTAL": len(Eval_Dict),
            "Lenth":lth,
            "Comp Score": final_score,
            "ANS":Eval_Dict,
        }

        if DEBUG:
            print("[Score]")
            print(final_score)
            print("[DICT]")
            print(Eval_Dict)

        if RECORD:
            ANALYSIS.write(ans2 + "\n")
            ANALYSIS.flush()

        return reward, ERROR_DICT, ANS_DICT
    
    return _fn

def evaluate_direct_GPT4():
    
    with open("ddpo_pytorch/prompt_single.yaml") as f:
        data = yaml.safe_load(f)

    PROMPT = data["prompt"]
    

    COMP_MAPPING = {
    "1 >> 2": 2,
    "1 > 2": 1,
    "1 = 2": 0,
    "1 < 2": -1,
    "1 << 2": -2,
    "2 >> 1": -2,
    "2 > 1": -1,
    "2 = 1": 0,
    "2 < 1": 1,
    "2 << 1": 2
    }

    def read_frame(imagelist, target_size=512):
        images = []
        total_width = 0
        max_height = 0

        for image_data in imagelist:
            img = image_data.convert("RGB")
            original_width, original_height = img.size

            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
            img = img.resize((new_width, new_height) )#, Image.ANTIALIAS)
            images.append(img)

            total_width += new_width
            max_height = max(max_height, new_height)

        combined_image = Image.new('RGB', (total_width + 5*(len(imagelist) - 1), max_height))
        

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 5
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
    
        img_byte_arr = img_byte_arr.getvalue()

        img_unit = ImgUnit(image=img_byte_arr)

        return img_unit
    


    def _fn(image1,image2, prompt="", Error=None, DEBUG=False, ANALYSIS=None, RECORD = False):
        """
        打分
        """
        
        imagelist = [image1, image2]
        img_unit = read_frame(imagelist)

        # Initialize the OpenAI client
        biz = bizs[0 % len(bizs)]
        client = GPT4o_Service(biz)

        lth = 0

        ans = client.run(PROMPT.format(locPrompt=prompt), images=[img_unit])
        
        cnt_1 = 1
        while ans is None:
            print("[WAITING]")
            ans = client.run(PROMPT.format(locPrompt=prompt), images=[img_unit])
            cnt_1 += 1

        lth += len(ans)

        better_image_pattern = r"\*\*Better Image\*\*: (Image 1|Image 2)"

        better_image = re.search(better_image_pattern, ans)
        
        better_image = better_image.group(1) if better_image else None

        isNULLANS = 0
        if better_image is None:
            isNULLANS = True
        else:
            better_image = 1 if better_image=="Image 1" else -1

        ERROR_DICT = {
            "FIRST RESPOND TIME DELAY":cnt_1,
            "Is Null Answer":isNULLANS
        }

        ANS_DICT = {
            "Lenth":lth,
        }
        if DEBUG:
            print(better_image, ans)
        if RECORD:
            ANALYSIS.write(ans)

        return better_image, ERROR_DICT, ANS_DICT
    
    return _fn


def evaluate_promptImagePairDirectAnswerWithHistory_GPT4():
    
    with open("ddpo_pytorch/prompt_directanswer.yaml") as f:
        data = yaml.safe_load(f)

    CKLST_PROMPT = data["cklst-prompt"]
    EVAL_PROMPT = data["eval-prompt"]

    COMP_MAPPING = {
        "1 >> 2": 2,
        "1 > 2": 1,
        "1 = 2": 0,
        "1 < 2": -1,
        "1 << 2": -2,
        "2 >> 1": -2,
        "2 > 1": -1,
        "2 = 1": 0,
        "2 < 1": 1,
        "2 << 1": 2
    }

    ANSWER_MAPPING = {
        "Image 1": 1,
        "Image 2": -1,
    #   "Tie": 0
    }

    def read_frame(imagelist, target_size=512):
        images = []
        total_width = 0
        max_height = 0

        for image_data in imagelist:
            img = image_data.convert("RGB")
            original_width, original_height = img.size

            new_height = target_size
            new_width = int((target_size / original_height) * original_width)
            img = img.resize((new_width, new_height) )#, Image.ANTIALIAS)
            images.append(img)

            total_width += new_width
            max_height = max(max_height, new_height)

        combined_image = Image.new('RGB', (total_width + 5*(len(imagelist) - 1), max_height))
        

        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.width + 5
        img_byte_arr = io.BytesIO()
        combined_image.save(img_byte_arr, format='JPEG')
    
        img_byte_arr = img_byte_arr.getvalue()

        img_unit = ImgUnit(image=img_byte_arr)

        return img_unit
    
    def _fn(image1,image2, prompt="", Error=None, DEBUG=False, ANALYSIS=None, RECORD=False):
        """
        打分
        """
        imagelist = [image1, image2]
        img_unit = read_frame(imagelist)

        # Initialize the OpenAI client
        biz = bizs[0 % len(bizs)]
        client = GPT4o_Service(biz)
        
        lth = 0

        #CKLST
        ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
        cnt_1 = 1
        while ans is None:
            print("[WAITING]")
            ans = client.run(CKLST_PROMPT.format(locPrompt=prompt), images=[img_unit])
            cnt_1 += 1

        lth += len(ans)
        CKLST = ans
        QSLST = re.findall(r'\d+\.\s\*\*Question:\*\*\s(.*?)\s\s', CKLST)
        QSLST = list(dict.fromkeys(QSLST))
        locQSLST = " ".join(QSLST)

        ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
        cnt_2 = 1
        while ans2 is None:
            print("[WAITING]")
            ans2 = client.run(EVAL_PROMPT.format(locQSLST=locQSLST, locPrompt=prompt), images=[img_unit])
            cnt_2 += 1

        lth += len(ans2)
        EVAL = ans2
        principle_pattern = r"\*{0,2}Principle\*{0,2}: (.*)"
        comparison_levels = r"\*{0,2}Comparison Level\*{0,2}:\*{0,2} (.*?)\n"
        principles = re.findall(principle_pattern, EVAL)
        comparison_levels = re.findall(comparison_levels, EVAL)

        final_answer_pattern = r"<Final answer>:\s*\*\*Better Image\*\*:\s*(Image 1|Image 2)"

        Final_answer = re.search(final_answer_pattern, ans2)
        isNULLANS = False
        if Final_answer is None:
            isNULLANS = True
            reward = None
        else:
            Final_answer = Final_answer.group(1)
            reward = ANSWER_MAPPING.get(Final_answer, None)
        if reward is None:
            isNULLANS = True

        Eval_Dict = {}
        NumberOfWrongCompRES = 0
        WrongCompRES = []
        for principle, comparison in zip(principles, comparison_levels):
            try:
                Eval_Dict[principle.strip("* ")] = COMP_MAPPING[comparison.strip("* ")]
            except Exception:
                NumberOfWrongCompRES += 1
                WrongCompRES.append(comparison.strip("* "))

        final_score = 0
        for eva, score in Eval_Dict.items():
            final_score += score
        
        isNULLCOMP = False
        if len(Eval_Dict)==0:
            isNULLCOMP = True
        if not isNULLCOMP:
            posANS, negANS = sum(k > 0 for k in Eval_Dict.values()), sum(k < 0 for k in Eval_Dict.values())
            posRATE, negRATE = posANS/len(Eval_Dict), negANS/len(Eval_Dict)
        else:
            posANS, negANS, posRATE, negRATE = 0,0,0,0
        if not principles:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()
        elif not Eval_Dict:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()
        elif not reward:
            Error.write("\nRESPOND\n"+ EVAL)
            Error.flush()

        ERROR_DICT = {
            "NumberOfWrongCompRES":NumberOfWrongCompRES,
            "WrongCompRES": WrongCompRES,
            "FIRST RESPOND TIME DELAY":cnt_1,
            "SECOND RESPOND TIME DELAY":cnt_2,
            "Is Null Answer":isNULLANS,
            "Is Null Compare": isNULLCOMP,
        }

        ANS_DICT = {
            "Positive Rate":posRATE,
            "Negative Rate":negRATE,
            "Null Rate": 1 - posRATE - negRATE,
            "TOTAL": len(Eval_Dict),
            "Lenth":lth,
            "Comp Score": final_score
        }

        if DEBUG:
            print("[Score]")
            print(final_score)
            print("[DICT]")
            print(Eval_Dict)

        if RECORD:
            ANALYSIS.write(ans2+"\n")
            ANALYSIS.flush()

        return reward, ERROR_DICT, ANS_DICT
    
    return _fn




def read_frame(imagelist, target_size=512):
    images = []
    total_width = 0
    max_height = 0

    for image_data in imagelist:
        img = image_data.convert("RGB")
        # img = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_width, original_height = img.size

        # if original_width > original_height:
        #     new_width = target_size
        #     new_height = int((target_size / original_width) * original_height)
        # else:
        new_height = target_size
        new_width = int((target_size / original_height) * original_width)
        img = img.resize((new_width, new_height) )#, Image.ANTIALIAS)
        images.append(img)

        total_width += new_width
        max_height = max(max_height, new_height)

    combined_image = Image.new('RGB', (total_width + 5*(len(imagelist) - 1), max_height))
    

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width + 5
    img_byte_arr = io.BytesIO()
    combined_image.save(img_byte_arr, format='JPEG')
    
    img_byte_arr = img_byte_arr.getvalue()
    #combined_image.save("test.jpeg")

    img_unit = ImgUnit(image=img_byte_arr)

    return img_unit

def process(batch):
    with open("ddpo_pytorch/prompt_relative.yaml") as f:
        data = yaml.safe_load(f)

    CKLST_PROMPT = data["cklst-prompt"]
    EVAL_PROMPT = data["eval-prompt"]
    FLITER_PROMPT = data['filter-prompt']

    COMP_MAPPING = {
    "1 >> 2": 2,
    "1 > 2": 1,
    "1 = 2": 0,
    "1 < 2": -1,
    "1 << 2": -2,
    "2 >> 1": -2,
    "2 > 1": -1,
    "2 = 1": 0,
    "2 < 1": 1,
    "2 << 1": 2
    }

    idx = 0

    image1 = batch["jpg_0"]
    image2 = batch["jpg_1"]
    prompt = batch["caption"]
    imagelist = [image1, image2]
    img_unit = read_frame(imagelist)

    biz = bizs[idx % len(bizs)]
    client = GPT4o_Service(biz)
    # CKLST_PROMPT.format(locPrompt=prompt)
    ans = client.run("What is in the image, plz describe it", images=[img_unit])
    cnt = 1
    print(cnt, ans)
    while ans is None:
        ans = client.run("What is in the image, plz describe it", images=[img_unit])
        cnt += 1
        print(cnt, ans)

    breakpoint()
    
    ans = client.run("HI")
    stat = not (ans is None)
    print(stat)
    return not (ans is None)



if __name__ == "__main__":
    idx = 0
    biz = bizs[idx % len(bizs)]
    client = GPT4o_Service(biz)

    dataset_name = "/m2v_intern/liujie/research/huggingface/dataset/imagereward/fidelity_rating_dataset"
    dataset = load_dataset(dataset_name, split="train", num_proc=2)
    dataset_length = len(dataset)
    index_list = list(range(0, dataset_length))
    #print(dataset_length)
    random.shuffle(index_list)
    #print(index_list)
    process( dataset[ index_list[0] ] )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        results = list(tqdm(executor.map(process, [ dataset[_] for _ in  index_list ])))

    
    breakpoint()



    for batch in dataloader:
        image1 = batch["jpg_0"]
        image2 = batch["jpg_1"]
        breakpoint()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        image1 = batch["jpg_0"]
        image2 = batch["jpg_1"]
        breakpoint()
        results = list(tqdm(executor.map(process, [ 0 for _ in range(8192)])))



    for _ in range(5):  # 随机抽取 200 条数据
        sample = random.choice(dataset)
        image1 = sample["jpg_0"]
        image2 = sample["jpg_1"]
        prompt = sample["caption"]
        fn = evaluate_promptImagePair_GPT4()
        fn(image1,image2, prompt)




