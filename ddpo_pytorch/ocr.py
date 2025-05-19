from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union
from PIL import Image

class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR奖励计算器
        :param use_gpu: 是否使用GPU加速PaddleOCR
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False  # 关闭不必要的日志输出
        )
        

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]) -> torch.Tensor:
        """
        计算OCR奖励
        :param images: 输入图像列表(PIL或numpy格式)
        :param prompts: 对应的目标文本列表
        :return: 奖励张量(CPU)
        """
        rewards = []
        
        # 确保输入长度一致
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # 转换图像格式
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR识别
                result = self.ocr.ocr(img, cls=False)
                
                # 提取识别文本（处理可能的多行结果）
                recognized_text = ''.join([res[1][0] for res in result[0]]) if result[0] else ''
                
                # 计算编辑距离奖励
                reward = -distance(recognized_text, prompt)
                
            except Exception as e:
                # 错误处理（如OCR解析失败）
                print(f"OCR processing failed: {str(e)}")
                reward = -len(prompt)  # 最大惩罚
                
            rewards.append(reward)

        # 转换为PyTorch张量
        return torch.tensor(rewards, dtype=torch.float32)