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
# from decord import VideoReader
import pandas as pd
from tqdm import tqdm

def sum(x):
    res = 0
    for y in x:
        if y is not None:
            res += y
    return res

def sum(x):
    res = 0
    for y in x:
        if y is not None:
            res += y
    return res

with open("config/pmt_yaml/qwen_full_rsps.yaml") as ym_file:
    data = yaml.safe_load(ym_file)
FullPmt = data["prompt"]

def evaluate_QwenVL2_7B(select="general"):
    
    def parse_output(text):
        result = {
            "final_verdict": None,  # 1 (Image 1), 0 (Tie), -1 (Image 2)
            "checklist_count": 0,  # Number of checklist dimensions
            "is_valid_format": False,  # If all required sections exist
            "analysis_length": 0,  # Total character count of analysis sections (Deep Analysis + Holistic Verdict)
            "total_length": len(text)  # Total character count of the entire text
        }
        
        # 1. Extract Final Verdict
        final_verdict_match = re.search(r"Answer:\s*(Image 1|Tie|Image 2|Image1|Image2|Image1/Tie/Image2)", text, re.IGNORECASE)
        if final_verdict_match:
            verdict = final_verdict_match.group(1).strip().lower()
            if verdict in ["image 1", "image1"]:
                result["final_verdict"] = 1
            elif verdict in ["tie","image1/tie/image2"]:
                result["final_verdict"] = 0
            elif verdict in ["image 2", "image2"]:
                result["final_verdict"] = -1
        
        # 2. Count Checklist Dimensions (based only on Checklist Generation section)
        checklist_section_match = re.search(r"Checklist Generation(.*?)(?:Detailed Comparison|Deep Analysis|Holistic Verdict)", text, re.IGNORECASE | re.DOTALL)
        if checklist_section_match:
            checklist_section = checklist_section_match.group(1)
            checklist_matches = re.findall(r"Dimension\s*\d+:", checklist_section, re.IGNORECASE)
            result["checklist_count"] = len(checklist_matches)
        
        # 3. Check Format Validity
        # Required sections: Checklist Generation, Detailed Comparison, Deep Analysis, Holistic Verdict
        required_sections = ["Checklist Generation", "Detailed Comparison", "Deep Analysis", "Holistic Verdict","Final Verdict"]
        section_found = all(re.search(rf"{section}", text, re.IGNORECASE) for section in required_sections)
        result["is_valid_format"] = section_found
        
        # 4. Extract Total Analysis Length (Deep Analysis + Holistic Verdict)
        analysis_section_match = re.search(r"Deep Analysis(.*?)(?:$)", text, re.IGNORECASE | re.DOTALL)
        if analysis_section_match:
            analysis_content = analysis_section_match.group(1).strip()
            result["analysis_length"] = len(analysis_content)
        

        return result
            
    def _fn_general(inputs, toolbox=None,accelerator=None,config=None):
        
        if toolbox is None:
            raise ValueError("pipe in Qwen")
        model, processor,logger = toolbox
        #config = accelerator.trackers[0].tracker.config
        inferenceState = config["inference"]
        invs = inputs.pop("invs")
        inputs = inputs.to("cuda")
        
        # generate方法
        #processor.tokenizer.padding_side = "left"
        pad_token_id = processor.tokenizer.pad_token_id
        #print(inputs.keys())
        # generated_outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=2048,
        #     pad_token_id=pad_token_id,  # 确保填充符正确
        #     return_dict_in_generate=False,  # 获取完整生成信息
        #     #use_cache=False,
        # )
        # generated_sequences = generated_outputs.sequences
        # attention_mask = (generated_sequences != pad_token_id).long()
        # input_length = inputs.input_ids.shape[1]
        #generated_sequences = model.generate(**inputs,max_new_tokens=1250, pad_token_id=pad_token_id,return_dict_in_generate=False, use_cache=True,)
        # generated_sequences = model.generate(
        #     **inputs,
        #     max_new_tokens=2048,
        #     pad_token_id=pad_token_id,  # 确保填充符正确
        #     return_dict_in_generate=False,  # 不需要返回完整生成信息
        #     use_cache=True,
        # )
        generate_kwargs = {}
        if not inferenceState:
            # 如果不是推理模式，设置多样性策略
            generate_kwargs = {
                "do_sample": True,           # 启用采样模式
                "temperature": 1.2,          # 增加随机性
                "top_k": 50,                 # 考虑前 50 个概率最高的词
                "top_p": 0.9,                # 核采样，动态控制候选词
                "num_return_sequences": 1    # 仅返回一个生成序列
            }
        # 调用生成方法
        generated_sequences = model.generate(
            **inputs,
            max_new_tokens=2048,               # 控制生成的最大长度
            pad_token_id=pad_token_id,         # 确保填充符正确
            return_dict_in_generate=False,     # 不需要返回完整生成信息
            use_cache=True,                    # 提高效率
            **generate_kwargs                  # 动态更新生成参数
        )
        attention_mask = (generated_sequences != pad_token_id).long()
        input_length = inputs.input_ids.shape[1]

        if not inferenceState:
            generated_output = model(
                input_ids=generated_sequences,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache = config["transformer"]["use_cache"],
            )
            
            #print(torch.cuda.memory_summary())
            logits = generated_output["logits"]
            generated_logits = logits  # (batch_size, generated_length, vocab_size)

            logp_list = []
            for i, output_ids in enumerate(generated_sequences[:, input_length:]):
                token_logits = generated_logits[i]
                probs = torch.softmax(token_logits, dim=-1)
                token_probs = probs[range(len(output_ids)), output_ids]
                logp = torch.log(token_probs)
                logp_list.append(logp)

            # 将 logp 转化为张量 (batch_size, generated_length)
            logp_tensor = torch.stack(logp_list, dim=0)
            total_logp = logp_tensor.mean(dim=1) 

        # 裁剪生成的 token，只保留生成部分
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids 
            in zip(inputs.input_ids, generated_sequences)
        ]
        
        # 解码生成的文本
        ans = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        if config.get("log_cot", None) and config.get("curr_batch_dir", None):
            for idx, sample in enumerate(ans):
                file_path = os.path.join(config["curr_batch_dir"], f"{idx}_cot.txt")
                with open(file_path, "w") as file:  # 使用上下文管理器
                    file.write(sample)

        #breakpoint()
        retInfo = []
        rewards = []
        fmts = []
        chz = []
        lth = []
        rzlths = []
        ckptcnts = []
        for txt,inv in zip(ans,invs):
            resDict = parse_output(txt)
            fmt = resDict["is_valid_format"]
            chiz = resDict["final_verdict"]
            rzlths.append(resDict["analysis_length"])
            #ckptcnts.append(resDict["checklist_count"])

            if not fmt:
                curr_reward = -0.5
            elif chiz is None:
                curr_reward = -0.5
            else:
                curr_reward =  int(fmt) + int( (1-2*inv) == chiz)*3 + int( chiz == 0 ) - 1.5
                if config["reward"]["reward_long_cot"]:
                    curr_reward += resDict["total_length"]/3000. - 0.95
            
            rewards.append(curr_reward)
            fmts.append(int(fmt))
            chz.append(int( (1-2*inv) == chiz)+0.5*(chiz==0))
            lth.append(len(txt))

        retInfo = {
            "fomat correctness": sum(fmts) / len(fmts)*1. if fmts else 0.,  # 防止列表为空
            "choose correctness": sum(chz) / len(chz)*1. if chz else 0.,
            "avg lth": sum(lth) / len(lth)*1. if lth else 0.,
            "avg reward": sum(rewards) / len(rewards)*1. if rewards else 0.,
            "avg reasoning": sum(rzlths) / len(rzlths)*1. if rzlths else 0.,
            #"avg check point count": sum(ckptcnts) / len(ckptcnts) if ckptcnts else 0,

        }
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(accelerator.device)
        chz_tensor = torch.tensor(chz, dtype=torch.float32).to(accelerator.device)
        # 使用 accelerator.gather() 收集所有卡上的 rewards_tensor
        all_rewards_tensor = accelerator.gather(rewards_tensor)
        all_chz_tensor = accelerator.gather(chz_tensor)
        # 在主进程上计算全局均值和标准差（跨 8 个样本）
        global_mean = all_rewards_tensor.mean()
        chz_mean = all_chz_tensor.mean()
        global_std = all_rewards_tensor.std()
        retInfo["global avg correctness"] = chz_mean* 1.
        retInfo["global avg reward"] = global_mean* 1.
        retInfo["global reward std"] = global_std**2 *1.
        
        logger.info("This is all rewards:")
        if accelerator.is_main_process:
            accelerator.print(all_rewards_tensor)
            
        logger.info("This is all chiz:")
        if accelerator.is_main_process:
            accelerator.print(all_chz_tensor)
            
        if not inferenceState:
            with torch.no_grad():
                with model.disable_adapter():
                    ref_generated_output = model(
                        input_ids=generated_sequences,
                        attention_mask=attention_mask,  # 关键：传入mask
                        return_dict=True,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    #ref_logits = ref_output.logits[:, input_length:]
            #print(torch.cuda.memory_summary())

            ref_generated_logits = ref_generated_output["logits"]
            ref_logp_list = []
            for i, output_ids in enumerate(generated_sequences[:, input_length:]):
                ref_token_logits = ref_generated_logits[i]
                ref_probs = torch.softmax(ref_token_logits, dim=-1)
                ref_token_probs = ref_probs[range(len(output_ids)), output_ids]
                ref_logp = torch.log(ref_token_probs)
                ref_logp_list.append(ref_logp)
            
            ref_logp_tensor = torch.stack(ref_logp_list, dim=0)
            ref_total_logp = ref_logp_tensor.mean(dim=1)
            # 计算策略比率 (ratios)
            ratios = torch.exp(total_logp - ref_total_logp)  # r_t = exp(logp - ref_logp)

            logger.info("This is now ratios:")
            if accelerator.is_main_process:
                accelerator.print(ratios)

            # 将rewards 标准化
            if config["reward"]["reward_method"] == "std":
                rewards_tensor = (rewards_tensor - global_mean) / (global_std + 1e-8)  # 避免除以0
            elif config["reward"]["reward_method"] == "unif":
                rewards_tensor = (rewards_tensor -1.05) / 1.5
            
            clip_epsilon = config["rl_conf"]["clip_epsilon"]
            entropy_coef = config["rl_conf"]["entropy_coef"]
            kl_loss_coef = config["rl_conf"]["kl_loss_coef"]
            # 裁剪策略比率
            clipped_ratios = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
            loss = - torch.min(ratios * rewards_tensor, clipped_ratios * rewards_tensor).mean()
            retInfo["pure loss"] = loss.item()
            if config["rl_conf"]["entropy_loss"]:
                probs = torch.softmax(generated_logits, dim=-1)  # 当前策略的概率分布
                entropy_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()  # 熵的负值
                loss -= entropy_coef * entropy_loss
                retInfo["entropy loss"] = -entropy_loss.item()
            if config["rl_conf"]["kl_3_estimator"]:
                kl_loss = (ratios - (total_logp - ref_total_logp)) - 1
                kl_loss = kl_loss.mean()
                #breakpoint()
                loss +=  kl_loss_coef * kl_loss
                retInfo["kl loss"] = kl_loss.item()
        else:
            loss = None
        
        return loss, retInfo

        # with tempfile.TemporaryDirectory() as tmpdir:
            
            #     for i, pil in enumerate(images):
            #         pil = pil.resize((256, 256))
            #         pil.save(os.path.join(tmpdir, f"{i}.jpg"))

            #     wandb.log(
            #         {
            #             "images": [
            #                 wandb.Image(
            #                     os.path.join(tmpdir, f"{i}.jpg"),
            #                     caption=f"{prompt:.100} | {((-1)**i)*reward:.2f}",
            #                 )
            #                 for i in range(2)  # 仅记录两个图像
            #             ],
            #         },
            #         commit=False,
            #     )
            # wandb.log(
            #     {
            #         "correct": int(flag) + (reward == 0) / 2,
            #         "Total inference Lenth": lth,
            #     },
            #     commit=False,
            # )
        
        
        
        # for principle, comparison in zip(principles, comparison_levels):
        #     try:
        #         Eval_Dict[principle.strip("* ")] = COMP_MAPPING[comparison.strip("* ")]
        #     except Exception:
        #         NumberOfWrongCompRES += 1
        #         WrongCompRES.append(comparison.strip("* "))

        # final_score = 0

        # for eva, score in Eval_Dict.items():
        #     final_score += score
        
        # isNULLANS = False
        # if len(Eval_Dict)==0:
        #     isNULLANS = True

        # if not isNULLANS:
        #     posANS, negANS = sum(k > 0 for k in Eval_Dict.values()), sum(k < 0 for k in Eval_Dict.values())
        #     posRATE, negRATE = posANS/len(Eval_Dict), negANS/len(Eval_Dict)
        # else:
        #     posANS, negANS, posRATE, negRATE = 0,0,0,0
        
        # if not principles:
        #     Error.write("\nRESPOND\n"+ EVAL)
        #     Error.flush()
        # elif not Eval_Dict:
        #     Error.write("\nRESPOND\n"+ EVAL)
        #     Error.flush()

        # ERROR_DICT = {
        #     "NumberOfWrongCompRES":NumberOfWrongCompRES,
        #     "WrongCompRES": WrongCompRES,
        #     "FIRST RESPOND TIME DELAY":cnt_1,
        #     "SECOND RESPOND TIME DELAY":cnt_2,
        #     "Is Null Answer":isNULLANS
        # }

        # ANS_DICT = {
        #     "Positive Rate":posRATE,
        #     "Negative Rate":negRATE,
        #     "Null Rate": 1 - posRATE - negRATE,
        #     "TOTAL": len(Eval_Dict),
        #     "Lenth":lth,
        # }

        # if DEBUG:
        #     print("[Score]")
        #     print(final_score)
        #     print("[DICT]")
        #     print(Eval_Dict)

        # if RECORD:
        #     ANALYSIS.write(ans2)
        #     ANALYSIS.flush()

    def _fn_grpo(inputs, toolbox=None,accelerator=None,config=None):
            
        if toolbox is None:
            raise ValueError("pipe in Qwen")
        model, processor,logger = toolbox
        inferenceState = False
        invs = inputs.pop("invs")
        invs = [x for x in invs for _ in range(config.grpo_1gpu_size)]
        inputs = inputs.to("cuda")
        pad_token_id = processor.tokenizer.pad_token_id
        generate_kwargs = {
            "do_sample": True,           # 启用采样模式
            "temperature": 1.5,          # 增加随机性
            "top_k": 200,                 # 考虑前 50 个概率最高的词
            # "top_p": 0.9,                # 核采样，动态控制候选词
            "num_return_sequences": config.grpo_1gpu_size    # 仅返回一个生成序列
        }
    
        generated_sequences = model.generate(
            **inputs,
            max_new_tokens=2048,               # 控制生成的最大长度
            pad_token_id=pad_token_id,         # 确保填充符正确
            return_dict_in_generate=False,     # 不需要返回完整生成信息
            use_cache=True,                    # 提高效率
            **generate_kwargs                  # 动态更新生成参数
        )
        attention_mask = (generated_sequences != pad_token_id).long()
        input_length = inputs.input_ids.shape[1]

        if not inferenceState:
            generated_output = model(
                input_ids=generated_sequences,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                use_cache = config["transformer"]["use_cache"],
            )
            

            logits = generated_output["logits"]
            generated_logits = logits  # (batch_size, generated_length, vocab_size)

            logp_list = []
            for i, output_ids in enumerate(generated_sequences[:, input_length:]):
                token_logits = generated_logits[i]
                probs = torch.softmax(token_logits, dim=-1)
                token_probs = probs[range(len(output_ids)), output_ids]
                logp = torch.log(token_probs)
                logp_list.append(logp)

            logp_tensor = torch.stack(logp_list, dim=0)
            total_logp = logp_tensor.mean(dim=1)
            # breakpoint() 

        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids 
            in zip(torch.vstack([inputs.input_ids]*config.grpo_1gpu_size), generated_sequences)
        ]
        #        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids  in zip(torch.vstack(inputs.input_ids, config.grpo_1gpu_size), generated_sequences)]
        
        ans = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        #breakpoint()
        if config.get("log_cot", None) and config.get("curr_batch_dir", None):
            for idx, sample in enumerate(ans):
                file_path = os.path.join(config["curr_batch_dir"], f"{idx}_cot.txt")
                with open(file_path, "w") as file:  # 使用上下文管理器
                    file.write(sample)

        retInfo = []
        rewards = []
        fmts = []
        chz = []
        lth = []
        rzlths = []
        ckptcnts = []
        valid = []
        for txt,inv in zip(ans,invs):
            resDict = parse_output(txt)
            fmt = resDict["is_valid_format"]
            chiz = resDict["final_verdict"]
            rzlths.append(resDict["analysis_length"])

            if not fmt:
                curr_reward = -0.5
            elif chiz is None:
                curr_reward = -0.5
            else:
                curr_reward =  int(fmt) + int( (1-2*inv) == chiz)*3 + int( chiz == 0 ) - 1.5
                if config["reward"]["reward_long_cot"]:
                    curr_reward += resDict["total_length"]/3000. - 0.95
            
            rewards.append(curr_reward)
            fmts.append(int(fmt))
            chz.append(int( (1-2*inv) == chiz)+0.5*(chiz==0))
            lth.append(len(txt))
            valid.append(int(chiz is None))
            

        retInfo = {
            "fomat correctness": sum(fmts) / len(fmts)*1. if fmts else 0.,  # 防止列表为空
            "choose correctness": sum(chz) / len(chz)*1. if chz else 0.,
            "avg lth": sum(lth) / len(lth)*1. if lth else 0.,
            "avg reward": sum(rewards) / len(rewards)*1. if rewards else 0.,
            "avg reasoning": sum(rzlths) / len(rzlths)*1. if rzlths else 0.,
            "valid rate": sum(valid) / len(valid)*1. if valid else 0.,
            #"avg check point count": sum(ckptcnts) / len(ckptcnts) if ckptcnts else 0,

        }
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(accelerator.device)
        chz_tensor = torch.tensor(chz, dtype=torch.float32).to(accelerator.device)
        # 使用 accelerator.gather() 收集所有卡上的 rewards_tensor
        all_rewards_tensor = accelerator.gather(rewards_tensor)
        all_chz_tensor = accelerator.gather(chz_tensor)
        # 在主进程上计算全局均值和标准差（跨 8 个样本）
        global_mean = all_rewards_tensor.mean()
        chz_mean = all_chz_tensor.mean()
        global_std = all_rewards_tensor.std() if len(all_rewards_tensor) > 1 else 0.
        retInfo["global avg correctness"] = chz_mean* 1.
        retInfo["global avg reward"] = global_mean* 1.
        retInfo["global reward std"] = global_std**2 *1.
        
        logger.info("This is all rewards:")
        if accelerator.is_main_process:
            accelerator.print(all_rewards_tensor)
            
        logger.info("This is all chiz:")
        if accelerator.is_main_process:
            accelerator.print(all_chz_tensor)
            
        if not inferenceState:
            with torch.no_grad():
                with model.disable_adapter():
                    ref_generated_output = model(
                        input_ids=generated_sequences,
                        attention_mask=attention_mask,  # 关键：传入mask
                        return_dict=True,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    #ref_logits = ref_output.logits[:, input_length:]
            #print(torch.cuda.memory_summary())

            ref_generated_logits = ref_generated_output["logits"]
            ref_logp_list = []
            for i, output_ids in enumerate(generated_sequences[:, input_length:]):
                ref_token_logits = ref_generated_logits[i]
                ref_probs = torch.softmax(ref_token_logits, dim=-1)
                ref_token_probs = ref_probs[range(len(output_ids)), output_ids]
                ref_logp = torch.log(ref_token_probs)
                ref_logp_list.append(ref_logp)
            
            ref_logp_tensor = torch.stack(ref_logp_list, dim=0)
            ref_total_logp = ref_logp_tensor.mean(dim=1)
            # 计算策略比率 (ratios)
            ratios = torch.exp(total_logp - ref_total_logp)  # r_t = exp(logp - ref_logp)

            logger.info("This is now ratios:")
            if accelerator.is_main_process:
                accelerator.print(ratios)

            # 将rewards 标准化
            if config["reward"]["reward_method"] == "std":
                rewards_tensor = (rewards_tensor - global_mean) / (global_std + 1e-8)  # 避免除以0
            elif config["reward"]["reward_method"] == "unif":
                rewards_tensor = (rewards_tensor -1.05) / 1.5
            
            clip_epsilon = config["rl_conf"]["clip_epsilon"]
            entropy_coef = config["rl_conf"]["entropy_coef"]
            kl_loss_coef = config["rl_conf"]["kl_loss_coef"]
            # 裁剪策略比率
            clipped_ratios = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
            loss = - torch.min(ratios * rewards_tensor, clipped_ratios * rewards_tensor).mean()
            retInfo["pure loss"] = loss.item()
            if config["rl_conf"]["entropy_loss"]:
                probs = torch.softmax(generated_logits, dim=-1)  # 当前策略的概率分布
                entropy_loss = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()  # 熵的负值
                loss -= entropy_coef * entropy_loss
                retInfo["entropy loss"] = -entropy_loss.item()
            if config["rl_conf"]["kl_3_estimator"]:
                kl_loss = (ratios - (total_logp - ref_total_logp)) - 1
                kl_loss = kl_loss.mean()
                #breakpoint()
                loss +=  kl_loss_coef * kl_loss
                retInfo["kl loss"] = kl_loss.item()
        else:
            loss = None
        
        return loss, retInfo
    
    return _fn_general if select=="general" else _fn_grpo

if __name__ == "__main__":
    pass

