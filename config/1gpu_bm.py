import ml_collections
from config.base_bm import get_config as base_config
from datetime import datetime
date_tag = datetime.now().strftime("%m%d")

def get_config():
    config = base_config()
    config.Project_name += "_1gpu"+ date_tag
    config.run_name = "1gpu_benchmark"
    config.inference = True
    config.dataset_url = "/m2v_intern/wangqunzhong/research/huggingface/dataset/ymhao/HPD_v2"
    config.deepspeed_stage = 0
    config.curr_batch_dir = ""

    ##### Resume
    config.resume_ckpt_id = ""
    config.resume_ckpt = ""
    config.resume_gpus = 6
    config.resume_zeRO = 2

    config.data_conf.chunk_size = 100
    ##### Reward #####
    config.reward.reward_method = "unif"
    config.reward.reward_pw_fn = "evaluate_QwenVL2_7B"

    ###### Training ######
    config.train.use_8bit_adam = True

    return config
