import ml_collections
from config.base_rl import get_config as base_config
from datetime import datetime
date_tag = datetime.now().strftime("%m%d")

def get_config():
    config = base_config()
    config.Project_name += "_1gpu_" + date_tag
    config.run_name = "1gpu_rl"
    config.inference = False
    config.dataset_url = "/m2v_intern/wangqunzhong/research/huggingface/dataset/ymhao/HPD_v2"
    config.log_freq = 2
    
    ##### Data #####
    config.data_conf.chunk_size = 300
    config.data_conf.verify_chunk_size = 50
    config.train.batch_size = 1

    ##### Reward #####
    config.reward.reward_method = "unif"
    config.reward.reward_pw_fn = "evaluate_QwenVL2_7B"
    config.reward.reward_long_cot = True

    ###### Training ######
    config.train.use_8bit_adam = True

    return config
