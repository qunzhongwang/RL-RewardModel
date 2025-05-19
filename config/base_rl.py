import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.inference_basedir = "log_inf_cont"
    ### New Feature ###
    #基本构造

    config.num_train_epochs = 2

    config.seed = 42

    config.logdir = "ckpt_log"
    config.loradir = "lora_log"
    config.save_freq = 20
    config.num_checkpoint_limit = 5
    config.log_freq = 5


    config.num_epochs = 5
    config.inference = False
    config.resume_ckpt = "0"

    config.mixed_precision = "bf16"#"bf16"
    config.deepspeed_stage = 0

    config.use_lora = True
    config.Project_name = "Qwen_RL_train"
    config.run_name = "base"
    config.debug_ver = False

    ###### Data #####
    config.data_conf = data_conf = ml_collections.ConfigDict()
    data_conf.dataset_num_proc = 64
    data_conf.dataset_url = "/m2v_intern/wangqunzhong/research/huggingface/dataset/ymhao/HPD_v2"
    data_conf.chunk_size = 20
    data_conf.verify_chunk_size = 20
    

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    pretrained.model = "Qwen/Qwen2-VL-7B-Instruct"
    pretrained.revision = "main"
    pretrained.attn_implementation = "flash_attention_2"

    ###### Reward ######
    config.reward = reward = ml_collections.ConfigDict()
    reward.reward_method = "unif"
    reward.reward_pw_fn = "evaluate_QwenVL2_7B"
    reward.reward_long_cot = False

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.batch_size = 1
    train.use_8bit_adam = False
    train.learning_rate = 3e-4
    train.gradient_accumulation_steps = 4
    train.max_grad_norm = 4.0
    train.lora_r = 32
    train.lora_alpha = 64
    train.lora_dropout = 0.1
    train.lora_target_modules = ["q_proj", "k_proj", "v_proj"]

    ##### Val #####
    config.val = val = ml_collections.ConfigDict()
    val.val_batch_size = 1


    ##### RL #####
    config.rl_conf = rl_conf = ml_collections.ConfigDict()
    rl_conf.adv_clip_max = 5
    rl_conf.clip_epsilon = 5e-2
    rl_conf.entropy_coef = 1e-2
    rl_conf.kl_loss_coef = 1e-4
    rl_conf.entropy_loss = False
    rl_conf.kl_3_estimator = False

    ##### transformer ######
    config.transformer = transformer = ml_collections.ConfigDict()
    transformer.use_cache = False
    
    


    return config
