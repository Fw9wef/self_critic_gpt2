N_GPUS = 2
#PATH_TO_PRETRAIN_MODEL = "../gpt2_summ_rl/output/long_train/4"
#PATH_TO_EXPERIMENT = "../gpt2_summ_rl/output/long_train/4"
PATH_TO_PRETRAIN_MODEL = "./experiments/delta_work/weights/chk_epoch_0_iteration_2000"
PATH_TO_EXPERIMENT = "./experiments/delta_work/weights/chk_epoch_0_iteration_2000"
N_EPOCHS = 2
BATCH_SIZE = 12
ACCUMULATION_STEPS = 4
DATA_FOLDER = "../gpt2_summ_rl/CNN-DM/gpt2_1024_data"
PATH_TO_IDS_FILE = "../gpt2_summ_rl/CNN-DM/ids.json"
MAX_GEN_LEN = 100
STEPS_BTW_VALIDATIONS = 100
LR = 1e-6
