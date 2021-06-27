N_GPUS = 1   # количество gpu для обучения сети
PATH_TO_EXPERIMENT = "./experiments/experiment_name"  # папка, в которую будут сохраняться веса и метрики
PATH_TO_PRETRAIN_MODEL = "./pretrain_models/checkpoint"  # папка с весами предобученной модели
N_EPOCHS = 1  # кол-во эпох
BATCH_SIZE = 2  # размер батча
ACCUMULATION_STEPS = 6  # количество батчей между шагами оптимизатора
DATA_FOLDER = "../gpt2_summ_rl/CNN-DM/gpt2_1024_data"  # папка с данными в формате json
PATH_TO_IDS_FILE = "../gpt2_summ_rl/CNN-DM/ids.json"  # путь к json файлу с разбиением датасета на трейн, валидацию и тест
MAX_GEN_LEN = 100  # максимальная длина генерации резюме
STEPS_BTW_VALIDATIONS = 500  # количество шагов оптимизатора между валидацией и сохранением модели
LR = 1e-6  # скорость обучения
