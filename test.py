import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from tqdm import tqdm
from dataset import Data
from utils import loss_fct, add_special_tokens, get_device_map, generate_abstract, get_r_one_rewards, validate
from settings import *
from logger import Logger


logger = Logger(PATH_TO_EXPERIMENT)
valid_data = Data(mode='valid')
val_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
tokenizer = add_special_tokens()

model = GPT2LMHeadModel.from_pretrained(PATH_TO_PRETRAIN_MODEL)
if N_GPUS > 1:
    device_map = get_device_map(N_GPUS)
    model.parallelize(device_map)
else:
    model.cuda()
model.eval()

total_seqs = 0
total_r1, total_r2, total_rl = 0, 0, 0

for i, batch in enumerate(tqdm(val_data_loader)):
    with torch.no_grad():
        input_seq = batch['article'].cuda()
        mask = batch['article_mask'].cuda()
        seq_inds = batch['article_position_ids'].cuda()
        abstract = batch['abstract']
        batch = {'article': input_seq,
                 'article_mask': mask,
                 'article_position_ids': seq_inds,
                 'abstract': abstract}
        n_seqs = abstract.shape[0]

        greedy_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=True,
                                              eos_token=tokenizer.bos_token_id,
                                              pad_token=tokenizer.pad_token_id)

        greedy_rewards, greedy_rouge_scores = get_r_one_rewards(batch['abstract'],
                                                                greedy_seqs[:, -MAX_GEN_LEN:].detach(), tokenizer)

    total_seqs += n_seqs
    for scores in greedy_rouge_scores:
        total_r1 += scores['rouge1'][2]
        total_r2 += scores['rouge2'][2]
        total_rl += scores['rougeL'][2]

    with open(PATH_TO_EXPERIMENT+"/test_results.txt", 'w') as f:
        f.write("%d sequenses tested\nrouge1: %.6f\nrouge2: %.6f\nrougeL: %f\n" % (total_seqs, total_r1/total_seqs, total_r2/total_seqs, total_rl/total_seqs))
