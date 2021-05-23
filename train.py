import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel
from tqdm import tqdm
from dataset import Data
from utils import loss_fct, add_special_tokens, get_device_map, generate_abstract, get_r_one_rewards, validate
from settings import *
from logger import Logger


logger = Logger(PATH_TO_EXPERIMENT)
train_data = Data(mode='train')
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
valid_data = Data(mode='valid', length=500)
val_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
tokenizer = add_special_tokens()

model = GPT2LMHeadModel.from_pretrained(PATH_TO_PRETRAIN_MODEL)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
if N_GPUS > 1:
    device_map = get_device_map(N_GPUS)
    model.parallelize(device_map)


accumulated_batches = 0
total_steps_passed = 0

for epoch in range(N_EPOCHS):
    for i, batch in enumerate(tqdm(train_data_loader)):

        sample_logits, sample_seqs, mask, seq_inds = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False)
        sample_rewards, sample_rouge_scores = get_r_one_rewards(batch['abstract'], sample_seqs.detach(), tokenizer)

        with torch.no_grad():
            _, greedy_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=True)
        greedy_rewards, greedy_rouge_scores = get_r_one_rewards(batch['abstract'], greedy_seqs.detach(), tokenizer)

        delta_reward = sample_rewards - greedy_rewards
        loss = loss_fct(delta_reward, sample_logits, sample_seqs, mask)
        loss.backward()
        accumulated_batches += 1
        logger.log(delta_reward, greedy_rewards, sample_rewards, val=False)

        if accumulated_batches % ACCUMULATION_STEPS == 0:
            optimizer.step()
            model.zero_grad()
            accumulated_batches = 0
            total_steps_passed += 1
            logger.write_rewards(val=False)

        if total_steps_passed % STEPS_BTW_VALIDATIONS == 0:
            results = validate(model, val_data_loader, tokenizer, logger, total_steps_passed)
            logger.save_model(model, epoch, total_steps_passed)
