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
valid_data = Data(mode='valid', length=12)
val_data_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
tokenizer = add_special_tokens()

model = GPT2LMHeadModel.from_pretrained(PATH_TO_PRETRAIN_MODEL)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
if N_GPUS > 1:
    device_map = get_device_map(N_GPUS)
    model.parallelize(device_map)
else:
    model.cuda()

accumulated_batches = 0
total_steps_passed = 0

for epoch in range(N_EPOCHS):
    for i, batch in enumerate(tqdm(train_data_loader)):
        input_seq = batch['article'].cuda()
        mask = batch['article_mask'].cuda()
        seq_inds = batch['article_position_ids'].cuda()
        abstract = batch['abstract']
        batch = {'article': input_seq,
                 'article_mask': mask,
                 'article_position_ids': seq_inds,
                 'abstract': abstract}

        with torch.no_grad():
            sample_seqs, mask, seq_inds = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False,
                                                            eos_token=tokenizer.bos_token_id,
                                                            pad_token=tokenizer.pad_token_id)
            sample_rewards, sample_rouge_scores = get_r_one_rewards(batch['abstract'],
                                                                    sample_seqs[:, -MAX_GEN_LEN:].detach(), tokenizer)
            greedy_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=True,
                                                  eos_token=tokenizer.bos_token_id,
                                                  pad_token=tokenizer.pad_token_id)
            greedy_rewards, greedy_rouge_scores = get_r_one_rewards(batch['abstract'],
                                                                    greedy_seqs[:, -MAX_GEN_LEN:].detach(), tokenizer)

        sample_logits = model(input_ids=sample_seqs.long(), attention_mask=mask.long(), position_ids=seq_inds.long())
        sample_logits = sample_logits[0][:, -MAX_GEN_LEN:]
        sample_seqs = sample_seqs[:, -MAX_GEN_LEN:]
        mask = mask[:, -MAX_GEN_LEN:]

        delta_reward = sample_rewards.cuda() - greedy_rewards.cuda()

        loss = loss_fct(delta_reward, sample_logits, sample_seqs.long(), mask)
        loss.backward()
        accumulated_batches += 1
        logger.log(delta_reward, greedy_rouge_scores, sample_rouge_scores, val=False)

        if accumulated_batches % ACCUMULATION_STEPS == 0:
            optimizer.step()
            model.zero_grad()
            accumulated_batches = 0
            total_steps_passed += 1
            logger.write_rewards(val=False)
            logger.save_example(batch['article'][0], greedy_seqs[0, -MAX_GEN_LEN:], batch['abstract'][0],
                                tokenizer, total_steps_passed)

        if total_steps_passed % STEPS_BTW_VALIDATIONS == 0:
            #results = validate(model, val_data_loader, tokenizer, logger, total_steps_passed)
            logger.save_model(model, epoch, total_steps_passed)
