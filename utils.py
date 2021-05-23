import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, top_k_top_p_filtering
from rouge_score import rouge_scorer
from tqdm import tqdm
from settings import *
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def add_special_tokens():
    """ Returns GPT2 tokenizer after adding separator and padding tokens """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def get_device_map(n_gpus):
    n_gpt_layers = 12
    per_gpu_layers = n_gpt_layers//n_gpus
    additional_layers = n_gpt_layers%n_gpus

    device_map = dict()
    total_layers = 0
    for i in range(n_gpus):
        if additional_layers <= 0:
            n_layers = per_gpu_layers
        else:
            n_layers = per_gpu_layers + 1
            additional_layers -= 1
        device_map[i] = [k + total_layers for k in range(n_layers)]
        total_layers += n_layers
    return device_map


def pad_seqs(gen_logits, input_seq, mask, seq_inds, n_pad, pad_token):
    batch_size = input_seq.shape[0]

    pad_tensor = torch.zeros((batch_size, n_pad))
    mask = torch.cat([mask, pad_tensor], dim=-1)

    pad_tensor[:, :] = pad_token
    input_seq = torch.cat([input_seq, pad_token], dim=-1)

    pad_tensor = torch.repeat_interleave(seq_inds[:, -1:], n_pad, dim=-1)
    seq_inds = torch.cat([seq_inds, pad_tensor])

    for _ in range(n_pad):
        gen_logits.append(torch.zeros((batch_size, 1)))

    return gen_logits, input_seq, mask, seq_inds


def generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False,
                      eos_token=-1, pad_token=-1, top_k = 0, top_p = 1.0):
    if greedy:
        top_k = 1

    input_seq = batch['article'].cuda()
    mask = batch['article_mask'].cuda()
    seq_inds = batch['article_position_ids'].cuda()
    batch_size = input_seq.shape[0]

    gen_logits = list()
    generation_finished = torch.zeros((batch_size, 1)).cuda()
    ones = torch.ones_like(generation_finished).cuda()
    for i in range(max_gen_len):
        outputs = model(input_ids=input_seq, attention_mask=mask, position_ids=seq_inds)
        next_token_logits = outputs[0][:, -1, :]
        gen_logits.append(next_token_logits)
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generation_finished = torch.where(next_token == eos_token, ones, generation_finished)
        input_seq = torch.cat([input_seq, next_token], dim=-1)
        mask = torch.cat([mask, (1-generation_finished).long()], dim=-1)
        new_inds = (seq_inds[:, -1:] + (1 - generation_finished)).long()
        seq_inds = torch.cat([seq_inds, new_inds], dim=-1)

        if torch.all(generation_finished == 1):
            gen_logits, input_seq, mask, seq_inds = pad_seqs(gen_logits, input_seq, mask, seq_inds, max_gen_len-i-1, pad_token)
            break

    return torch.cat(gen_logits, dim=-1), input_seq[:, -MAX_GEN_LEN:],\
           mask[:, -MAX_GEN_LEN:], seq_inds[:, -MAX_GEN_LEN:]


def get_r_one_rewards(gt_seqs, sample_seqs, tokenizer):
    rewards = []
    for gt, pred in zip(gt_seqs, sample_seqs):
        gt_text = tokenizer.decode(gt.tolist(), skip_special_tokens=True)
        pred_text = tokenizer.decode(pred.tolist(), skip_special_tokens=True)
        r_scores = rouge_scorer.score(gt_text, pred_text)
        r_one = r_scores['rouge1'][2]
        rewards.append(r_one)
    return torch.Tensor(rewards), r_scores


def logprobs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def loss_fct(delta_reward, sample_logits, sample_seqs, sample_mask):
    token_logprobs = logprobs_from_logits(sample_logits, sample_seqs)
    loss = delta_reward.unsquize(1) * token_logprobs * sample_mask
    loss = torch.sum(loss, dim=-1) / torch.sum(sample_mask, dim=-1)
    loss = -torch.mean(loss)
    return loss


def validate(model, val_data_loader, tokenizer, logger, total_steps_passed):
    for i, batch in enumerate(tqdm(val_data_loader)):
        with torch.no_grads():

            _, greedy_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=True)
            greedy_rewards, greedy_rouge_scores = get_r_one_rewards(batch['abstract'], greedy_seqs.detach(), tokenizer)

            _, sample_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False)
            sample_rewards, sample_rouge_scores = get_r_one_rewards(batch['abstract'], sample_seqs.detach(), tokenizer)

            delta_reward = sample_rewards - greedy_rewards
            logger.log(delta_reward, greedy_rewards, sample_rewards, val=True)

    logger.write_rewards(val=True)
    logger.save_example(batch['article'][0], greedy_seqs[0], batch['abstract'][0],
                        tokenizer, total_steps_passed)
