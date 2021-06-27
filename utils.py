import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, top_k_top_p_filtering
from tqdm import tqdm
from settings import *


# создаем rouge scorer
from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


# создаем bleurt scorer
import tensorflow as tf
from bleurt import score
with tf.device('cpu'):
    scorer = score.BleurtScorer('../bleurt/bleurt/bleurt-base-512/')


def add_special_tokens():
    """ Returns GPT2 tokenizer after adding separator and padding tokens """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {'pad_token': '<|pad|>', 'sep_token': '<|sep|>'}
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def get_device_map(n_gpus):
    """
    Вспомогательная функция, которая распределяет слои gpt2 small по gpu для вертикального распараллеливания
    Params:
        n_gpus: int: количество используемых gpu.
    Returns:
        device_map: dict: словарь с распределением слоев по gpu (https://huggingface.co/transformers/model_doc/gpt2.html#transformers.GPT2LMHeadModel.parallelize)
    """
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


def pad_seqs(input_seq, mask, seq_inds, n_pad, pad_token):
    """
    Вспомогательная функция. Дополняет тензоры до нужной длины справа.
    Тензор с токенами дополняется pad токенами
    Тензор с маской дополняется нулями
    Тензор с индексами дополняется правыми значениями в тензоре. Индексы - порядковые номера токенов в последовательности.
    Params:
        input_seq: torch.Tensor: тензор с токенами
        mask: torch.Tensor: тензор с маской
        seq_inds: torch.Tensor: тензор с индексами
        n_pad: int: на сколько нужно дополнить тензоры
        pad_token: int:
    Return:
        input_seq: дополненный тензор токенов
        mask: дополненный тензор с маской
        seq_inds: дополненный тензор индексов
    """
    batch_size = input_seq.shape[0]

    # дополнение маски
    pad_tensor = torch.zeros((batch_size, n_pad)).cuda()
    mask = torch.cat([mask, pad_tensor], dim=-1)

    # дополнение тензора с токенами
    pad_tensor[:, :] = pad_token
    input_seq = torch.cat([input_seq, pad_tensor], dim=-1)

    # дополнение тензора с индексами
    pad_tensor = torch.repeat_interleave(seq_inds[:, -1:], n_pad, dim=-1)
    seq_inds = torch.cat([seq_inds, pad_tensor], dim=-1)

    return input_seq, mask, seq_inds


def generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False,
                      eos_token=-1, pad_token=-1, top_k=0, top_p=1.0):
    """
    Функция генерирует резюме для батча данных. Работает как в жадном режиме, так и в режиме сэмплирования токенов.
    Params:
        model: transformers.GPT2LMHeadModel: модель резюмирования
        batch: dict: словарь с входными данными. 'article'
                                                 'article_mask'
                                                 'article_position_ids'
        max_gen_len: int: максимальная длина генерируемых резюме
        greedy: bool: если True - при генерации выбирается токен с наибольшей вероятностью.
                      Если False - производится сэмплирование из распределения
        eos_token: int: токен конца генерации резюме.
        pad_token: int: pad токен
        top_k: int: фильтрация топ top_k наиболее вероятных токенов
        top_p: float: фильтрация топ top_p токенов по суммарной вероятности
    Return:
        input_seq: torch.Tensor: тензор токенов текста и резюме, разделенных токеном разделения и, возможно, pad токенами
        mask: torch.Tensor: тензор соответствующих масок pad токенов
        seq_inds: torch.Tensor: тензор индексов токенов. pad токены не увеличивают индекс.
    """
    if greedy:
        top_k = 1

    input_seq = batch['article']
    mask = batch['article_mask']
    seq_inds = batch['article_position_ids']
    batch_size = input_seq.shape[0]

    generation_finished = torch.zeros((batch_size, 1)).cuda()
    ones = torch.ones_like(generation_finished).cuda()
    for i in range(max_gen_len):
        with torch.no_grad():
            outputs = model(input_ids=input_seq, attention_mask=mask, position_ids=seq_inds)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generation_finished = torch.where(next_token == eos_token, ones, generation_finished)
        input_seq = torch.cat([input_seq, next_token], dim=-1)
        mask = torch.cat([mask, (1-generation_finished).long()], dim=-1)
        new_inds = (seq_inds[:, -1:] + (1 - generation_finished)).long()
        seq_inds = torch.cat([seq_inds, new_inds], dim=-1)

        if torch.all(generation_finished == 1):
            input_seq, mask, seq_inds = pad_seqs(input_seq, mask, seq_inds, max_gen_len-i-1, pad_token)
            break

    return input_seq, mask.long(), seq_inds.long()


def get_r_one_rewards(gt_seqs, sample_seqs, tokenizer):
    """
    Функция вычисления наград и метрик.
    Params:
        gt_seqs: torch.Tensor: тензор с токенами гт резюме
        sample_seqs: torch.Tensor: тензор с токенами сгенерированного резюме
        tokenizer: GPT2Tokenizer: токенезатор
    Return:
        rewards: torch.Tensor: тензор с целевой метрикой
        r_scores: list: список словарей, содержащих все вычисляемые метрики качества.
    """
    rewards = []
    r_scores = []
    for gt, pred in zip(gt_seqs, sample_seqs):
        gt_text = tokenizer.decode(gt.tolist(), skip_special_tokens=True)
        pred_text = tokenizer.decode(pred.tolist(), skip_special_tokens=True)
        bleurt_reward = scorer.score([gt_text], [pred_text])[0]
        r_scores.append(rouge_scorer.score(gt_text, pred_text))
        r_scores[-1]['bleurt'] = bleurt_reward
        #r_one = r_scores[-1]['rouge1'][2]
        rewards.append(bleurt_reward)
    return torch.Tensor(rewards), r_scores


def logprobs_from_logits(logits, labels):
    """
    По логитам и сгенерированным токенам находит логарифмы вероятности сгенерированных токенов
    Params:
        logits: torch.Tensor: логиты генерации токенов
        labels: torch.Tensor: сгенерированные токены
    Return:
        logpy: torch.Tensor: логарифм вероятности сгенерированного токена
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy


def loss_fct(delta_reward, sample_logits, sample_seqs, sample_mask):
    """
    Функция осуществляет вычисление ошибки сети для использования алгоритма reinforce.
    Params:
        delta_reward: torch.Tensor: награды за эпизоды (или разности наград для self-critic алгоритма)
        sample_logits: torch.Tensor: логиты генерации токенов
        sample_seqs: torch.Tensor: токены сгенерированных резюме
        sample_mask: torch.Tensor: маска pad тензоров
    Return:
        loss: torch.Tensor: ошибка сети
    """
    token_logprobs = logprobs_from_logits(sample_logits, sample_seqs)
    loss = token_logprobs * sample_mask
    loss = delta_reward.unsqueeze(1) * torch.sum(loss, dim=-1) / (torch.sum(sample_mask, dim=-1) + 1e-6)
    loss = -torch.mean(loss)
    return loss


def validate(model, val_data_loader, tokenizer, logger, total_steps_passed):
    """
    Функция выполняет валидацию и записывает результаты и примеры генерируемых резюме в txt файлы
    Params:
        model: модель генерации
        val_data_loader: даталоадер с валидационным датасетом
        tokenizer: токенизатор
        logger: логгер
        total_steps_passed: токущая итерация / количество пройденных шагов оптимизации
    """
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

            greedy_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False,
                                                  eos_token=tokenizer.bos_token_id,
                                                  pad_token=tokenizer.pad_token_id)
            greedy_rewards, greedy_rouge_scores = get_r_one_rewards(batch['abstract'],
                                                                    greedy_seqs[:, -MAX_GEN_LEN:].detach(), tokenizer)

            sample_seqs, _, _ = generate_abstract(model, batch, max_gen_len=MAX_GEN_LEN, greedy=False,
                                                  eos_token=tokenizer.bos_token_id,
                                                  pad_token=tokenizer.pad_token_id)
            sample_rewards, sample_rouge_scores = get_r_one_rewards(batch['abstract'],
                                                                    sample_seqs[:, -MAX_GEN_LEN:].detach(), tokenizer)

            delta_reward = sample_rewards - greedy_rewards

            logger.log(delta_reward, greedy_rouge_scores, sample_rouge_scores, val=True)

    logger.write_rewards(val=True)
    logger.save_example(batch['article'][0], greedy_seqs[0, -MAX_GEN_LEN:], batch['abstract'][0],
                        tokenizer, total_steps_passed)
