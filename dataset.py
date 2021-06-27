import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import add_special_tokens
from settings import *


class Data(Dataset):
    '''
    Это класс датасета. Формат требуемых данных аналогичен тому, что представлен в репозитории с предобучением.
    Но этот класс возвращает образцы немного в другом виде.
    '''
    def __init__(self, mode='train', length=None):
        """
        Params:
            mode: str: 'train', 'valid' или 'test'. В зависимости от значения использует трейновые, валидационные или тестовые данные
            length: int или None: позволяет ограничить размер датасета.
        """
        self.root_dir = DATA_FOLDER
        self.tokenizer = add_special_tokens()
        self.pad = self.tokenizer.encode(self.tokenizer.pad_token)
        self.files = np.sort([x for x in os.listdir(self.root_dir) if x.endswith('.json')])
        self.mode = mode
        with open(PATH_TO_IDS_FILE , 'r') as f:
            self.data = json.load(f)
        if mode == 'train':
            self.idxs = self.data['train_ids']
        elif mode == 'valid':
            self.idxs = self.data['valid_ids']
        else:
            self.idxs = self.data['test_ids']

        if length == None:
            self.len = len(self.idxs)
        else:
            self.len = length


    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        """
        Params:
            idx: int: номер единицы данных
        Return:
            sample: dict: словарь, содержащий 'article' - тензор с токенами текста (содержит разделитель). Может оканчиваться pad токенами.
                                              'article_mask' - маска пэддингов
                                              'article_position_ids' - порядковые номера токенов текста без учета пэддингов
                                              'abstract' - тензор с токенами резюме
        """
        idx = self.files[self.idxs[idx]]
        file_name = os.path.join(self.root_dir, str(idx))
        with open(file_name, 'r') as f:
            data = json.load(f)
        # изначально тензоры текста и резюме заполнены pad токенами
        article = self.pad * 924
        abstract = self.pad * 100
        if len(data['abstract']) < 100:
            abstract_content = data['abstract'] + self.tokenizer.encode(self.tokenizer.bos_token)
        else:
            abstract_content = data['abstract'][:-1] + self.tokenizer.encode(self.tokenizer.bos_token)
        abstract[:len(abstract_content)] = abstract_content
        abstract = torch.Tensor(abstract)

        article_content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token)
        article[:len(article_content)] = article_content
        article = torch.Tensor(article)

        # вычисление маски пэддингов и порядковых номеров токенов
        mask = torch.where(article == self.pad[0], torch.zeros_like(article), torch.ones_like(article))
        position_ids = torch.cumsum(mask, -1) - 1

        sample = {'article': article.long(), 'article_mask': mask.long(),
                  'article_position_ids': position_ids.long(), 'abstract': abstract.long()}
        return sample
