"""
Dataset для задачи next token prediction
Автор: Субботин Игорь
Проект: Text Autocomplete
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import ast

class NextTokenDataset(Dataset):
    def __init__(self, csv_file: str, max_len: int = 20, min_len: int = 3):
        self.df = pd.read_csv(csv_file)
        self.max_len = max_len
        self.min_len = min_len
        
        self.df['token_list'] = self.df['tokens'].apply(ast.literal_eval)
        
        self.df = self.df[self.df['token_list'].apply(len) >= min_len]
        self.df = self.df[self.df['token_list'].apply(len) <= max_len]
        self.df = self.df.reset_index(drop=True)

        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)

        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

        self.pad_idx = self.vocab.get(self.pad_token, 0)
        self.unk_idx = self.vocab.get(self.unk_token, 1)

        self.idx2word = {v: k for k, v in self.vocab.items()}
        
        print(f"Загружено {len(self.df)} примеров")
        print(f"Размер словаря: {self.vocab_size}")
        
    def _build_vocab(self) -> dict:
        vocab = {'<PAD>': 0, '<UNK>': 1}
        
        for tokens in self.df['token_list']:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        
        return vocab
    
    def text_to_ids(self, tokens: list) -> list:
        return [self.vocab.get(token, self.unk_idx) for token in tokens]
    
    def ids_to_text(self, ids: list) -> str:
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in ids]
        # Убираем специальные токены для читаемости
        tokens = [t for t in tokens if t not in ['<PAD>', '<UNK>']]
        return ' '.join(tokens)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        tokens = self.df.iloc[idx]['token_list']

        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        input_ids = self.text_to_ids(input_tokens)
        target_ids = self.text_to_ids(target_tokens)

        target_length = self.max_len - 1
        if len(input_ids) < target_length:
            input_ids = input_ids + [self.pad_idx] * (target_length - len(input_ids))
            target_ids = target_ids + [self.pad_idx] * (target_length - len(target_ids))
        else:
            input_ids = input_ids[:target_length]
            target_ids = target_ids[:target_length]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


def collate_batch(batch):
    inputs, targets = zip(*batch)

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded