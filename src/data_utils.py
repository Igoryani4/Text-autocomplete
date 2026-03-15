"""
Функции для загрузки и очистки данных
Автор: Субботин Игорь
Проект: Text Autocomplete
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

def load_data(filepath: str, n_rows: Optional[int] = None) -> pd.DataFrame:
    
    print(f"Загрузка данных из {filepath}...")
    columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(filepath, encoding='latin-1', names=columns, nrows=n_rows)
    print(f"Загружено {len(df)} строк")
    return df[['text']]

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    return text.split()

def prepare_dataset(df: pd.DataFrame, sample_size: Optional[int] = None) -> pd.DataFrame:
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    print("Очистка текстов...")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'] != ''].reset_index(drop=True)
    print("Токенизация...")
    df['tokens'] = df['clean_text'].apply(tokenize)
    print(f"Готово! {len(df)} примеров")
    return df

def split_data(df: pd.DataFrame, data_dir: str = './data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val, test = train_test_split(df, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1111, random_state=42)
    
    print(f"Train: {len(train)} примеров")
    print(f"Val: {len(val)} примеров")
    print(f"Test: {len(test)} примеров")

    train.to_csv(f'{data_dir}/train.csv', index=False)
    val.to_csv(f'{data_dir}/val.csv', index=False)
    test.to_csv(f'{data_dir}/test.csv', index=False)
    
    return train, val, test

if __name__ == "__main__":
    df = load_data('data/training.1600000.processed.noemoticon.csv', n_rows=1000)
    df_clean = prepare_dataset(df)
    split_data(df_clean)
    print("Тест завершен успешно!")