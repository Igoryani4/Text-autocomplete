"""
Функции для оценки качества моделей (LSTM и Transformer)
Автор: Субботин Игорь
Проект: Text Autocomplete
"""

import torch
import numpy as np
from rouge_score import rouge_scorer
from transformers import pipeline
from typing import List, Dict, Tuple

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    if not predictions or not references:
        return {'rouge1': 0.0, 'rouge2': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': []}
    
    for pred, ref in zip(predictions, references):
        try:
            score = scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
        except Exception as e:
            print(f"Ошибка при вычислении ROUGE: {e}")
            continue
    
    return {
        'rouge1': float(np.mean(scores['rouge1'])) if scores['rouge1'] else 0.0,
        'rouge2': float(np.mean(scores['rouge2'])) if scores['rouge2'] else 0.0
    }

def evaluate_lstm_model(
    model, 
    dataloader, 
    dataset, 
    device, 
    num_examples: int = 100
) -> Tuple[Dict[str, float], List[str], List[str]]:
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i * dataloader.batch_size >= num_examples:
                break
                
            inputs = inputs.to(device)
            
            for j in range(min(inputs.size(0), num_examples - len(predictions))):
                input_seq = inputs[j].cpu().tolist()
                input_seq = [x for x in input_seq if x != dataset.pad_idx]
                
                if len(input_seq) < 4:
                    continue

                split_point = max(1, len(input_seq) * 3 // 4)
                prefix_tokens = input_seq[:split_point]
                reference_tokens = input_seq[split_point:]
                
                if len(reference_tokens) == 0:
                    continue
                
                try:
                    generated_ids = model.generate(
                        prefix_tokens, 
                        max_new_tokens=min(len(reference_tokens), 10)
                    )

                    generated_text = dataset.ids_to_text(generated_ids)
                    reference_text = dataset.ids_to_text(reference_tokens)
                    
                    predictions.append(generated_text)
                    references.append(reference_text)
                    
                except Exception as e:
                    print(f"Ошибка при генерации: {e}")
                    continue
    
    if predictions:
        rouge_scores = compute_rouge(predictions, references)
        return rouge_scores, predictions[:10], references[:10]
    
    return {'rouge1': 0.0, 'rouge2': 0.0}, [], []

def evaluate_transformer(
    val_dataset, 
    num_samples: int = 100
) -> Tuple[Dict[str, float], List[str], List[str]]:
    print("\n" + "="*60)
    print("ЗАГРУЗКА TRANSFORMER МОДЕЛИ DistilGPT2")
    print("="*60)
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Используем устройство: {'cuda' if device == 0 else 'cpu'}")

    try:
        generator = pipeline(
            'text-generation', 
            model='distilgpt2',
            device=device
        )
        print("Модель загружена!")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return {'rouge1': 0.0, 'rouge2': 0.0}, [], []
    
    predictions = []
    references = []
    
    print(f"\nОцениваем на {num_samples} примерах...")
    
    for i in range(min(num_samples, len(val_dataset))):
        tokens = val_dataset.df.iloc[i]['token_list']
        
        if len(tokens) < 4:
            continue

        split_point = len(tokens) * 3 // 4
        prefix = ' '.join(tokens[:split_point])
        reference = ' '.join(tokens[split_point:])
        max_new = min(len(tokens) - split_point, 20)
        
        try:
            output = generator(
                prefix,
                max_new_tokens=max_new,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=50256
            )
            
            generated_text = output[0]['generated_text']

            if generated_text.startswith(prefix):
                continuation = generated_text[len(prefix):].strip()
            else:
                continuation = generated_text
            
            predictions.append(continuation)
            references.append(reference)
            
            if (i + 1) % 20 == 0:
                print(f"Обработано {i + 1}/{num_samples} примеров")
            
        except Exception as e:
            print(f"Ошибка на примере {i}: {e}")
            continue
    
    print(f"\n Успешно обработано {len(predictions)} примеров")
    
    if predictions:
        rouge_scores = compute_rouge(predictions, references)
        print(f"\n Результаты DistilGPT2:")
        print(f"   ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"   ROUGE-2: {rouge_scores['rouge2']:.4f}")
        return rouge_scores, predictions[:10], references[:10]
    
    return {'rouge1': 0.0, 'rouge2': 0.0}, [], []

def print_examples(predictions: List[str], references: List[str], num_examples: int = 5):
    print("\n" + "="*60)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ")
    print("="*60)
    
    for i in range(min(num_examples, len(predictions))):
        print(f"\nПример {i+1}:")
        print(f"  Pred: {predictions[i]}")
        print(f"  Ref:  {references[i]}")
        print("-"*40)

def compare_models(
    lstm_rouge: Dict[str, float],
    transformer_rouge: Dict[str, float],
    lstm_examples: Tuple[List[str], List[str]],
    transformer_examples: Tuple[List[str], List[str]]
):
    print("\n" + "="*70)
    print("СРАВНЕНИЕ LSTM vs TRANSFORMER (DistilGPT2)")
    print("="*70)
    
    # Метрики
    print("\n МЕТРИКИ ROUGE:")
    print(f"{'Модель':<20} {'ROUGE-1':<12} {'ROUGE-2':<12}")
    print("-"*44)
    print(f"{'LSTM':<20} {lstm_rouge['rouge1']:.4f}{'':8} {lstm_rouge['rouge2']:.4f}")
    print(f"{'DistilGPT2':<20} {transformer_rouge['rouge1']:.4f}{'':8} {transformer_rouge['rouge2']:.4f}")
    
    # Разница
    diff_1 = transformer_rouge['rouge1'] - lstm_rouge['rouge1']
    diff_2 = transformer_rouge['rouge2'] - lstm_rouge['rouge2']
    print(f"\n Улучшение Transformer:")
    print(f"   ROUGE-1: +{diff_1:.4f}")
    print(f"   ROUGE-2: +{diff_2:.4f}")
    
    # Примеры
    print("\n" + "="*70)
    print("СРАВНЕНИЕ ПРИМЕРОВ")
    print("="*70)
    
    lstm_preds, lstm_refs = lstm_examples
    tf_preds, tf_refs = transformer_examples
    
    for i in range(min(3, len(lstm_preds), len(tf_preds))):
        print(f"\n Пример {i+1}:")
        print(f"   Reference: {lstm_refs[i]}")
        print(f"   LSTM:      {lstm_preds[i]}")
        print(f"   GPT2:      {tf_preds[i]}")
        print("-"*70)