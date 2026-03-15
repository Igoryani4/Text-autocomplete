"""
Функции для тренировки LSTM модели
Автор: Субботин Игорь
Проект: Text Autocomplete
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Tuple, Dict
import sys
sys.path.append('.')
from src.evaluate import evaluate_lstm_model

def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(
            outputs.reshape(-1, outputs.size(-1)),
            targets.reshape(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model: nn.Module, dataloader: DataLoader, 
             criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_lstm(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
               dataset, device: torch.device, epochs: int = 5, lr: float = 0.001) -> Tuple[List, List, List]:

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    train_losses = []
    val_losses = []
    rouge_scores = []
    
    print("Начинаем тренировку...")
    for epoch in range(epochs):
        print(f"\n--- Эпоха {epoch + 1}/{epochs} ---")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        rouge, examples_pred, examples_ref = evaluate_lstm_model(
            model, val_loader, dataset, device, num_examples=50
        )
        rouge_scores.append(rouge)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"ROUGE-1: {rouge['rouge1']:.4f}, ROUGE-2: {rouge['rouge2']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if examples_pred:
            print("\nПримеры предсказаний:")
            for i in range(min(3, len(examples_pred))):
                print(f"  Pred: {examples_pred[i]}")
                print(f"  Ref:  {examples_ref[i]}")
                print()
    
    return train_losses, val_losses, rouge_scores