"""
LSTM модель для автодополнения текста
Автор: Субботин Игорь
Проект: Text Autocomplete
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class LSTMAutocomplete(nn.Module):
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        logits = self.fc(lstm_out)
        
        return logits
    
    def generate(self, start_tokens: List[int], max_new_tokens: int = 10) -> List[int]:

        self.eval()
        
        device = next(self.parameters()).device
        
        with torch.no_grad():
            input_seq = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            generated = start_tokens.copy()
            
            for _ in range(max_new_tokens):
                logits = self.forward(input_seq)

                next_token_logits = logits[0, -1, :]

                next_token_id = torch.argmax(next_token_logits, dim=-1).item()

                generated.append(next_token_id)

                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                input_seq = torch.cat([input_seq, next_token_tensor], dim=-1)

                if next_token_id == self.pad_idx:
                    break
            
            return generated
    
    def generate_with_temperature(self, start_tokens: List[int], max_new_tokens: int = 10, 
                                 temperature: float = 0.8) -> List[int]:

        self.eval()

        device = next(self.parameters()).device
        
        with torch.no_grad():
            input_seq = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
            generated = start_tokens.copy()
            
            for _ in range(max_new_tokens):
                logits = self.forward(input_seq)
                next_token_logits = logits[0, -1, :] / temperature

                probs = F.softmax(next_token_logits, dim=-1)

                next_token_id = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token_id)
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
                input_seq = torch.cat([input_seq, next_token_tensor], dim=-1)
            
            return generated