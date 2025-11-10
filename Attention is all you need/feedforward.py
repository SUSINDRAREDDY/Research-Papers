from turtle import forward
import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, d_model, hidden_layers, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, hidden_layers)
        self.linear2 = nn.Linear(hidden_layers, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.linear2(self.dropout(self.relu(self.linear1(x)))) # (batch_size, seq_len, d_model)