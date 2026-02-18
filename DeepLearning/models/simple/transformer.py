"""Transformer for Sequential Eye-Tracking Classification"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """Transformer encoder for sequence classification."""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.3, fc_hidden_dim=64, 
                 max_seq_len=500, num_classes=1):
        super(SimpleTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Classification head with deeper FC layers
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, fc_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_size, seq_len)
        Returns:
            (batch_size, num_classes) logits
        """
        # Transpose to (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)
        
        # Project to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Global pooling - both mean and max
        avg_pool = torch.mean(transformer_out, dim=1)  # (batch_size, d_model)
        max_pool = torch.max(transformer_out, dim=1)[0]  # (batch_size, d_model)
        
        # Concatenate pooled features
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, d_model * 2)
        
        # Classification
        out = self.fc(pooled)
        return out
    
    def get_config(self):
        """Return model configuration."""
        return {
            'type': 'Transformer',
            'input_size': self.input_size,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers
        }
