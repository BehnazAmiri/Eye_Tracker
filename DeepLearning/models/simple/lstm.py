"""LSTM for Sequential Eye-Tracking Classification"""

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """Bidirectional LSTM with global pooling for sequence classification."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, 
                 dropout=0.3, fc_hidden_dim=64, bidirectional=True, num_classes=1):
        super(SimpleLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_size * (2 if bidirectional else 1) * 2
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, fc_hidden_dim * 2),
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
        """Forward pass with mean+max pooling."""
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        
        avg_pool = torch.nn.functional.adaptive_avg_pool1d(lstm_out, 1).squeeze(2)
        max_pool = torch.nn.functional.adaptive_max_pool1d(lstm_out, 1).squeeze(2)
        out_features = torch.cat([avg_pool, max_pool], dim=1)
        
        out = self.fc(out_features)
        return out
    
    def get_config(self):
        """Return model configuration."""
        return {
            'type': 'LSTM',
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'bidirectional': self.bidirectional
        }
