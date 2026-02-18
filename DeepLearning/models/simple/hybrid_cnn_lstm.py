"""Hybrid CNN-LSTM for Sequential Eye-Tracking Classification"""

import torch
import torch.nn as nn


class HybridCNNLSTM(nn.Module):
    """Hybrid model combining CNN feature extraction with LSTM temporal modeling."""
    
    def __init__(self, input_channels, cnn_channels=64, cnn_kernel_size=7,
                 lstm_hidden_size=128, lstm_num_layers=2, dropout=0.3, fc_hidden_dim=64, num_classes=1):
        super(HybridCNNLSTM, self).__init__()
        
        self.input_channels = input_channels
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=cnn_kernel_size, padding=cnn_kernel_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.Dropout(dropout),
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = lstm_hidden_size * 2 * 2
        
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
        """Forward pass with global pooling."""
        x = self.conv_layers(x)
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
            'type': 'Hybrid_CNN_LSTM',
            'input_channels': self.input_channels
        }
