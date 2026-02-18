"""1D CNN for Sequential Eye-Tracking Classification"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for sequence classification."""
    
    def __init__(self, input_channels, num_conv_layers=4, base_channels=64, 
                 kernel_size=7, dropout=0.3, fc_hidden_dim=128, num_classes=1):
        super(CNN1D, self).__init__()
        
        self.input_channels = input_channels
        
        conv_layers = []
        in_ch = input_channels
        
        for i in range(num_conv_layers):
            out_ch = base_channels * (2 ** min(i, 2))
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            in_ch = out_ch
            if kernel_size > 3:
                kernel_size -= 2
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        self.global_max = nn.AdaptiveMaxPool1d(1)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        
        # FC layers - DEEPER with more capacity (2x channels because of concat)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_ch * 2, fc_hidden_dim * 2),
            nn.BatchNorm1d(fc_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim * 2, fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.BatchNorm1d(fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_channels, sequence_length)
        Returns:
            (batch_size, 1) logits (no sigmoid applied)
        """
        x = self.conv_layers(x)
        
        # Apply both poolings
        x_max = self.global_max(x)
        x_avg = self.global_avg(x)
        
        # Concatenate along channel dimension (which is dim 1)
        x = torch.cat([x_max, x_avg], dim=1)
        
        x = self.fc(x)
        return x
    
    def get_config(self):
        """Return model configuration."""
        return {
            'type': 'CNN',
            'input_channels': self.input_channels
        }
