"""MLP for Eye-Tracking Classification (Aggregated Features)"""

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """Multi-Layer Perceptron for binary classification."""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3, use_batch_norm=True, num_classes=2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm: 
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer: num_classes for CrossEntropyLoss, 1 for BCELoss
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def get_config(self):
        """Return model configuration."""
        return {
            'type': 'MLP',
            'architecture': str(self.model)
        }
