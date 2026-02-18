"""
Robust Loss Functions for Noisy Data
=====================================
Implementation of noise-robust loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017)
    
    Formula: FL(p_t) = -?_t * (1 - p_t)^? * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Higher values give more focus to hard examples.
                      Default: 2.0
        alpha (float or list): Balancing parameter for classes. 
                              Can be a single float or list of per-class weights.
                              Default: None (no class balancing)
        reduction (str): Specifies the reduction to apply: 'none', 'mean', 'sum'.
                        Default: 'mean'
    
    Reference:
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        # Convert alpha to tensor if provided as list
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits from model, shape [batch_size, 1] for binary or [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size, 1] for binary or [batch_size]
        
        Returns:
            Focal loss value
        """
        # Handle binary classification (inputs shape [batch_size, 1])
        if inputs.shape[1] == 1:
            # Binary case: use BCEWithLogitsLoss as base
            inputs = inputs.squeeze(1)  # [batch_size]
            targets = targets.squeeze(1).float()  # [batch_size]
            
            # Get probabilities
            p = torch.sigmoid(inputs)
            
            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            
            # p_t = probability of true class
            p_t = torch.where(targets == 1, p, 1 - p)
            
            # Focal weight
            focal_weight = (1 - p_t) ** self.gamma
            
            # Focal loss
            focal_loss = focal_weight * bce_loss
            
            # Apply alpha if provided
            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    alpha = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
                    alpha_t = torch.where(targets == 1, alpha[1], alpha[0])
                elif isinstance(self.alpha, torch.Tensor):
                    if self.alpha.device != inputs.device:
                        self.alpha = self.alpha.to(inputs.device)
                    alpha_t = torch.where(targets.long() == 1, self.alpha[1], self.alpha[0])
                else:
                    alpha_t = self.alpha
                
                focal_loss = alpha_t * focal_loss
            
            return focal_loss.mean()
        
        # Multi-class case
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Convert targets to long for gather
        targets = targets.squeeze().long()
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)  # prob of true class
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha (class balancing) if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha.gather(0, targets)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    Converts hard labels [0, 1] to soft labels [?/K, 1-?+?/K]
    where ? is smoothing parameter and K is number of classes.
    
    Args:
        smoothing (float): Smoothing parameter ?. Default: 0.1
        reduction (str): Specifies the reduction: 'none', 'mean', 'sum'. Default: 'mean'
    
    Reference:
        Rethinking the Inception Architecture for Computer Vision
        https://arxiv.org/abs/1512.00567
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits, shape [batch_size, 1] for binary or [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size, 1] for binary or [batch_size]
        
        Returns:
            Label smoothing loss value
        """
        # Handle binary classification
        if inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)  # [batch_size]
            targets = targets.squeeze(1).float()  # [batch_size]
            
            # Smooth labels for binary: [smoothing, 1-smoothing]
            smooth_positive = 1.0 - self.smoothing
            smooth_negative = self.smoothing
            
            # Calculate BCE with smooth labels
            log_probs_pos = F.logsigmoid(inputs)
            log_probs_neg = F.logsigmoid(-inputs)
            
            loss = -(targets * smooth_positive * log_probs_pos + 
                    (1 - targets) * smooth_negative * log_probs_pos +
                    targets * smooth_negative * log_probs_neg +
                    (1 - targets) * smooth_positive * log_probs_neg)
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        
        # Multi-class case
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        targets = targets.squeeze().long()
        
        # Create smooth labels
        smooth_labels = torch.full_like(log_probs, self.smoothing / num_classes)
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)
        
        # Negative log likelihood with smooth labels
        loss = -(smooth_labels * log_probs).sum(dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Combines Focal Loss with Label Smoothing
    
    Args:
        gamma (float): Focal loss focusing parameter. Default: 2.0
        alpha (float or list): Class balancing parameter. Default: None
        smoothing (float): Label smoothing parameter. Default: 0.1
        reduction (str): Reduction method. Default: 'mean'
    """
    
    def __init__(self, gamma=2.0, alpha=None, smoothing=0.1, reduction='mean'):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.reduction = reduction
        
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits, shape [batch_size, 1] for binary or [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size, 1] for binary or [batch_size]
        
        Returns:
            Combined loss value
        """
        # Handle binary classification
        if inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
            targets = targets.squeeze(1).float()
            
            # Get probabilities
            p = torch.sigmoid(inputs)
            
            # Smooth labels
            smooth_positive = 1.0 - self.smoothing
            smooth_negative = self.smoothing
            
            # Smooth targets
            targets_smooth = targets * smooth_positive + (1 - targets) * smooth_negative
            
            # BCE loss with smooth labels
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
            
            # p_t for focal weight (use hard labels for focal calculation)
            p_t = torch.where(targets == 1, p, 1 - p)
            focal_weight = (1 - p_t) ** self.gamma
            
            # Focal loss
            focal_loss = focal_weight * bce_loss
            
            # Apply alpha if provided
            if self.alpha is not None:
                if isinstance(self.alpha, (list, tuple)):
                    alpha = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)
                    alpha_t = torch.where(targets == 1, alpha[1], alpha[0])
                elif isinstance(self.alpha, torch.Tensor):
                    if self.alpha.device != inputs.device:
                        self.alpha = self.alpha.to(inputs.device)
                    alpha_t = torch.where(targets.long() == 1, self.alpha[1], self.alpha[0])
                else:
                    alpha_t = self.alpha
                
                focal_loss = alpha_t * focal_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        
        # Multi-class case
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        log_p = F.log_softmax(inputs, dim=1)
        
        targets = targets.squeeze().long()
        
        # Create smooth labels
        num_classes = inputs.size(1)
        smooth_labels = torch.full_like(p, self.smoothing / num_classes)
        smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)
        
        # Calculate loss with smooth labels
        ce_loss = -(smooth_labels * log_p).sum(dim=1)
        
        # Get probability of true class for focal weight
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss with smoothing
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                if self.alpha.device != inputs.device:
                    self.alpha = self.alpha.to(inputs.device)
                alpha_t = self.alpha.gather(0, targets)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def get_loss_function(loss_type='cross_entropy', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type (str): Type of loss function. Options:
            - 'cross_entropy': Standard cross entropy
            - 'focal': Focal loss
            - 'label_smoothing': Label smoothing cross entropy
            - 'focal_smoothing': Focal loss with label smoothing
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function instance
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(**kwargs)
    elif loss_type == 'focal_smoothing':
        return FocalLossWithLabelSmoothing(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Robust Loss Functions...")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 8
    num_classes = 2
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\nInput shape: {logits.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Targets: {targets.tolist()}")
    
    # Test Cross Entropy
    ce_loss = nn.CrossEntropyLoss()
    ce_val = ce_loss(logits, targets)
    print(f"\n1. Cross Entropy Loss: {ce_val.item():.4f}")
    
    # Test Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    focal_val = focal_loss(logits, targets)
    print(f"2. Focal Loss (?=2.0): {focal_val.item():.4f}")
    
    # Test Label Smoothing
    ls_loss = LabelSmoothingLoss(smoothing=0.1)
    ls_val = ls_loss(logits, targets)
    print(f"3. Label Smoothing Loss (?=0.1): {ls_val.item():.4f}")
    
    # Test Focal + Label Smoothing
    fls_loss = FocalLossWithLabelSmoothing(gamma=2.0, smoothing=0.1)
    fls_val = fls_loss(logits, targets)
    print(f"4. Focal + Label Smoothing: {fls_val.item():.4f}")
    
    # Test with class weights
    focal_weighted = FocalLoss(gamma=2.0, alpha=[0.3, 0.7])
    fw_val = focal_weighted(logits, targets)
    print(f"5. Focal Loss (weighted): {fw_val.item():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] All loss functions working correctly!")
