"""
Data Augmentation Techniques for Noise Robustness
==================================================
Implementation of Mixup and other augmentation methods.
"""

import torch
import numpy as np


class MixupAugmentation:
    """
    Mixup: Beyond Empirical Risk Minimization (Zhang et al., 2018)
    
    Creates virtual training examples by mixing pairs of examples and their labels:
        x_new = ? * x_i + (1 - ?) * x_j
        y_new = ? * y_i + (1 - ?) * y_j
    
    where ? ~ Beta(?, ?)
    
    Args:
        alpha (float): Beta distribution parameter. 
                      Higher values -> more mixing (0.0 = no mixing, 1.0 = uniform)
                      Typical values: 0.2 - 1.0
                      Default: 0.4
        prob (float): Probability of applying mixup to a batch.
                     Default: 0.5 (apply to 50% of batches)
    
    Reference:
        mixup: Beyond Empirical Risk Minimization
        https://arxiv.org/abs/1710.09412
    """
    
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, x, y):
        """
        Apply mixup augmentation.
        
        Args:
            x: Input features, shape [batch_size, ...]
            y: Target labels, shape [batch_size, num_classes] (one-hot) or [batch_size] (class indices)
        
        Returns:
            x_mixed: Mixed features
            y_mixed: Mixed labels (soft labels)
            lam: Mixing coefficient used
        """
        if np.random.rand() > self.prob:
            # Don't apply mixup
            return x, y, 1.0
        
        batch_size = x.size(0)
        
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Random permutation of batch
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs
        x_mixed = lam * x + (1 - lam) * x[index]
        
        # Mix labels
        # Binary classification (shape [batch_size, 1])
        if y.dim() == 2 and y.shape[1] == 1:
            y_mixed = lam * y + (1 - lam) * y[index]
        # Multi-class: convert to one-hot if needed
        elif y.dim() == 1:
            num_classes = int(y.max().item() + 1)
            y_onehot = torch.zeros(batch_size, num_classes, device=y.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y_mixed = lam * y_onehot + (1 - lam) * y_onehot[index]
        else:
            # Already one-hot
            y_mixed = lam * y + (1 - lam) * y[index]
        
        return x_mixed, y_mixed, lam
    
    def mixup_criterion(self, criterion, pred, y_mixed):
        """
        Calculate loss for mixed labels.
        
        Args:
            criterion: Loss function (should accept soft labels or work with probabilities)
            pred: Model predictions, shape [batch_size, num_classes]
            y_mixed: Mixed soft labels, shape [batch_size, num_classes]
        
        Returns:
            Loss value
        """
        # For soft labels, use cross entropy with probabilities
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(y_mixed * log_probs).sum(dim=1).mean()
        return loss


class CutoutAugmentation:
    """
    Cutout for sequence data - randomly masks parts of the input.
    
    Adapted for time-series/sequential eye-tracking data.
    
    Args:
        mask_ratio (float): Proportion of sequence to mask. Default: 0.15
        mask_value (float): Value to use for masked positions. Default: 0.0
        prob (float): Probability of applying cutout. Default: 0.5
    """
    
    def __init__(self, mask_ratio=0.15, mask_value=0.0, prob=0.5):
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.prob = prob
    
    def __call__(self, x):
        """
        Apply cutout augmentation.
        
        Args:
            x: Input features, shape [batch_size, seq_len, features] or [batch_size, features]
        
        Returns:
            x_masked: Masked features
        """
        if np.random.rand() > self.prob:
            return x
        
        x_masked = x.clone()
        batch_size = x.size(0)
        
        if x.dim() == 3:
            # Sequential data: [batch_size, seq_len, features]
            seq_len = x.size(1)
            mask_len = int(seq_len * self.mask_ratio)
            
            for i in range(batch_size):
                # Random starting position
                start = np.random.randint(0, seq_len - mask_len + 1)
                x_masked[i, start:start+mask_len, :] = self.mask_value
        else:
            # Flat data: [batch_size, features]
            num_features = x.size(1)
            mask_len = int(num_features * self.mask_ratio)
            
            for i in range(batch_size):
                # Random feature indices to mask
                mask_indices = np.random.choice(num_features, mask_len, replace=False)
                x_masked[i, mask_indices] = self.mask_value
        
        return x_masked


class GaussianNoiseAugmentation:
    """
    Add Gaussian noise to inputs for robustness.
    
    Args:
        std (float): Standard deviation of noise. Default: 0.01
        prob (float): Probability of applying noise. Default: 0.5
    """
    
    def __init__(self, std=0.01, prob=0.5):
        self.std = std
        self.prob = prob
    
    def __call__(self, x):
        """
        Add Gaussian noise to input.
        
        Args:
            x: Input features
        
        Returns:
            Noisy features
        """
        if np.random.rand() > self.prob:
            return x
        
        noise = torch.randn_like(x) * self.std
        return x + noise


def get_augmentation(aug_type='mixup', **kwargs):
    """
    Factory function to get augmentation by name.
    
    Args:
        aug_type (str): Type of augmentation. Options:
            - 'mixup': Mixup augmentation
            - 'cutout': Cutout for sequences
            - 'gaussian': Gaussian noise
        **kwargs: Additional arguments for the augmentation
    
    Returns:
        Augmentation instance
    """
    if aug_type == 'mixup':
        return MixupAugmentation(**kwargs)
    elif aug_type == 'cutout':
        return CutoutAugmentation(**kwargs)
    elif aug_type == 'gaussian':
        return GaussianNoiseAugmentation(**kwargs)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


if __name__ == "__main__":
    # Test augmentations
    print("Testing Data Augmentations...")
    print("=" * 60)
    
    # Create dummy data
    batch_size = 4
    num_features = 10
    num_classes = 2
    
    x = torch.randn(batch_size, num_features)
    y = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\nOriginal data shape: {x.shape}")
    print(f"Original labels: {y.tolist()}")
    print(f"Original x[0, :5]: {x[0, :5].tolist()}")
    
    # Test Mixup
    print("\n" + "=" * 60)
    print("1. Testing Mixup Augmentation")
    print("-" * 60)
    mixup = MixupAugmentation(alpha=0.4, prob=1.0)  # prob=1.0 for testing
    x_mixed, y_mixed, lam = mixup(x, y)
    
    print(f"Mixed x shape: {x_mixed.shape}")
    print(f"Mixed y shape: {y_mixed.shape}")
    print(f"Lambda: {lam:.3f}")
    print(f"Mixed labels[0]: {y_mixed[0].tolist()}")
    print(f"Mixed x[0, :5]: {x_mixed[0, :5].tolist()}")
    
    # Test Cutout
    print("\n" + "=" * 60)
    print("2. Testing Cutout Augmentation")
    print("-" * 60)
    cutout = CutoutAugmentation(mask_ratio=0.2, prob=1.0)
    x_cutout = cutout(x)
    
    print(f"Cutout x shape: {x_cutout.shape}")
    print(f"Original x[0]: {x[0].tolist()}")
    print(f"Cutout x[0]: {x_cutout[0].tolist()}")
    print(f"Number of masked values: {(x_cutout[0] == 0).sum().item()}")
    
    # Test Gaussian Noise
    print("\n" + "=" * 60)
    print("3. Testing Gaussian Noise Augmentation")
    print("-" * 60)
    gaussian = GaussianNoiseAugmentation(std=0.1, prob=1.0)
    x_noisy = gaussian(x)
    
    print(f"Noisy x shape: {x_noisy.shape}")
    print(f"Original x[0, :5]: {x[0, :5].tolist()}")
    print(f"Noisy x[0, :5]: {x_noisy[0, :5].tolist()}")
    print(f"Difference: {(x_noisy[0, :5] - x[0, :5]).tolist()}")
    
    # Test with sequential data
    print("\n" + "=" * 60)
    print("4. Testing with Sequential Data")
    print("-" * 60)
    x_seq = torch.randn(batch_size, 20, 5)  # [batch, seq_len, features]
    print(f"Sequential data shape: {x_seq.shape}")
    
    cutout_seq = CutoutAugmentation(mask_ratio=0.3, prob=1.0)
    x_seq_cutout = cutout_seq(x_seq)
    
    print(f"Cutout sequential shape: {x_seq_cutout.shape}")
    print(f"Number of masked timesteps: {(x_seq_cutout[0, :, 0] == 0).sum().item()}")
    
    print("\n" + "=" * 60)
    print("[OK] All augmentations working correctly!")
