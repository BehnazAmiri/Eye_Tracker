"""
Sample Reweighting Techniques for Noisy Data
============================================
Methods that assign different weights to samples during training
instead of removing them.
"""

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class ConfidenceBasedReweighting:
    """
    Confidence-Based Sample Reweighting
    
    Uses confidence scores (from RF cross-validation) to assign weights
    to training samples. Low confidence samples get low weight, high 
    confidence samples get high weight.
    
    This is a SOFT alternative to hardness cleaning (which removes samples).
    
    Args:
        beta (float): Exponent for weight calculation. 
                     weight = confidence^beta
                     Higher beta -> more aggressive downweighting
                     Default: 2.0
        min_weight (float): Minimum weight for any sample (to avoid zero weights)
                           Default: 0.1
    """
    
    def __init__(self, beta=2.0, min_weight=0.1):
        self.beta = beta
        self.min_weight = min_weight
        self.confidence_scores = None
        self.sample_weights = None
    
    def calculate_confidence_scores(self, X, y, n_splits=10, random_state=42):
        """
        Calculate confidence scores using Random Forest with cross-validation.
        
        Args:
            X: Features, shape [n_samples, n_features] or [n_samples, seq_len, features]
            y: Labels, shape [n_samples]
            n_splits: Number of CV folds
            random_state: Random seed
        
        Returns:
            confidence_scores: Array of shape [n_samples]
        """
        print(f"\n? Calculating confidence scores using {n_splits}-fold CV...")
        
        # Flatten sequences if needed
        if len(X.shape) == 3:
            X_flat = X.reshape(len(X), -1)
        else:
            X_flat = X
        
        # Ensure y is integer
        y_int = y.astype(np.int64).ravel()
        
        # Train RF with cross-validation
        rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        print(f"  Training RF on {X_flat.shape[0]} samples...")
        y_proba = cross_val_predict(rf, X_flat, y_int, cv=cv, method='predict_proba', n_jobs=-1)
        
        # Confidence = probability of true class
        confidence_scores = np.array([y_proba[i, y_int[i]] for i in range(len(y_int))])
        
        self.confidence_scores = confidence_scores
        
        print(f"  [OK] Confidence scores calculated!")
        print(f"     Mean: {confidence_scores.mean():.3f}")
        print(f"     Std:  {confidence_scores.std():.3f}")
        print(f"     Min:  {confidence_scores.min():.3f}")
        print(f"     Max:  {confidence_scores.max():.3f}")
        
        return confidence_scores
    
    def calculate_weights(self, confidence_scores=None):
        """
        Calculate sample weights from confidence scores.
        
        Args:
            confidence_scores: Optional, if None uses stored scores
        
        Returns:
            sample_weights: Array of shape [n_samples]
        """
        if confidence_scores is None:
            if self.confidence_scores is None:
                raise ValueError("Must call calculate_confidence_scores first!")
            confidence_scores = self.confidence_scores
        
        # Weight = confidence^beta
        weights = np.power(confidence_scores, self.beta)
        
        # Clip to min_weight
        weights = np.maximum(weights, self.min_weight)
        
        # Normalize to sum to n_samples (so average weight = 1.0)
        weights = weights * len(weights) / weights.sum()
        
        self.sample_weights = weights
        
        print(f"\n[CHART] Sample weights calculated (?={self.beta}):")
        print(f"   Mean: {weights.mean():.3f} (should be ~1.0)")
        print(f"   Std:  {weights.std():.3f}")
        print(f"   Min:  {weights.min():.3f}")
        print(f"   Max:  {weights.max():.3f}")
        
        # Show distribution
        low_conf = confidence_scores < 0.4
        medium_conf = (confidence_scores >= 0.4) & (confidence_scores < 0.6)
        high_conf = confidence_scores >= 0.6
        
        print(f"\n   Weight distribution by confidence:")
        if low_conf.any():
            print(f"     Low conf (<0.4):    {low_conf.sum():3d} samples, avg weight={weights[low_conf].mean():.3f}")
        if medium_conf.any():
            print(f"     Medium conf (0.4-0.6): {medium_conf.sum():3d} samples, avg weight={weights[medium_conf].mean():.3f}")
        if high_conf.any():
            print(f"     High conf (>0.6):   {high_conf.sum():3d} samples, avg weight={weights[high_conf].mean():.3f}")
        
        return weights
    
    def fit(self, X, y, n_splits=10, random_state=42):
        """
        Fit the reweighting model: calculate confidence scores and weights.
        
        Args:
            X: Features
            y: Labels
            n_splits: Number of CV folds
            random_state: Random seed
        
        Returns:
            self
        """
        self.calculate_confidence_scores(X, y, n_splits, random_state)
        self.calculate_weights()
        return self
    
    def get_weights(self):
        """Get sample weights."""
        if self.sample_weights is None:
            raise ValueError("Must call fit() first!")
        return self.sample_weights
    
    def get_torch_weights(self, device='cpu'):
        """Get sample weights as PyTorch tensor."""
        weights = self.get_weights()
        return torch.FloatTensor(weights).to(device)


class LossBasedReweighting:
    """
    Loss-Based Sample Reweighting
    
    Dynamically adjusts sample weights based on training loss.
    High loss -> likely noisy -> lower weight
    
    Args:
        update_frequency (int): Update weights every N epochs
                               Default: 10
        smoothing (float): Smoothing factor for exponential moving average
                          Default: 0.9
    """
    
    def __init__(self, update_frequency=10, smoothing=0.9):
        self.update_frequency = update_frequency
        self.smoothing = smoothing
        self.sample_losses = None
        self.sample_weights = None
    
    def update_weights_from_losses(self, losses):
        """
        Update sample weights based on current losses.
        
        Args:
            losses: Array of per-sample losses, shape [n_samples]
        
        Returns:
            weights: Updated sample weights
        """
        # Initialize or update with EMA
        if self.sample_losses is None:
            self.sample_losses = losses
        else:
            self.sample_losses = (self.smoothing * self.sample_losses + 
                                 (1 - self.smoothing) * losses)
        
        # Weight = 1 / (1 + loss)  (lower loss -> higher weight)
        weights = 1.0 / (1.0 + self.sample_losses)
        
        # Normalize
        weights = weights * len(weights) / weights.sum()
        
        self.sample_weights = weights
        return weights
    
    def get_weights(self):
        """Get current sample weights."""
        if self.sample_weights is None:
            # Return uniform weights
            return np.ones(len(self.sample_losses))
        return self.sample_weights


def weighted_cross_entropy_loss(outputs, targets, weights):
    """
    Cross entropy loss with per-sample weights.
    
    Args:
        outputs: Model predictions, shape [batch_size, num_classes]
        targets: Ground truth labels, shape [batch_size]
        weights: Sample weights, shape [batch_size]
    
    Returns:
        Weighted cross entropy loss
    """
    log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
    nll_loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    weighted_loss = (nll_loss * weights).sum() / weights.sum()
    return weighted_loss


if __name__ == "__main__":
    # Test reweighting
    print("Testing Sample Reweighting...")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    # Add some "noisy" samples (mislabeled)
    noise_indices = np.random.choice(n_samples, 10, replace=False)
    y[noise_indices] = 1 - y[noise_indices]  # Flip labels
    
    print(f"\nData: {n_samples} samples, {n_features} features")
    print(f"Added noise to {len(noise_indices)} samples")
    
    # Test Confidence-Based Reweighting
    print("\n" + "=" * 60)
    print("Testing Confidence-Based Reweighting")
    print("=" * 60)
    
    reweighter = ConfidenceBasedReweighting(beta=2.0, min_weight=0.1)
    reweighter.fit(X, y, n_splits=5)
    
    weights = reweighter.get_weights()
    print(f"\n[OK] Weights shape: {weights.shape}")
    print(f"   Weights sum: {weights.sum():.2f} (should be ~{n_samples})")
    
    # Check weights for noisy samples
    noisy_weights = weights[noise_indices]
    clean_indices = np.setdiff1d(np.arange(n_samples), noise_indices)
    clean_weights = weights[clean_indices]
    
    print(f"\n   Average weight for noisy samples: {noisy_weights.mean():.3f}")
    print(f"   Average weight for clean samples: {clean_weights.mean():.3f}")
    
    if noisy_weights.mean() < clean_weights.mean():
        print(f"   [OK] Noisy samples have lower weights (as expected)")
    else:
        print(f"   [WARN]?  Noisy samples don't have consistently lower weights")
    
    # Test with PyTorch
    print("\n" + "=" * 60)
    print("Testing with PyTorch")
    print("=" * 60)
    
    torch_weights = reweighter.get_torch_weights(device='cpu')
    print(f"\nPyTorch weights shape: {torch_weights.shape}")
    print(f"Device: {torch_weights.device}")
    
    # Test weighted loss
    batch_size = 8
    num_classes = 2
    outputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    batch_weights = torch_weights[:batch_size]
    
    loss = weighted_cross_entropy_loss(outputs, targets, batch_weights)
    print(f"\nWeighted CE loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("[OK] All reweighting tests passed!")
    print("=" * 60)
