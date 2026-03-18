"""
Data Augmentation for Eye-Tracking Time-Series
===============================================
Implements augmentation for sequential eye-tracking data to:
1. Increase training set size (helps generalization)
2. Balance class distribution (reduces false positives on class 0)

Main function: apply_augmentation(X_train, y_train, config)

Input shape: (n_samples, n_features, seq_len)
Output shape: (n_samples_augmented, n_features, seq_len)

Strategy:
- Class 0 (low cognitive load) gets augmented MORE aggressively
  to counteract the training imbalance (77 class0 vs 125 class1).
- This directly addresses the 10 False Positives problem in the best LSTM model.
"""

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample augmentation functions
# All functions work on shape (n_features, seq_len)
# ─────────────────────────────────────────────────────────────────────────────

def _add_gaussian_noise(x: np.ndarray, std_ratio: float = 0.03) -> np.ndarray:
    """
    Add Gaussian noise proportional to per-feature standard deviation.
    Simulates sensor measurement noise.

    Args:
        x: (n_features, seq_len)
        std_ratio: noise std = std_ratio * feature_std  (default 3%)
    """
    x_aug = x.copy()
    for f in range(x.shape[0]):
        feat_std = np.std(x[f]) + 1e-8
        noise = np.random.normal(0, std_ratio * feat_std, x.shape[1])
        x_aug[f] += noise
    return x_aug


def _time_shift(x: np.ndarray, max_shift: int = 30) -> np.ndarray:
    """
    Randomly shift the sequence in time by up to max_shift steps.
    Simulates variation in response timing.

    Args:
        x: (n_features, seq_len)
        max_shift: maximum shift in timesteps (default 30 @ 150Hz = 0.2s)
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return x.copy()
    x_aug = np.zeros_like(x)
    if shift > 0:
        x_aug[:, shift:] = x[:, :-shift]
        x_aug[:, :shift] = x[:, :shift][:, ::-1]    # mirror-pad
    else:
        s = -shift
        x_aug[:, :x.shape[1]-s] = x[:, s:]
        x_aug[:, x.shape[1]-s:] = x[:, x.shape[1]-s:][:, ::-1]  # mirror-pad
    return x_aug


def _scale(x: np.ndarray, scale_range: Tuple[float, float] = (0.92, 1.08)) -> np.ndarray:
    """
    Apply random per-feature amplitude scaling.
    Simulates individual differences in pupil dilation magnitude.

    Args:
        x: (n_features, seq_len)
        scale_range: (min_scale, max_scale)
    """
    x_aug = x.copy()
    for f in range(x.shape[0]):
        factor = np.random.uniform(scale_range[0], scale_range[1])
        x_aug[f] *= factor
    return x_aug


def _time_warp(x: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Smooth random warping of the time axis.
    Simulates variations in response speed.

    Args:
        x: (n_features, seq_len)
        sigma: warp intensity (default 5%)
    """
    seq_len = x.shape[1]
    # Create smooth random warp curve
    knots = np.linspace(0, seq_len - 1, num=5)
    warp_vals = knots + np.random.normal(0, sigma * seq_len, size=5)
    warp_vals[0] = 0
    warp_vals[-1] = seq_len - 1
    warp_vals = np.clip(np.sort(warp_vals), 0, seq_len - 1)

    orig_idx = np.arange(seq_len)
    new_idx = np.interp(orig_idx, knots, warp_vals)
    new_idx = np.clip(new_idx, 0, seq_len - 1)

    x_aug = np.zeros_like(x)
    for f in range(x.shape[0]):
        x_aug[f] = np.interp(orig_idx, new_idx, x[f])
    return x_aug


# ─────────────────────────────────────────────────────────────────────────────
# Augment a single sample using a combination of methods
# ─────────────────────────────────────────────────────────────────────────────

def _augment_sample(x: np.ndarray, methods: list) -> np.ndarray:
    """Apply a randomly chosen subset of augmentation methods to one sample."""
    x_aug = x.copy()

    # Randomly pick which methods to apply (at least 1)
    n_apply = np.random.randint(1, len(methods) + 1)
    chosen = np.random.choice(methods, size=n_apply, replace=False)

    for method in chosen:
        if method == 'noise':
            x_aug = _add_gaussian_noise(x_aug)
        elif method == 'time_shift':
            x_aug = _time_shift(x_aug)
        elif method == 'scale':
            x_aug = _scale(x_aug)
        elif method == 'time_warp':
            x_aug = _time_warp(x_aug)

    return x_aug


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def apply_augmentation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply class-balanced data augmentation to training data.

    Strategy:
        • Class 0 (low cognitive load / minority) is augmented MORE to
          reduce the training imbalance and lower false positives.
        • Class 0 is replicated until it matches or slightly exceeds class 1 count.
        • Class 1 is augmented by aug_factor - 1 additional copies.

    Args:
        X_train : (n_samples, n_features, seq_len) numpy array
        y_train : (n_samples,) numpy array of labels (0 or 1)
        config  : Config object with config.augmentation dict

    Returns:
        X_aug : augmented training data
        y_aug : augmented labels
    """
    aug_cfg = config.augmentation if hasattr(config, 'augmentation') else {}
    aug_factor = aug_cfg.get('augment_factor', 2)

    # Parse methods (may be list or comma-separated string)
    methods_raw = aug_cfg.get('augment_methods', ['noise', 'scale', 'time_shift'])
    if isinstance(methods_raw, str):
        methods = [m.strip() for m in methods_raw.split(',') if m.strip()]
    else:
        methods = list(methods_raw)

    n_total = len(X_train)
    n_class0 = int(np.sum(y_train == 0))
    n_class1 = int(np.sum(y_train == 1))

    idx_class0 = np.where(y_train == 0)[0]
    idx_class1 = np.where(y_train == 1)[0]

    print(f"\n  [AUG] Class distribution before: class0={n_class0}, class1={n_class1}")
    print(f"  [AUG] Augmentation factor: {aug_factor}x  |  Methods: {methods}")
    print(f"  [AUG] Strategy: BALANCE-FIRST -> oversample class0 to class1 count, then scale both")

    # ── Phase 1: Balance (bring class0 up to class1 count) ──
    # e.g. 77 class0 → 125 class0 (48 extra augmented samples)
    # This directly reduces False Positives in the LSTM best model
    n_balance_extra = max(0, n_class1 - n_class0)  # e.g. 125-77 = 48

    # ── Phase 2: Scale both classes by (aug_factor-1) extra copies ──
    # e.g. aug_factor=2: +1 full copy of each → 250 class0, 250 class1
    n_scale_extra_per_class = (aug_factor - 1) * n_class1  # both same size after balancing

    n_extra_class0 = n_balance_extra + n_scale_extra_per_class
    n_extra_class1 = n_scale_extra_per_class

    X_new_class0 = []
    y_new_class0 = []
    for _ in range(n_extra_class0):
        src_idx = np.random.choice(idx_class0)
        x_aug = _augment_sample(X_train[src_idx], methods)
        X_new_class0.append(x_aug)
        y_new_class0.append(0)

    X_new_class1 = []
    y_new_class1 = []
    for _ in range(n_extra_class1):
        src_idx = np.random.choice(idx_class1)
        x_aug = _augment_sample(X_train[src_idx], methods)
        X_new_class1.append(x_aug)
        y_new_class1.append(1)

    # ── Combine original + augmented ──
    parts_X = [X_train]
    parts_y = [y_train]

    if X_new_class0:
        parts_X.append(np.stack(X_new_class0))
        parts_y.append(np.array(y_new_class0))
    if X_new_class1:
        parts_X.append(np.stack(X_new_class1))
        parts_y.append(np.array(y_new_class1))

    X_aug = np.concatenate(parts_X, axis=0)
    y_aug = np.concatenate(parts_y, axis=0)

    # Shuffle
    perm = np.random.permutation(len(X_aug))
    X_aug = X_aug[perm]
    y_aug = y_aug[perm]

    n_class0_after = int(np.sum(y_aug == 0))
    n_class1_after = int(np.sum(y_aug == 1))

    print(f"  [AUG] Class distribution after:  class0={n_class0_after}, class1={n_class1_after}")
    print(f"  [AUG] Total samples: {n_total} -> {len(X_aug)}")

    return X_aug, y_aug
