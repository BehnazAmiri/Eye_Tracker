"""
Simple Trainer for Binary Classification
========================================
Train and evaluate deep learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class SimpleTrainer:
    """
    Simple trainer for binary classification.
    
    Args:
        model: PyTorch model
        device: Device to train on ('cuda' or 'cpu')
        learning_rate: Learning rate (default: 0.001)
        weight_decay: L2 regularization (default: 0.01)
        class_weights: Tuple of (weight_class_0, weight_class_1) for handling class imbalance (optional)
    """
    
    def __init__(self, model, device='cuda', learning_rate=0.001, weight_decay=0.01, class_weights=None):
        self.model = model.to(device)
        self.device = device
        
        # Always use BCEWithLogitsLoss for better numerical stability
        # (combines Sigmoid + BCELoss in one more stable operation)
        if class_weights is not None:
            # pos_weight is the weight for the POSITIVE class (class 1)
            # Higher pos_weight gives more importance to class 1
            # Formula: pos_weight = weight_class_1 / weight_class_0
            pos_weight = torch.tensor([class_weights[1] / class_weights[0]], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"    Using weighted BCEWithLogitsLoss: pos_weight={pos_weight[0]:.3f}")
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Add learning rate scheduler (reduces LR when validation loss plateaus)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch on GPU (if available).
        
        Note: All computations happen on self.device (cuda/GPU if available).
        The .cpu() calls later are only for transferring results to RAM for metrics calculation.
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Move data to GPU
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Model outputs logits (no sigmoid) - computed on GPU
            outputs = self.model(X_batch)
            
            # BCEWithLogitsLoss expects logits (will apply sigmoid internally) - computed on GPU
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass on GPU
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, loader, y_true):
        """
        Evaluate model on a dataset.
        
        Model inference runs on GPU (self.device).
        Results are transferred to CPU only for numpy/sklearn metrics calculation.
        
        Args:
            loader: Data loader
            y_true: True labels
        
        Returns:
            dict: Metrics
        """
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)  # Data on GPU
                logits = self.model(X_batch)  # Inference on GPU
                # Apply sigmoid to convert logits to probabilities (on GPU)
                # Then transfer to CPU for numpy operations
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                all_probs.extend(probs)
        
        all_probs = np.array(all_probs)
        y_pred = (all_probs >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, all_probs),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'predictions': all_probs.tolist(),
            'predicted_labels': y_pred.tolist(),
            'true_labels': y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
        }
        
        return metrics
    
    def fit(self, train_loader, test_loader, y_train, y_test, n_epochs=100, verbose=True, patience=10):
        """
        Train the model with early stopping.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader (used as validation for early stopping)
            y_train: Training labels
            y_test: Test labels
            n_epochs: Maximum number of epochs
            verbose: Print progress
            patience: Early stopping patience (epochs without improvement)
        
        Returns:
            dict: Training results
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"TRAINING CONFIGURATION")
            print(f"{'='*70}")
            print(f"Device: {self.device}")
            print(f"Max Epochs: {n_epochs} | Early Stopping Patience: {patience}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f} | Weight Decay: {self.optimizer.param_groups[0]['weight_decay']:.6f}")
            print(f"Train Samples: {len(train_loader.dataset)} | Test Samples: {len(test_loader.dataset)}")
            print(f"Train Batches: {len(train_loader)} | Test Batches: {len(test_loader)}")
            print(f"Batch Size: Train={train_loader.batch_size}, Test={test_loader.batch_size}")
            print(f"{'='*70}")
            print(f"TRAINING PROGRESS (Loss and Accuracy per Epoch)")
            print(f"{'='*70}")
        
        train_accuracies = []
        val_accuracies = []
        val_losses = []
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None
        current_lr = self.optimizer.param_groups[0]['lr']
        
        # ======================================================================
        # MAIN TRAINING LOOP (All computations on GPU if device='cuda')
        # ======================================================================
        for epoch in range(n_epochs):
            # Training phase (on GPU)
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Evaluation phase (inference on GPU, metrics on CPU)
            self.model.eval()
            train_preds = []
            train_labels = []
            val_preds = []
            val_labels = []
            val_loss = 0
            
            with torch.no_grad():
                # Training accuracy (inference on GPU)
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)  # Data to GPU
                    logits = self.model(X_batch)  # Inference on GPU
                    # Transfer results to CPU for sklearn metrics
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs >= 0.5).astype(int)
                    train_preds.extend(preds)
                    train_labels.extend(y_batch.numpy())
                
                # Validation loss and accuracy (inference on GPU)
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)  # Data to GPU
                    y_true_labels = y_batch.numpy()  # Save labels before moving to GPU
                    y_batch = y_batch.to(self.device).unsqueeze(1)  # Labels to GPU
                    logits = self.model(X_batch)  # Inference on GPU
                    loss = self.criterion(logits, y_batch)  # Loss computation on GPU
                    val_loss += loss.item()
                    
                    # Transfer results to CPU for sklearn metrics
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs >= 0.5).astype(int)
                    val_preds.extend(preds)
                    val_labels.extend(y_true_labels)
            
            val_loss = val_loss / len(test_loader)
            val_losses.append(val_loss)
            train_acc = accuracy_score(train_labels, train_preds)
            val_acc = accuracy_score(val_labels, val_preds)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Update learning rate based on validation loss
            old_lr = current_lr
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                # Save best model weights
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                if verbose:
                    print(f"  Epoch {epoch+1:3d}/{n_epochs}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f} | LR={current_lr:.6f} [OK] BEST")
            else:
                patience_counter += 1
                if verbose:
                    print(f"  Epoch {epoch+1:3d}/{n_epochs}: TrainLoss={train_loss:.4f}, TrainAcc={train_acc:.4f} | ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f} | LR={current_lr:.6f} [Patience: {patience_counter}/{patience}]")
                
                # Print LR change if it happened
                if current_lr != old_lr and verbose:
                    print(f"    -> Learning rate reduced: {old_lr:.6f} -> {current_lr:.6f}")
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"\n  ? Early stopping at epoch {epoch+1}")
                        print(f"  [OK] Best epoch: {best_epoch+1} (val_loss={best_val_loss:.4f}, val_acc={val_accuracies[best_epoch]:.4f})")
                    break
        
        # Restore best model weights
        if best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
            if verbose:
                print(f"\n{'='*70}")
                print(f"TRAINING SUMMARY")
                print(f"{'='*70}")
                print(f"[OK] Training completed successfully!")
                print(f"  Total Epochs Trained: {epoch+1}/{n_epochs}")
                print(f"  Best Epoch: {best_epoch+1}")
                print(f"  Best Validation Loss: {best_val_loss:.4f}")
                print(f"  Best Validation Accuracy: {val_accuracies[best_epoch]:.4f}")
                print(f"  Final Train Accuracy: {train_accuracies[best_epoch]:.4f}")
                print(f"[OK] Best model weights restored")
                print(f"{'='*70}")
        
        # Final evaluation
        from torch.utils.data import DataLoader
        eval_train_loader = DataLoader(
            train_loader.dataset, 
            batch_size=train_loader.batch_size, 
            shuffle=False
        )
        eval_test_loader = DataLoader(
            test_loader.dataset, 
            batch_size=test_loader.batch_size, 
            shuffle=False
        )
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FINAL EVALUATION (Best Model from Epoch {best_epoch+1})")
            print(f"{'='*70}")
        
        train_metrics = self.evaluate(eval_train_loader, y_train)
        test_metrics = self.evaluate(eval_test_loader, y_test)
        
        if verbose:
            self._print_metrics("Train", train_metrics)
            self._print_metrics("Test", test_metrics)
            print(f"\n{'?'*70}")
            print(f"Confusion Matrix (Test Set)")
            print(f"{'?'*70}")
            cm = np.array(test_metrics['confusion_matrix'])
            print(f"              Predicted")
            print(f"              Class 0  Class 1")
            print(f"Actual Class 0: {cm[0][0]:4d}     {cm[0][1]:4d}    (TN: {cm[0][0]}, FP: {cm[0][1]})")
            print(f"       Class 1: {cm[1][0]:4d}     {cm[1][1]:4d}    (FN: {cm[1][0]}, TP: {cm[1][1]})")
            print(f"{'?'*70}")
            print(f"TN=True Negative, FP=False Positive, FN=False Negative, TP=True Positive")
            print(f"{'='*70}\n")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'train_losses': self.train_losses,
            'val_losses': val_losses,
            'best_epoch': best_epoch + 1,
            'history': {
                'loss': self.train_losses,
                'val_loss': val_losses,
                'accuracy': train_accuracies,
                'val_accuracy': val_accuracies
            }
        }
    
    def _print_metrics(self, split_name, metrics):
        """Print metrics nicely with percentages."""
        print(f"\n{'?'*70}")
        print(f"{split_name} Set Performance")
        print(f"{'?'*70}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['f1']:.4f}  ({metrics['f1']*100:.2f}%)")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}  ({metrics['roc_auc']*100:.2f}%)")
        print(f"{'?'*70}")
