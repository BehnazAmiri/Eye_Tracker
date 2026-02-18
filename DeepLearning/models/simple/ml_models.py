"""
Machine Learning Models for Eye-Tracking Classification
=======================================================
Traditional ML models: Random Forest, SVM, Logistic Regression, XGBoost
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Optional: XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def create_ml_model(model_type, class_weights=None, **kwargs):
    """
    Factory function to create machine learning models.
    
    Args:
        model_type: 'rf', 'svm', 'logreg', or 'xgboost'
        class_weights: Tuple of (weight_class_0, weight_class_1) for handling imbalance
        **kwargs: Additional model-specific parameters
    
    Returns:
        Sklearn-compatible classifier
    
    Examples:
        >>> clf = create_ml_model('rf', n_estimators=100)
        >>> clf = create_ml_model('svm', C=1.0, kernel='rbf')
        >>> clf = create_ml_model('xgboost', n_estimators=100, max_depth=6)
    """
    model_type = model_type.lower()
    
    # Calculate scale_pos_weight for XGBoost if class_weights provided
    scale_pos_weight = None
    if class_weights is not None:
        # XGBoost uses scale_pos_weight = weight_class_1 / weight_class_0
        scale_pos_weight = class_weights[1] / class_weights[0]
    
    if model_type == 'rf':
        # Random Forest - Optimized for small datasets
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),  # More trees for stability
            max_depth=kwargs.get('max_depth', 8),  # Prevent overfitting
            min_samples_split=kwargs.get('min_samples_split', 10),  # Larger splits
            min_samples_leaf=kwargs.get('min_samples_leaf', 4),  # Larger leaves
            max_features=kwargs.get('max_features', 'sqrt'),
            random_state=kwargs.get('random_state', 42),
            class_weight='balanced',  # Handle imbalance
            n_jobs=kwargs.get('n_jobs', -1)
        )
    
    elif model_type == 'svm':
        # Support Vector Machine
        return SVC(
            C=kwargs.get('C', 1.0),
            kernel=kwargs.get('kernel', 'rbf'),
            gamma=kwargs.get('gamma', 'scale'),
            probability=True,  # Enable probability estimates
            random_state=kwargs.get('random_state', 42),
            class_weight='balanced'  # Handle imbalance
        )
    
    elif model_type == 'logreg':
        # Logistic Regression
        return LogisticRegression(
            C=kwargs.get('C', 1.0),
            penalty=kwargs.get('penalty', 'l2'),
            solver=kwargs.get('solver', 'lbfgs'),
            max_iter=kwargs.get('max_iter', 1000),
            random_state=kwargs.get('random_state', 42),
            class_weight='balanced'  # Handle imbalance
        )
    
    elif model_type == 'xgboost':
        # XGBoost - Optimized for small datasets with regularization
        if not HAS_XGBOOST:
            raise ValueError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )
        
        return xgb.XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 150),  # More rounds
            max_depth=kwargs.get('max_depth', 4),  # Shallower for generalization
            learning_rate=kwargs.get('learning_rate', 0.05),  # Lower LR for stability
            subsample=kwargs.get('subsample', 0.8),
            colsample_bytree=kwargs.get('colsample_bytree', 0.8),
            gamma=kwargs.get('gamma', 1.0),  # Regularization
            reg_alpha=kwargs.get('reg_alpha', 0.5),  # L1 regularization
            reg_lambda=kwargs.get('reg_lambda', 2.0),  # L2 regularization
            min_child_weight=kwargs.get('min_child_weight', 3),  # Prevent overfitting
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1.0,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=kwargs.get('random_state', 42)
        )
    
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from: rf, svm, logreg, xgboost"
        )


def get_model_info(model_type):
    """
    Get information about a specific ML model.
    
    Args:
        model_type: 'rf', 'svm', 'logreg', or 'xgboost'
    
    Returns:
        dict: Model information
    """
    info = {
        'rf': {
            'name': 'Random Forest',
            'type': 'Ensemble - Decision Trees',
            'strengths': 'Robust to overfitting, handles non-linear relationships, feature importance',
            'weaknesses': 'Can be slow on large datasets, not good for extrapolation',
            'best_for': 'Complex feature interactions, interpretability'
        },
        'svm': {
            'name': 'Support Vector Machine',
            'type': 'Kernel-based',
            'strengths': 'Effective in high dimensions, works well with clear margin of separation',
            'weaknesses': 'Slow on large datasets, sensitive to feature scaling',
            'best_for': 'High-dimensional data, binary classification'
        },
        'logreg': {
            'name': 'Logistic Regression',
            'type': 'Linear Model',
            'strengths': 'Fast, interpretable, works well with linearly separable data',
            'weaknesses': 'Assumes linear relationship, may underfit complex data',
            'best_for': 'Baseline model, interpretable predictions'
        },
        'xgboost': {
            'name': 'XGBoost',
            'type': 'Ensemble - Gradient Boosting',
            'strengths': 'State-of-the-art performance, handles missing values, regularization',
            'weaknesses': 'Requires careful tuning, can overfit with wrong parameters',
            'best_for': 'Kaggle-like competitions, best raw performance'
        }
    }
    
    return info.get(model_type.lower(), {'name': 'Unknown', 'type': 'Unknown'})


__all__ = ['create_ml_model', 'get_model_info', 'HAS_XGBOOST']
