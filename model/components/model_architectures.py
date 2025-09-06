from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Optional imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class BaseModelArchitecture(ABC):
    """Abstract base class for model architectures"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_trained = False
        self.model = None
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model architecture
        
        Args:
            X: Feature matrix
            y: Binary labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Generate probability predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probabilities for positive class
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        probabilities = self.predict_proba(X)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        return {
            'accuracy': accuracy_score(y, binary_predictions),
            'precision': precision_score(y, binary_predictions, zero_division=0),
            'recall': recall_score(y, binary_predictions, zero_division=0),
            'f1': f1_score(y, binary_predictions, zero_division=0),
            'auc_roc': roc_auc_score(y, probabilities)
        }
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save trained model"""
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load trained model"""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available"""
        return None


class LogisticRegressionArchitecture(BaseModelArchitecture):
    """Logistic Regression model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = LogisticRegression(
            C=config.get('C', 1.0),
            max_iter=config.get('max_iter', 1000),
            random_state=config.get('random_state', 42),
            class_weight=config.get('class_weight', 'balanced'),
            solver=config.get('solver', 'liblinear')
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train logistic regression"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str) -> None:
        """Save model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.is_trained = data['is_trained']
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get logistic regression coefficients"""
        if not self.is_trained:
            return None
        return np.abs(self.model.coef_[0])


class SVMArchitecture(BaseModelArchitecture):
    """Support Vector Machine architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = SVC(
            C=config.get('C', 1.0),
            kernel=config.get('kernel', 'rbf'),
            probability=True,  # Required for predict_proba
            random_state=config.get('random_state', 42),
            class_weight=config.get('class_weight', 'balanced')
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train SVM"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str) -> None:
        """Save model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.is_trained = data['is_trained']


class RandomForestArchitecture(BaseModelArchitecture):
    """Random Forest model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', None),
            random_state=config.get('random_state', 42),
            class_weight=config.get('class_weight', 'balanced'),
            n_jobs=config.get('n_jobs', -1)
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train random forest"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str) -> None:
        """Save model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.is_trained = data['is_trained']
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get random forest feature importance"""
        if not self.is_trained:
            return None
        return self.model.feature_importances_


class LightGBMArchitecture(BaseModelArchitecture):
    """LightGBM model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm required for LightGBM architecture")
            
        super().__init__(config)
        
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.1),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': -1,
            'random_state': config.get('random_state', 42),
            'is_unbalance': config.get('is_unbalance', True)
        }
        
        self.num_boost_round = config.get('num_boost_round', 100)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train LightGBM"""
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def save(self, filepath: str) -> None:
        """Save LightGBM model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # Save LightGBM model in its native format
        lgb_filepath = filepath.replace('.pkl', '_lgb.txt')
        self.model.save_model(lgb_filepath)
        
        # Save metadata
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'num_boost_round': self.num_boost_round,
                'config': self.config,
                'is_trained': self.is_trained,
                'lgb_filepath': lgb_filepath
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load LightGBM model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            self.num_boost_round = data['num_boost_round']
            self.config = data['config']
            self.is_trained = data['is_trained']
            lgb_filepath = data['lgb_filepath']
        
        self.model = lgb.Booster(model_file=lgb_filepath)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get LightGBM feature importance"""
        if not self.is_trained:
            return None
        return self.model.feature_importance(importance_type='gain')


class XGBoostArchitecture(BaseModelArchitecture):
    """XGBoost model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost required for XGBoost architecture")
            
        super().__init__(config)
        
        self.model = xgb.XGBClassifier(
            n_estimators=config.get('n_estimators', 100),
            learning_rate=config.get('learning_rate', 0.1),
            max_depth=config.get('max_depth', 6),
            random_state=config.get('random_state', 42),
            scale_pos_weight=config.get('scale_pos_weight', 1),
            eval_metric='logloss'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train XGBoost"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str) -> None:
        """Save model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'config': self.config,
                'is_trained': self.is_trained
            }, f)
    
    def load(self, filepath: str) -> None:
        """Load model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.config = data['config']
            self.is_trained = data['is_trained']
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get XGBoost feature importance"""
        if not self.is_trained:
            return None
        return self.model.feature_importances_


def get_model_architecture(architecture_type: str, config: Dict[str, Any]) -> BaseModelArchitecture:
    """Factory function to create model architectures"""
    architectures = {
        'logistic': LogisticRegressionArchitecture,
        'svm': SVMArchitecture,
        'random_forest': RandomForestArchitecture,
        'lightgbm': LightGBMArchitecture,
        'xgboost': XGBoostArchitecture
    }
    
    if architecture_type not in architectures:
        raise ValueError(f"Unknown architecture type: {architecture_type}. Available: {list(architectures.keys())}")
    
    return architectures[architecture_type](config)