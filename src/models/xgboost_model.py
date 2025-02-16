from src.models.base_model import BaseModel
from src.config import Config
import xgboost as xgb
import numpy as np


class XGBoostModel(BaseModel):
    def __init__(self, random_state=Config.SEED):
        super().__init__()
        self.model = xgb.XGBClassifier()
        self.base_params = {
            "random_state": random_state,
            "objective": "binary:logistic",
        }
        self.learnable_params = {
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
            "n_estimators": {"type": "int", "low": 50, "high": 300},
        }
        self.params = {**self.base_params}

    def fit(self, X, y):
        self.model.set_params(**self.params)
        self.model.fit(np.array(X), np.array(y))
