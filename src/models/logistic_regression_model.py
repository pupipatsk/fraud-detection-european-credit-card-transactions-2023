from sklearn.linear_model import LogisticRegression
from src.config import Config
from src.models.base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, random_state=Config.SEED):
        super().__init__()
        self.model = LogisticRegression()
        self.base_params = {"random_state": random_state}
        self.learnable_params = {
            "C": {"type": "loguniform", "low": 0.01, "high": 10}  # Regularization strength
        }
        self.params = {**self.base_params}

    def fit(self, X, y):
        self.model.set_params(**self.params)
        self.model.fit(X, y)