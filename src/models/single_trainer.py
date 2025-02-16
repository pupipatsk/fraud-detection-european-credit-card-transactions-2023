from src.models.base_model import BaseModel
from src.config import Config
import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold


class SingleTrainer:
    def __init__(self, model: BaseModel, main_metric: str):
        self.model = model  # Custom model instance (e.g., LogisticRegressionModel)
        self.target = ""
        self.main_metric = main_metric

    def train(self, df_train, target, tune_params=False):
        self.target = target  # update target
        X = df_train.drop(columns=[self.target]).values
        y = df_train[self.target].values

        if tune_params:
            best_params = self.tune_hyperparameters(df_train)
            self.model.model.set_params(**best_params)
        self.model.fit(X, y)

    def _retrieve_search_space(self, model, trial):
        """Retrieve and suggest hyperparameter search space for Optuna tuning."""
        trial_params = {}
        for param, search_space in model.learnable_params.items():
            if search_space["type"] == "int":
                trial_params[param] = trial.suggest_int(
                    param, search_space["low"], search_space["high"]
                )
            elif search_space["type"] == "loguniform":
                trial_params[param] = trial.suggest_loguniform(
                    param, search_space["low"], search_space["high"]
                )
        return trial_params

    def _cross_validate(self, X, y, params):
        """Perform cross-validation and return fold scores"""
        n_splits = 5
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=Config.SEED)
        fold_scores = []
        for train_index, val_index in cv.split(X, y):
            x_tr, x_val = X[train_index], X[val_index]
            y_tr, y_val = y[train_index], y[val_index]

            model = type(self.model)()  # Re-instantiate model
            model.model.set_params(**params)
            model.fit(x_tr, y_tr)
            score = model.evaluate(x_val, y_val)[self.main_metric]
            fold_scores.append(score)
        return fold_scores

    def objective(self, trial, df_train):
        """Optuna objective function for hyperparameter tuning"""
        # Search space
        trial_params = self._retrieve_search_space(self.model, trial)
        # Cross-validation
        X = df_train.drop(columns=[self.target]).values
        y = df_train[self.target].values
        fold_scores = self._cross_validate(X, y, trial_params)
        return np.mean(fold_scores)

    def tune_hyperparameters(self, df_train, n_trials=3) -> dict:
        """Optimize hyperparameters using Optuna"""
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, df_train), n_trials=n_trials)
        return study.best_params
