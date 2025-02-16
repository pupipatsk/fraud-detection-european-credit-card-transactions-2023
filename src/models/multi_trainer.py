from typing import Dict
import os
import time
import joblib
from .base_model import BaseModel
from .single_trainer import SingleTrainer


class MultiTrainer:
    def __init__(
        self,
        df_train,
        df_test,
        target: str,
        models: Dict[str, BaseModel],
        main_metric: str,
        verbose=True,
        output_dir: str = None,
    ):
        self.df_train = df_train
        self.df_test = df_test
        self.target = target
        self.models: Dict[str, BaseModel] = models  # {model_name: model_instance}
        self.trained_models: Dict[str, BaseModel] = {}
        self.main_metric = main_metric
        self.verbose = verbose
        self.output_dir = output_dir

    @staticmethod
    def _save_model(model: BaseModel, output_dir, file_format="pkl", verbose=True):
        time_now = time.strftime("%Y-%m-%d-%H%M")
        model_name = model.__class__.__name__
        file_name = f"{time_now}-{model_name}"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{file_name}.{file_format}")
        joblib.dump(model, file_path)
        if verbose:
            print(f"Model saved to {file_path}")

    def train_all_models(self, tune_params: bool = False):
        """Train and tune all models"""
        for model_name, model in self.models.items():
            if self.verbose:
                print(f"Training {model_name}...")
                start_time = time.time()

            single_trainer = SingleTrainer(model, self.main_metric)
            single_trainer.train(self.df_train, self.target, tune_params)
            self.trained_models[model_name] = single_trainer.model
            # Save model
            if self.output_dir:
                self._save_model(
                    single_trainer.model, self.output_dir, verbose=self.verbose
                )

            if self.verbose:
                print(
                    f"Training time {model_name}: {time.time() - start_time:.2f} seconds."
                )

    def evaluate_all_models(self) -> Dict[str, Dict]:
        """Evaluate all trained models on both train and test sets."""
        results = {"train": {}, "test": {}}

        for dataset_name, df in [("train", self.df_train), ("test", self.df_test)]:
            X = df.drop(columns=[self.target]).values
            y = df[self.target].values

            for model_name, model in self.trained_models.items():
                res = model.evaluate(X, y)
                score = res.get(self.main_metric, None)  # Avoid KeyError
                results[dataset_name][model_name] = res
                if self.verbose:
                    print(
                        f"{dataset_name.upper()} | {model_name} {self.main_metric}: {score:.4f}"
                        if score is not None
                        else f"{dataset_name.upper()} | {model_name}: Metric not available"
                    )

        return results
