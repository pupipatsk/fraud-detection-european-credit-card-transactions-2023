from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BaseModel:
    def __init__(self):
        self.model = None
        self.params = {}
        self.metrics = {
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "accuracy": accuracy_score,
            "roc_auc": roc_auc_score,
        }  # {metric_name: metric_function}

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y) -> dict:
        y_pred = self.predict(X)
        results = {}
        for metric_name, metric_fn in self.metrics.items():
            try:
                if metric_name == "roc_auc":
                    y_prob = self.model.predict_proba(X)[
                        :, 1
                    ]  # Get probabilities for the positive class (1)
                    results[metric_name] = metric_fn(y, y_prob)
                else:
                    results[metric_name] = metric_fn(y, y_pred)
            except ValueError:
                results[metric_name] = None
        return results
