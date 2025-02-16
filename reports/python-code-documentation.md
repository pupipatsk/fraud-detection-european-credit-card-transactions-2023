# Python Code Documentation

## Configuration Module (`config.py`)

This module contains configuration settings for reproducibility and visualization customization.

### Class: `Config`

#### Attributes

- `SEED (int)`: Default seed for reproducibility.
- `PLOT_CONFIG (Dict[str, object])`: Dictionary containing Matplotlib configuration settings.

#### Methods

- `apply_plot_config() -> None`: Applies the Matplotlib configuration settings.

## Data Processing Module (`data_processor.py`)

### Class: `DataProcessor`

Processes and prepares datasets for training.

#### Attributes

- `random_state (int)`: Seed for reproducibility.

#### Methods

- `initial_train_test_split(df_dataset: pd.DataFrame, test_size: float = 0.10) -> Tuple[pd.DataFrame, pd.DataFrame]`: Splits dataset into training and test sets.
- `cut_outliers(df: pd.DataFrame) -> pd.DataFrame`: Removes outliers based on quantile thresholds.
- `normalize(df_train: pd.DataFrame, df_test: pd.DataFrame, num_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]`: Normalizes numerical features using StandardScaler.
- `process(df_dataset: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]`: Processes dataset including outlier removal, normalization, and splitting.

## Model Module (`base_model.py`)

### Class: `BaseModel`

A base class for machine learning models with training, prediction, and evaluation functionalities.

#### Attributes

- `model (Optional[Any])`: Machine learning model instance.
- `params (Dict[str, Any])`: Hyperparameters.
- `metrics (Dict[str, callable])`: Evaluation metrics.

#### Methods

- `fit(X: np.ndarray, y: np.ndarray) -> None`: Trains the model.
- `predict(X: np.ndarray) -> np.ndarray`: Generates predictions.
- `evaluate(X: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]`: Evaluates model performance.

## Logistic Regression Model (`logistic_regression_model.py`)

### Class: `LogisticRegressionModel`

A logistic regression model wrapper extending `BaseModel`.

#### Attributes

- `model (LogisticRegression)`: Logistic regression model instance.
- `params (Dict[str, Any])`: Final model parameters.

#### Methods

- `fit(X: np.ndarray, y: np.ndarray) -> None`: Trains the logistic regression model.

## XGBoost Model (`xgboost_model.py`)

### Class: `XGBoostModel`

A wrapper for XGBoost classifier extending `BaseModel`.

#### Attributes

- `model (xgb.XGBClassifier)`: XGBoost classifier instance.
- `params (Dict[str, Any])`: Hyperparameters.

#### Methods

- `fit(X: np.ndarray, y: np.ndarray) -> None`: Trains the XGBoost model.

## Training Module (`single_trainer.py`)

### Class: `SingleTrainer`

Handles training and hyperparameter tuning for a single model.

#### Methods

- `train(df_train: pd.DataFrame, target: str, tune_params: bool = False) -> None`: Trains the model.
- `tune_hyperparameters(df_train: pd.DataFrame, n_trials: int = 3) -> Dict[str, Any]`: Optimizes hyperparameters using Optuna.

## Multi-Model Trainer (`multi_trainer.py`)

### Class: `MultiTrainer`

Trains and evaluates multiple models.

#### Methods

- `train_all_models(tune_params: bool = False) -> None`: Trains all models.
- `evaluate_all_models() -> Dict[str, Dict[str, Dict[str, float]]]`: Evaluates trained models.

## Data Loader (`dataloader.py`)

### Functions

- `optimize_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame`: Optimizes DataFrame memory usage.
- `load_data(file_path: str) -> pd.DataFrame`: Loads and optimizes a dataset.
- `save_data(df: pd.DataFrame, file_name: str, file_directory: str, file_format: str = "parquet") -> None`: Saves dataset as CSV or Parquet.

## CSV to Parquet Conversion (`csv_to_parquet.py`)

### Function

- `csv_to_parquet_pyarrow(csv_file_path: str, parquet_file_path: str) -> None`: Converts CSV to Parquet using PyArrow.
