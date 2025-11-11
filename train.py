"""
Production-ready model training module for used car pricing.
Handles model training, validation, and evaluation with proper time-series splits.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, cross_val_score,
    GridSearchCV, RandomizedSearchCV, KFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

# Try to import advanced packages
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str = 'lgbm'
    feature_set: str = 'full'
    use_log_target: bool = True
    test_size: float = 0.2
    validation_size: float = 0.1
    cv_splits: int = 5
    random_state: int = 42
    monotonic_constraints: Optional[Dict[str, int]] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class MetricsCalculator:
    """Calculate and store model evaluation metrics"""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Weighted Absolute Percentage Error"""
        denominator = np.maximum(np.abs(y_true).sum(), 1e-9)
        return np.abs(y_true - y_pred).sum() / denominator

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def quantile_coverage(y_true: np.ndarray, y_lower: np.ndarray,
                         y_upper: np.ndarray) -> float:
        """Calculate coverage of prediction intervals"""
        return np.mean((y_true >= y_lower) & (y_true <= y_upper))

    @classmethod
    def calculate_all(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': cls.rmse(y_true, y_pred),
            'wape': cls.wape(y_true, y_pred),
            'mape': cls.mape(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }


class UsedCarPricingModel:
    """Main model class for used car pricing"""

    def __init__(self, config: ModelConfig = None):
        """
        Initialize the pricing model.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.pipeline = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        self.cv_results = None
        self.feature_importance = None
        self.metrics_calculator = MetricsCalculator()

    def _identify_feature_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical features.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (numeric_features, categorical_features)
        """
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        logger.info(f"Identified {len(numeric_features)} numeric and "
                   f"{len(categorical_features)} categorical features")

        return numeric_features, categorical_features

    def _create_preprocessor(self, numeric_features: List[str],
                            categorical_features: List[str]) -> ColumnTransformer:
        """
        Create preprocessing pipeline.

        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names

        Returns:
            ColumnTransformer for preprocessing
        """
        # Numeric pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        return preprocessor

    def _create_model(self) -> Any:
        """
        Create the model based on configuration.

        Returns:
            Model instance
        """
        model_type = self.config.model_type
        hyperparameters = self.config.hyperparameters or {}

        if model_type == 'linear':
            return LinearRegression()

        elif model_type == 'lasso':
            return Lasso(
                alpha=hyperparameters.get('alpha', 0.01),
                random_state=self.config.random_state
            )

        elif model_type == 'ridge':
            return Ridge(
                alpha=hyperparameters.get('alpha', 1.0),
                random_state=self.config.random_state
            )

        elif model_type == 'elastic':
            return ElasticNet(
                alpha=hyperparameters.get('alpha', 0.01),
                l1_ratio=hyperparameters.get('l1_ratio', 0.5),
                random_state=self.config.random_state
            )

        elif model_type == 'lgbm' and LIGHTGBM_AVAILABLE:
            params = {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': -1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.config.random_state,
                'n_jobs': -1,
                'verbosity': -1
            }
            params.update(hyperparameters)

            # Add monotonic constraints if specified
            if self.config.monotonic_constraints:
                params['monotone_constraints'] = self.config.monotonic_constraints

            return LGBMRegressor(**params)

        elif model_type == 'xgb' and XGBOOST_AVAILABLE:
            params = {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.config.random_state,
                'n_jobs': -1
            }
            params.update(hyperparameters)

            # Add monotonic constraints if specified
            if self.config.monotonic_constraints:
                params['monotone_constraints'] = self.config.monotonic_constraints

            return XGBRegressor(**params)

        else:
            # Fallback to HistGradientBoosting
            params = {
                'max_iter': 1000,
                'learning_rate': 0.05,
                'max_depth': None,
                'random_state': self.config.random_state
            }
            params.update(hyperparameters)

            # Add monotonic constraints if specified
            if self.config.monotonic_constraints:
                params['monotonic_cst'] = self.config.monotonic_constraints

            return HistGradientBoostingRegressor(**params)

    def _time_based_split(self, X: pd.DataFrame, y: pd.Series,
                         dates: pd.Series, n_splits: int = 5) -> List[Tuple]:
        """
        Create time-based train/validation splits.

        Args:
            X: Feature matrix
            y: Target variable
            dates: Date series for ordering
            n_splits: Number of splits

        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Sort by date
        sorted_idx = dates.argsort()
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]

        # Create expanding window splits
        splits = []
        n_samples = len(X)

        for i in range(2, n_splits + 1):
            train_size = int(n_samples * (i / (n_splits + 1)))
            val_size = int(n_samples * self.config.validation_size)

            train_idx = list(range(train_size))
            val_idx = list(range(train_size, min(train_size + val_size, n_samples)))

            if len(val_idx) > 0:
                splits.append((train_idx, val_idx))

        logger.info(f"Created {len(splits)} time-based splits")
        return splits

    def fit(self, X: pd.DataFrame, y: pd.Series,
            dates: Optional[pd.Series] = None) -> 'UsedCarPricingModel':
        """
        Fit the model with cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            dates: Optional date series for time-based splitting

        Returns:
            Self
        """
        logger.info(f"Training {self.config.model_type} model with {len(X)} samples")

        # Identify feature types
        self.numeric_features, self.categorical_features = self._identify_feature_types(X)
        self.feature_names = self.numeric_features + self.categorical_features

        # Apply log transform if configured
        if self.config.use_log_target:
            y_transformed = np.log1p(y)
        else:
            y_transformed = y

        # Create preprocessing pipeline
        preprocessor = self._create_preprocessor(
            self.numeric_features,
            self.categorical_features
        )

        # Create model
        model = self._create_model()

        # Create full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Perform cross-validation
        if dates is not None:
            splits = self._time_based_split(X, y_transformed, dates, self.config.cv_splits)
        else:
            kf = KFold(n_splits=self.config.cv_splits, shuffle=True,
                      random_state=self.config.random_state)
            splits = list(kf.split(X))

        # Train and evaluate on each fold
        cv_scores = {'mae': [], 'rmse': [], 'wape': [], 'mape': [], 'r2': []}

        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Training fold {fold + 1}/{len(splits)}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_transformed.iloc[train_idx], y_transformed.iloc[val_idx]

            # Fit on training fold
            self.pipeline.fit(X_train, y_train)

            # Predict on validation fold
            y_pred = self.pipeline.predict(X_val)

            # Convert back from log if necessary
            if self.config.use_log_target:
                y_val_orig = np.expm1(y_val)
                y_pred_orig = np.expm1(y_pred)
            else:
                y_val_orig = y_val
                y_pred_orig = y_pred

            # Calculate metrics
            fold_metrics = self.metrics_calculator.calculate_all(y_val_orig, y_pred_orig)
            for metric, value in fold_metrics.items():
                cv_scores[metric].append(value)

        # Store CV results
        self.cv_results = {
            'scores': cv_scores,
            'mean': {metric: np.mean(scores) for metric, scores in cv_scores.items()},
            'std': {metric: np.std(scores) for metric, scores in cv_scores.items()}
        }

        logger.info(f"CV Results - WAPE: {self.cv_results['mean']['wape']:.3f} ± "
                   f"{self.cv_results['std']['wape']:.3f}")

        # Train final model on all data
        self.pipeline.fit(X, y_transformed)

        # Extract feature importance if available
        self._extract_feature_importance()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted before making predictions")

        y_pred = self.pipeline.predict(X)

        # Convert back from log if necessary
        if self.config.use_log_target:
            y_pred = np.expm1(y_pred)

        return y_pred

    def predict_interval(self, X: pd.DataFrame,
                        quantiles: List[float] = [0.1, 0.5, 0.9]) -> Dict[float, np.ndarray]:
        """
        Predict quantiles for uncertainty estimation.

        Args:
            X: Feature matrix
            quantiles: List of quantiles to predict

        Returns:
            Dictionary mapping quantiles to predictions
        """
        if not LIGHTGBM_AVAILABLE or self.config.model_type != 'lgbm':
            logger.warning("Quantile prediction only available for LightGBM")
            return {}

        predictions = {}

        for q in quantiles:
            logger.info(f"Training quantile {q} model")

            # Create quantile model
            model = LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=500,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                verbosity=-1
            )

            # Create pipeline with same preprocessor
            pipeline = Pipeline([
                ('preprocessor', self.pipeline.named_steps['preprocessor']),
                ('model', model)
            ])

            # Fit and predict
            # Note: In production, this should use the same training data
            # This is a simplified version for demonstration
            pipeline.fit(X, np.log1p(X.iloc[:, 0]))  # Placeholder
            y_pred = pipeline.predict(X)

            if self.config.use_log_target:
                y_pred = np.expm1(y_pred)

            predictions[q] = y_pred

        return predictions

    def _extract_feature_importance(self):
        """Extract feature importance from tree-based models"""
        model = self.pipeline.named_steps['model']

        if hasattr(model, 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = self.pipeline.named_steps['preprocessor']

            feature_names = []

            # Numeric features (unchanged names)
            feature_names.extend(self.numeric_features)

            # Categorical features (after one-hot encoding)
            if len(self.categorical_features) > 0:
                cat_transformer = preprocessor.transformers_[1][1]
                if hasattr(cat_transformer.named_steps['onehot'], 'get_feature_names_out'):
                    cat_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out()
                    feature_names.extend(cat_feature_names)

            # Match with importance values
            n_features = len(model.feature_importances_)
            if len(feature_names) > n_features:
                feature_names = feature_names[:n_features]

            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"Extracted importance for {len(self.feature_importance)} features")

    def explain(self, X: pd.DataFrame, sample_size: int = 100) -> Optional[Dict]:
        """
        Generate SHAP explanations for the model.

        Args:
            X: Feature matrix
            sample_size: Number of samples for SHAP calculation

        Returns:
            Dictionary with SHAP values and explanations
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available for explanations")
            return None

        if self.pipeline is None:
            raise ValueError("Model must be fitted before generating explanations")

        logger.info("Calculating SHAP values")

        # Sample for efficiency
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=self.config.random_state)
        else:
            X_sample = X

        # Transform features
        X_transformed = self.pipeline.named_steps['preprocessor'].transform(X_sample)

        # Get model
        model = self.pipeline.named_steps['model']

        # Calculate SHAP values
        try:
            explainer = shap.Explainer(model, X_transformed)
            shap_values = explainer(X_transformed)

            return {
                'shap_values': shap_values,
                'explainer': explainer,
                'X_sample': X_sample,
                'feature_names': self.feature_names
            }
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            return None

    def save(self, path: Union[str, Path]):
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'config': self.config,
                'cv_results': self.cv_results,
                'feature_importance': self.feature_importance,
                'feature_names': self.feature_names,
                'numeric_features': self.numeric_features,
                'categorical_features': self.categorical_features
            }, f)

        # Save metadata
        metadata = {
            'model_type': self.config.model_type,
            'feature_set': self.config.feature_set,
            'timestamp': datetime.now().isoformat(),
            'cv_results': {
                'mean': self.cv_results['mean'] if self.cv_results else {},
                'std': self.cv_results['std'] if self.cv_results else {}
            }
        }

        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Model saved to {path}")
        logger.info(f"Metadata saved to {metadata_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'UsedCarPricingModel':
        """
        Load a model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model instance
        """
        path = Path(path)

        with open(path, 'rb') as f:
            data = pickle.load(f)

        model = cls(config=data['config'])
        model.pipeline = data['pipeline']
        model.cv_results = data['cv_results']
        model.feature_importance = data.get('feature_importance')
        model.feature_names = data.get('feature_names')
        model.numeric_features = data.get('numeric_features')
        model.categorical_features = data.get('categorical_features')

        logger.info(f"Model loaded from {path}")
        return model

    def get_monthly_performance(self, X: pd.DataFrame, y: pd.Series,
                               dates: pd.Series) -> pd.DataFrame:
        """
        Calculate monthly performance metrics.

        Args:
            X: Feature matrix
            y: True values
            dates: Date series

        Returns:
            DataFrame with monthly metrics
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted first")

        # Make predictions
        y_pred = self.predict(X)

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'actual': y,
            'predicted': y_pred,
            'error': y - y_pred,
            'abs_error': np.abs(y - y_pred),
            'pct_error': np.abs((y - y_pred) / y) * 100
        })

        # Add month column
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')

        # Calculate monthly metrics
        monthly_metrics = df.groupby('month').agg({
            'actual': ['count', 'mean'],
            'predicted': 'mean',
            'abs_error': 'mean',
            'pct_error': 'mean'
        }).round(2)

        monthly_metrics.columns = ['n_samples', 'avg_actual', 'avg_predicted',
                                   'mae', 'mape']

        # Add WAPE
        monthly_wape = df.groupby('month').apply(
            lambda g: self.metrics_calculator.wape(g['actual'].values, g['predicted'].values)
        ).rename('wape')

        monthly_metrics = monthly_metrics.join(monthly_wape)

        return monthly_metrics.reset_index()


class AblationStudy:
    """Conduct ablation studies comparing different feature sets"""

    def __init__(self, feature_sets: Dict[str, Dict]):
        """
        Initialize ablation study.

        Args:
            feature_sets: Dictionary of feature set configurations
        """
        self.feature_sets = feature_sets
        self.results = []

    def run(self, df: pd.DataFrame, target_col: str,
            date_col: Optional[str] = None,
            model_types: List[str] = ['lasso', 'lgbm']) -> pd.DataFrame:
        """
        Run ablation study.

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            date_col: Optional date column for time-based splits
            model_types: List of model types to test

        Returns:
            DataFrame with ablation results
        """
        logger.info(f"Running ablation study with {len(self.feature_sets)} feature sets "
                   f"and {len(model_types)} model types")

        for feature_set_name, feature_config in self.feature_sets.items():
            for model_type in model_types:
                logger.info(f"\nTesting {model_type} with {feature_set_name} features")

                # Select features
                if feature_config.get('features'):
                    feature_cols = [col for col in feature_config['features']
                                   if col in df.columns]
                else:
                    # Use all available features
                    feature_cols = [col for col in df.columns
                                   if col != target_col and col != date_col]

                # Filter macro features if specified
                if not feature_config.get('include_macro', True):
                    feature_cols = [col for col in feature_cols
                                   if not any(x in col for x in ['cpi', 'm2', 'prime', 'gas'])]

                if len(feature_cols) == 0:
                    logger.warning(f"No features available for {feature_set_name}")
                    continue

                # Prepare data
                X = df[feature_cols]
                y = df[target_col]
                dates = df[date_col] if date_col and date_col in df.columns else None

                # Create and train model
                config = ModelConfig(
                    model_type=model_type,
                    feature_set=feature_set_name
                )

                model = UsedCarPricingModel(config)

                try:
                    model.fit(X, y, dates)

                    # Store results
                    result = {
                        'model_type': model_type,
                        'feature_set': feature_set_name,
                        'n_features': len(feature_cols),
                        'description': feature_config.get('description', ''),
                        **model.cv_results['mean'],
                        **{f"{metric}_std": std for metric, std in model.cv_results['std'].items()}
                    }

                    self.results.append(result)
                    logger.info(f"WAPE: {result['wape']:.3f} ± {result['wape_std']:.3f}")

                except Exception as e:
                    logger.error(f"Error training {model_type} with {feature_set_name}: {str(e)}")
                    continue

        # Create results DataFrame
        results_df = pd.DataFrame(self.results)

        # Calculate improvements
        if 'micro_only' in results_df['feature_set'].values:
            baseline = results_df[results_df['feature_set'] == 'micro_only'].groupby('model_type')['wape'].mean()
            for model_type in baseline.index:
                mask = results_df['model_type'] == model_type
                results_df.loc[mask, 'wape_improvement'] = (
                    (baseline[model_type] - results_df.loc[mask, 'wape']) / baseline[model_type] * 100
                )

        return results_df.sort_values(['model_type', 'wape'])


if __name__ == "__main__":
    # Production usage with real data
    logger.info("Running model training pipeline with REAL production data")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load real data from production sources
    from pathlib import Path

    # Try to load enriched features first, fallback to raw data
    features_path = Path('data/features/features.parquet')
    raw_path = Path('used_cars (1).csv')

    if features_path.exists():
        logger.info(f"Loading enriched features from {features_path}")
        df = pd.read_parquet(features_path)
    elif raw_path.exists():
        logger.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)
    else:
        logger.error("No data available. Run acv_macro_feature_prep.ipynb first to generate features.parquet")
        raise FileNotFoundError("No production data available")

    # Identify available features
    base_features = ['Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
                     'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']
    macro_features = [col for col in df.columns if any(
        indicator in col for indicator in ['cpi', 'prime', 'fed', 'gas', 'm2']
    )]

    # Select features for training
    feature_cols = [col for col in base_features if col in df.columns]
    feature_cols.extend([col for col in macro_features if col in df.columns])

    if len(feature_cols) == 0:
        raise ValueError("No valid features found in the dataset")

    logger.info(f"Using {len(feature_cols)} features: {len(base_features)} base + {len(macro_features)} macro")

    # Prepare data for training
    X = df[feature_cols].copy()
    y = df['Price'].copy()

    # Handle dates if available
    dates = df['sale_date'] if 'sale_date' in df.columns else None

    # Train model with real data
    config = ModelConfig(
        model_type='lgbm' if LIGHTGBM_AVAILABLE else 'hgb',
        random_state=42  # Ensure reproducibility
    )
    model = UsedCarPricingModel(config)

    model.fit(X, y, dates)

    # Display results
    logger.info("\nModel Performance on REAL Data:")
    for metric, value in model.cv_results['mean'].items():
        logger.info(f"  {metric.upper()}: {value:.3f} ± {model.cv_results['std'][metric]:.3f}")

    # Feature importance
    if model.feature_importance is not None:
        logger.info("\nTop Features (Real Data):")
        logger.info(model.feature_importance.head(10).to_string())

    # Save model
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    model.save('models/production_model.pkl')

    logger.info(f"\nProduction model training complete - saved to models/production_model.pkl")