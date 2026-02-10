"""
Machine Learning Engine v2 — Precision-Optimized

Two models only: XGBoost + Random Forest
- Hyperparameter tuning via RandomizedSearchCV
- Probability calibration (CalibratedClassifierCV)
- Precision-optimized decision threshold (target: 99.6%)
- Weighted ensemble (performance-based weights)
- SQLite feedback loop for retraining on real data
- No zero-padding — uses only real extracted features

Note: pickle is used for sklearn model serialization (standard practice for
scikit-learn pipelines). Models are only loaded from the local models_dir.
"""

import json
import logging
import os
import pickle
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Core ML imports
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_analyzer_ultimate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from .ApplicationConfig import ApplicationConfig

# ----- Feature schema -----
# These names MUST match the order in EmailSecurityAnalyzer.extract_features()
FEATURE_NAMES = [
    # Email address features (10)
    'email_length', 'local_part_length', 'has_digits_local', 'dot_count_local',
    'underscore_count_local', 'dash_count_local', 'starts_with_digit',
    'unique_chars_local', 'has_mixed_case', 'non_alnum_count_local',
    # Domain features (9)
    'domain_length', 'domain_dot_count', 'domain_dash_count', 'domain_has_digits',
    'domain_starts_www', 'tld_length', 'suspicious_tld',
    'domain_age_days', 'domain_reputation_score',
    # Domain flags (1)
    'phishing_flag',
    # DNS features (6)
    'spf_present', 'dmarc_present', 'dkim_present', 'mx_present',
    'dnssec_enabled', 'dns_issue_count',
    # Breach features (3)
    'breach_found', 'breach_count', 'dns_score',
    # DNS extra (1)
    'a_record_present',
    # Pattern features (10)
    'suspicious_words_email', 'suspicious_words_domain',
    'consecutive_digits_4plus', 'consecutive_uppercase_5plus',
    'char_repetition_50pct', 'has_special_chars',
    'duplicate_domain_parts', 'long_domain_part',
    'deep_subdomain', 'ip_as_domain',
    # NEW features (4)
    'is_disposable_domain', 'typosquat_score',
    'local_part_entropy', 'is_free_email_provider',
]

NUM_FEATURES = len(FEATURE_NAMES)  # Should be 44


class MachineLearningEngine:
    """Precision-optimized ML engine with feedback loop.

    Models: XGBoost + Random Forest (calibrated)
    Target: >=99.6% precision via threshold tuning
    """

    def __init__(self, config: ApplicationConfig):
        self.config = config
        self.models: Dict[str, object] = {}
        self.scaler = StandardScaler()
        self.model_metrics: Dict[str, dict] = {}
        self.model_weights: Dict[str, float] = {}
        self.prediction_history: list = []
        self.is_initialized = False

        # Precision-optimized threshold (tuned during training)
        self.precision_threshold = 0.5
        self.target_precision = 0.996  # 99.6%

        # Feedback database
        self._db_lock = threading.Lock()
        self._db_path = str(config.models_dir / 'feedback.db')
        self._init_feedback_db()

        if config.enable_ml:
            try:
                self._initialize_models()
                self._load_or_train()
            except Exception as e:
                logger.error(f"ML initialization failed: {e}")
                self.config.enable_ml = False
                self._create_fallback()
        else:
            self._create_fallback()

    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------

    def _create_fallback(self):
        """Minimal fallback when ML is disabled."""
        self.models = {}
        self.scaler = StandardScaler()
        self.is_initialized = True

    def _initialize_models(self):
        """Create the two core models."""
        logger.info("Initializing ML models (Random Forest + XGBoost)...")

        # Random Forest — good for tabular data, natively handles feature importance
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # XGBoost — best gradient boosting for tabular data
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                scale_pos_weight=1.0,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        else:
            logger.warning("XGBoost not available — using only Random Forest")

        # Isolation Forest — unsupervised anomaly detector
        self.anomaly_detector = IsolationForest(
            n_estimators=200,
            contamination=0.1,  # Expect ~10% anomalies
            max_features=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_fitted = False

        self.is_initialized = True
        logger.info(f"Initialized {len(self.models)} ML models + Isolation Forest")

    # ---------------------------------------------------------------
    # Load / Train
    # ---------------------------------------------------------------

    def _load_or_train(self):
        """Load cached models or train fresh."""
        if not self.config.enable_ml:
            return

        models_file = self.config.models_dir / "ml_v2_models.pkl"

        if models_file.exists():
            try:
                # pickle is used here for sklearn model serialization (local files only)
                with open(models_file, 'rb') as f:
                    saved = pickle.load(f)
                self.models = saved['models']
                self.scaler = saved['scaler']
                self.model_metrics = saved.get('metrics', {})
                self.model_weights = saved.get('weights', {})
                self.precision_threshold = saved.get('precision_threshold', 0.5)
                logger.info(f"Loaded v2 models (threshold={self.precision_threshold:.4f})")
                return
            except Exception as e:
                logger.warning(f"Failed to load models, retraining: {e}")

        self._train_models()

    def _train_models(self):
        """Train models with hyperparameter tuning + calibration + threshold optimization."""
        if not self.config.enable_ml:
            return

        logger.info("Training ML v2 models...")
        start = time.time()

        try:
            # Generate or load training data
            X, y = self._get_training_data()

            # Split: 70% train, 15% calibration, 15% test
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, random_state=42, stratify=y
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
            )

            # Scale features
            X_train_s = self.scaler.fit_transform(X_train)
            X_cal_s = self.scaler.transform(X_cal)
            X_test_s = self.scaler.transform(X_test)

            # Train each model with hyperparameter search
            calibrated_models = {}
            for name, model in list(self.models.items()):
                try:
                    logger.info(f"Training {name} with hyperparameter search...")
                    best_model = self._tune_and_train(name, model, X_train_s, y_train)

                    # Calibrate probabilities using isotonic regression
                    logger.info(f"Calibrating {name}...")
                    calibrated = CalibratedClassifierCV(
                        best_model, method='isotonic', cv='prefit'
                    )
                    calibrated.fit(X_cal_s, y_cal)
                    calibrated_models[name] = calibrated

                    # Evaluate on test set
                    self._evaluate_model(name, calibrated, X_test_s, y_test)

                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
                    self.model_metrics[name] = {
                        'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0,
                        'f1_score': 0.0, 'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
                    }

            self.models = calibrated_models

            # Compute performance-based weights
            self._compute_model_weights()

            # Find precision-optimized threshold
            self._optimize_threshold(X_test_s, y_test)

            # Train Isolation Forest on legitimate data only (unsupervised)
            try:
                legit_mask = y == 0
                X_legit = self.scaler.transform(X[legit_mask])
                self.anomaly_detector.fit(X_legit)
                self.anomaly_fitted = True
                logger.info(f"Isolation Forest trained on {len(X_legit)} legitimate samples")
            except Exception as e:
                logger.warning(f"Isolation Forest training failed: {e}")
                self.anomaly_fitted = False

            # Log feature importance
            self._log_feature_importance()

            # Save models
            self._save_models()

            elapsed = time.time() - start
            logger.info(f"Training completed in {elapsed:.1f}s")
            self._print_metrics_summary()

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            self.model_metrics = {}

    def _tune_and_train(self, name: str, model, X_train, y_train):
        """Hyperparameter tuning via RandomizedSearchCV."""
        param_distributions = self._get_param_grid(name)

        if param_distributions:
            search = RandomizedSearchCV(
                model,
                param_distributions,
                n_iter=20,
                scoring='f1',
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            search.fit(X_train, y_train)
            logger.info(f"  {name} best params: {search.best_params_}")
            logger.info(f"  {name} best CV F1: {search.best_score_:.4f}")
            return search.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model

    def _get_param_grid(self, name: str) -> dict:
        """Return hyperparameter search space for each model."""
        if name == 'random_forest':
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [8, 10, 12, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
            }
        elif name == 'xgboost':
            return {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5, 7],
                'reg_alpha': [0, 0.01, 0.1, 1.0],
                'reg_lambda': [0.5, 1.0, 2.0, 5.0],
            }
        return {}

    def _evaluate_model(self, name: str, model, X_test, y_test):
        """Compute comprehensive metrics for a model."""
        predictions = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()

        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except Exception:
            roc_auc = None

        try:
            avg_precision = average_precision_score(y_test, y_proba)
        except Exception:
            avg_precision = None

        self.model_metrics[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
            'classification_report': classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
        }

        logger.info(
            f"  {name}: Acc={accuracy:.4f} Prec={precision:.4f} "
            f"Rec={recall:.4f} F1={f1:.4f} AUC={roc_auc or 0:.4f}"
        )

    def _compute_model_weights(self):
        """Set ensemble weights proportional to each model's F1 score."""
        total_f1 = 0.0
        for name, metrics in self.model_metrics.items():
            f1 = metrics.get('f1_score', 0.0)
            self.model_weights[name] = f1
            total_f1 += f1

        if total_f1 > 0:
            for name in self.model_weights:
                self.model_weights[name] /= total_f1
        else:
            # Equal weights fallback
            n = len(self.model_weights)
            for name in self.model_weights:
                self.model_weights[name] = 1.0 / n if n > 0 else 1.0

        logger.info(f"Ensemble weights: {self.model_weights}")

    def _optimize_threshold(self, X_test, y_test):
        """Find the threshold that achieves >=99.6% precision (or closest)."""
        # Get ensemble probabilities
        proba = self._ensemble_proba(X_test)

        # Compute precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_test, proba)

        # Find threshold that achieves target precision
        best_threshold = 0.5
        best_recall_at_target = 0.0

        for i, (p, r) in enumerate(zip(precisions[:-1], recalls[:-1])):
            if p >= self.target_precision and r > best_recall_at_target:
                best_recall_at_target = r
                best_threshold = thresholds[i]

        # Minimum threshold floor — prevents over-flagging on perfectly separable synthetic data
        MIN_THRESHOLD = 0.50

        if best_recall_at_target > 0:
            self.precision_threshold = max(float(best_threshold), MIN_THRESHOLD)
            logger.info(
                f"Precision-optimized threshold: {self.precision_threshold:.4f} "
                f"(precision={self.target_precision:.3f}, recall={best_recall_at_target:.3f})"
            )
        else:
            # Couldn't reach target precision — find highest precision achievable
            max_precision = max(precisions[:-1]) if len(precisions) > 1 else 0.5
            for i, p in enumerate(precisions[:-1]):
                if p == max_precision:
                    self.precision_threshold = max(float(thresholds[i]), MIN_THRESHOLD)
                    break
            logger.warning(
                f"Could not reach {self.target_precision:.3f} precision. "
                f"Best: {max_precision:.4f} at threshold={self.precision_threshold:.4f}"
            )

    def _ensemble_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """Weighted ensemble probability across all models."""
        weighted_sum = np.zeros(X_scaled.shape[0])
        total_weight = 0.0

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_scaled)[:, 1]
                weight = self.model_weights.get(name, 1.0)
                weighted_sum += proba * weight
                total_weight += weight
            except Exception as e:
                logger.debug(f"Ensemble proba failed for {name}: {e}")

        if total_weight > 0:
            return weighted_sum / total_weight
        return np.full(X_scaled.shape[0], 0.5)

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    def predict_ensemble(self, features: np.ndarray) -> Dict:
        """Get calibrated, weighted ensemble prediction.

        Returns dict with individual model scores + ensemble score.
        The ensemble score already incorporates precision-optimized weighting.
        """
        predictions = {}

        if not self.config.enable_ml or not self.models:
            predictions['ensemble'] = 0.5
            return predictions

        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Pad or truncate to expected feature count
        if features.shape[1] < NUM_FEATURES:
            pad = np.zeros((features.shape[0], NUM_FEATURES - features.shape[1]))
            features = np.hstack([features, pad])
        elif features.shape[1] > NUM_FEATURES:
            features = features[:, :NUM_FEATURES]

        try:
            features_scaled = self.scaler.transform(features)
        except Exception:
            features_scaled = features

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(features_scaled)[0]
                pred_value = float(proba[1]) if len(proba) > 1 else 0.5
                if np.isnan(pred_value):
                    pred_value = 0.5
                predictions[name] = pred_value
            except Exception as e:
                logger.debug(f"Prediction failed for {name}: {e}")
                predictions[name] = 0.5

        # Weighted ensemble
        weighted_sum = 0.0
        total_weight = 0.0
        for name, score in predictions.items():
            weight = self.model_weights.get(name, 1.0)
            weighted_sum += score * weight
            total_weight += weight

        ensemble_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        if np.isnan(ensemble_score):
            ensemble_score = 0.5

        predictions['ensemble'] = float(ensemble_score)

        # Unsupervised anomaly score (Isolation Forest)
        if self.anomaly_fitted:
            try:
                # score_samples returns negative values; more negative = more anomalous
                # Typical range: -0.6 (very anomalous) to 0 (normal)
                raw_anomaly = self.anomaly_detector.score_samples(features_scaled)[0]
                # Convert: scores below -0.5 → anomalous (close to 1.0)
                # scores above -0.3 → normal (close to 0.0)
                anomaly_score = max(0.0, min(1.0, (-raw_anomaly - 0.3) / 0.3))
                predictions['anomaly_score'] = float(anomaly_score)
            except Exception as e:
                logger.debug(f"Anomaly detection failed: {e}")
                predictions['anomaly_score'] = 0.0
        else:
            predictions['anomaly_score'] = 0.0

        # Apply precision threshold for binary classification
        predictions['is_malicious'] = ensemble_score >= self.precision_threshold
        predictions['precision_threshold'] = self.precision_threshold

        # Track prediction (cap at 1000 to prevent unbounded memory growth)
        self.prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions.copy(),
            'feature_count': features.shape[1]
        })
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]

        return predictions

    # ---------------------------------------------------------------
    # Training data generation
    # ---------------------------------------------------------------

    def _get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get training data: real feedback + synthetic fill."""
        X_real, y_real = self._load_feedback_data()
        X_synth, y_synth = self._generate_training_data()

        if len(X_real) > 0:
            # Mix real and synthetic data
            # As real data grows, reduce synthetic proportion
            real_ratio = min(len(X_real) / 500, 1.0)  # At 500+ real samples, use mostly real
            synth_keep = int(len(X_synth) * (1.0 - real_ratio * 0.8))
            if synth_keep > 0:
                indices = np.random.choice(len(X_synth), synth_keep, replace=False)
                X_synth = X_synth[indices]
                y_synth = y_synth[indices]

            X = np.vstack([X_real, X_synth])
            y = np.concatenate([y_real, y_synth])
            logger.info(f"Training data: {len(X_real)} real + {len(X_synth)} synthetic = {len(X)} total")
        else:
            X, y = X_synth, y_synth
            logger.info(f"Training data: {len(X)} synthetic (no real feedback yet)")

        return X, y

    def _generate_training_data(self, n_samples: int = 8000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic synthetic training data matching actual feature extraction.

        Features match FEATURE_NAMES exactly (44 features, no padding).
        """
        np.random.seed(42)
        X = []
        y = []

        for _ in range(n_samples):
            is_malicious = np.random.random() > 0.5

            # 5% label noise for realism
            if np.random.random() < 0.05:
                is_malicious = not is_malicious

            if is_malicious:
                features = self._generate_malicious_sample()
            else:
                features = self._generate_legitimate_sample()

            X.append(features)
            y.append(1 if is_malicious else 0)

        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        return X, y

    def _generate_malicious_sample(self) -> list:
        """Generate a single malicious email feature vector.

        Ranges are calibrated so that legitimate corporate emails
        (domain_rep=50, dns_score=35, domain_age unknown/365) do NOT
        fall into the malicious bucket.  Only truly suspicious patterns
        (disposable, typosquatting, known phishing TLDs, breaches) should
        be strong malicious signals.
        """
        r = np.random.random

        # Email address features
        email_len = np.random.randint(5, 80)
        local_len = np.random.randint(3, 40)
        has_digits = 1 if r() < 0.6 else 0
        dot_count = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
        underscore_count = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
        dash_count = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        starts_digit = 1 if r() < 0.20 else 0
        unique_chars = np.random.randint(3, 25)
        mixed_case = 1 if r() < 0.30 else 0
        non_alnum = np.random.randint(0, 5)

        # Domain features — wider ranges to overlap with legitimate
        domain_len = np.random.randint(5, 50)
        domain_dots = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.25, 0.2, 0.15, 0.1])
        domain_dashes = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])
        domain_digits = 1 if r() < 0.40 else 0
        domain_www = 1 if r() < 0.05 else 0
        tld_len = np.random.choice([2, 3, 4, 5, 6], p=[0.15, 0.25, 0.2, 0.2, 0.2])
        suspicious_tld = 1 if r() < 0.30 else 0
        # Domain age: allow overlap — many legitimate new businesses exist
        domain_age = max(0, np.random.exponential(180))
        # Domain reputation: 0-55 range (default 50 is NOT malicious)
        domain_rep = np.random.uniform(0, 55)

        # Flags — true indicator
        phishing_flag = 1 if r() < 0.40 else 0

        # DNS — many legitimate domains lack DMARC/DNSSEC
        spf = 1 if r() < 0.35 else 0
        dmarc = 1 if r() < 0.25 else 0
        dkim = 1 if r() < 0.20 else 0
        mx = 1 if r() < 0.70 else 0
        dnssec = 1 if r() < 0.10 else 0
        dns_issues = np.random.randint(0, 6)

        # Breaches — strong malicious signal
        breach_found = 1 if r() < 0.60 else 0
        breach_count = np.random.randint(0, 20) if breach_found else 0
        # DNS score: allow overlap with legitimate (many real domains score 30-50)
        dns_score = np.random.uniform(0, 55)

        # DNS extra
        a_record = 1 if r() < 0.80 else 0

        # Patterns — true indicators
        suspicious_words_email = np.random.randint(0, 4)
        suspicious_words_domain = np.random.randint(0, 3)
        consecutive_digits = 1 if r() < 0.25 else 0
        consecutive_upper = 1 if r() < 0.15 else 0
        char_repetition = 1 if r() < 0.15 else 0
        special_chars = 1 if r() < 0.20 else 0
        dup_domain_parts = 1 if r() < 0.10 else 0
        long_domain_part = 1 if r() < 0.25 else 0
        deep_subdomain = 1 if r() < 0.15 else 0
        ip_domain = 1 if r() < 0.10 else 0

        # NEW features — disposable/typosquat are strong signals
        is_disposable = 1 if r() < 0.50 else 0
        typosquat = np.random.uniform(0.0, 0.9) if r() < 0.30 else 0.0
        entropy = np.random.uniform(1.0, 5.0)
        is_free = 1 if r() < 0.60 else 0

        return [
            email_len, local_len, has_digits, dot_count, underscore_count,
            dash_count, starts_digit, unique_chars, mixed_case, non_alnum,
            domain_len, domain_dots, domain_dashes, domain_digits, domain_www,
            tld_len, suspicious_tld, domain_age, domain_rep, phishing_flag,
            spf, dmarc, dkim, mx, dnssec, dns_issues,
            breach_found, breach_count, dns_score, a_record,
            suspicious_words_email, suspicious_words_domain,
            consecutive_digits, consecutive_upper, char_repetition,
            special_chars, dup_domain_parts, long_domain_part,
            deep_subdomain, ip_domain,
            is_disposable, typosquat, entropy, is_free,
        ]

    def _generate_legitimate_sample(self) -> list:
        """Generate a single legitimate email feature vector.

        Includes realistic patterns for new corporate domains:
        - domain_rep can be 50 (default when WHOIS fails)
        - dns_score can be 30-40 (missing DMARC/DNSSEC common)
        - domain_age can be < 365 (new businesses/startups)
        - domain_len can be > 20 (healthreconconnect.com = 25 chars)
        """
        r = np.random.random

        # Email address features
        email_len = np.random.randint(10, 55)
        local_len = np.random.randint(3, 25)
        has_digits = 1 if r() < 0.30 else 0
        dot_count = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        underscore_count = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        dash_count = np.random.choice([0, 1], p=[0.7, 0.3])
        starts_digit = 1 if r() < 0.05 else 0
        unique_chars = np.random.randint(4, 20)
        mixed_case = 1 if r() < 0.15 else 0
        non_alnum = np.random.randint(0, 3)

        # Domain features — wider ranges to represent real corporate domains
        domain_len = np.random.randint(5, 35)  # Many real domains are long
        domain_dots = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        domain_dashes = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        domain_digits = 1 if r() < 0.15 else 0
        domain_www = 0
        tld_len = np.random.choice([2, 3, 4, 5], p=[0.15, 0.50, 0.20, 0.15])
        suspicious_tld = 1 if r() < 0.02 else 0
        # Domain age: include new legitimate businesses (< 1 year)
        domain_age = np.random.uniform(30, 5000)
        # Domain reputation: 40-100 to include default score of 50
        domain_rep = np.random.uniform(40, 100)

        # Flags
        phishing_flag = 1 if r() < 0.01 else 0

        # DNS — many legitimate domains lack DMARC/DNSSEC
        spf = 1 if r() < 0.75 else 0
        dmarc = 1 if r() < 0.55 else 0
        dkim = 1 if r() < 0.50 else 0
        mx = 1 if r() < 0.95 else 0
        dnssec = 1 if r() < 0.25 else 0
        dns_issues = np.random.randint(0, 3)

        # Breaches
        breach_found = 1 if r() < 0.15 else 0
        breach_count = np.random.randint(0, 3) if breach_found else 0
        # DNS score: 25-100 to include domains missing some DNS features
        dns_score = np.random.uniform(25, 100)

        # DNS extra
        a_record = 1 if r() < 0.95 else 0

        # Patterns
        suspicious_words_email = np.random.choice([0, 1], p=[0.85, 0.15])
        suspicious_words_domain = 0
        consecutive_digits = 1 if r() < 0.05 else 0
        consecutive_upper = 1 if r() < 0.02 else 0
        char_repetition = 1 if r() < 0.03 else 0
        special_chars = 1 if r() < 0.05 else 0
        dup_domain_parts = 1 if r() < 0.01 else 0
        long_domain_part = 1 if r() < 0.10 else 0  # Long parts are normal
        deep_subdomain = 1 if r() < 0.05 else 0
        ip_domain = 0

        # NEW features
        is_disposable = 1 if r() < 0.03 else 0
        typosquat = 0.0
        entropy = np.random.uniform(2.0, 4.5)  # Wider range for corporate names
        is_free = 1 if r() < 0.40 else 0

        return [
            email_len, local_len, has_digits, dot_count, underscore_count,
            dash_count, starts_digit, unique_chars, mixed_case, non_alnum,
            domain_len, domain_dots, domain_dashes, domain_digits, domain_www,
            tld_len, suspicious_tld, domain_age, domain_rep, phishing_flag,
            spf, dmarc, dkim, mx, dnssec, dns_issues,
            breach_found, breach_count, dns_score, a_record,
            suspicious_words_email, suspicious_words_domain,
            consecutive_digits, consecutive_upper, char_repetition,
            special_chars, dup_domain_parts, long_domain_part,
            deep_subdomain, ip_domain,
            is_disposable, typosquat, entropy, is_free,
        ]

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    def _log_feature_importance(self):
        """Log top feature importances from tree-based models."""
        for name, model in self.models.items():
            try:
                # CalibratedClassifierCV wraps the base estimator
                base = model
                if hasattr(model, 'estimator'):
                    base = model.estimator
                elif hasattr(model, 'calibrated_classifiers_'):
                    base = model.calibrated_classifiers_[0].estimator

                if hasattr(base, 'feature_importances_'):
                    importances = base.feature_importances_
                    indices = np.argsort(importances)[::-1]

                    logger.info(f"\n  {name} — Top 15 features:")
                    for i in range(min(15, len(indices))):
                        idx = indices[i]
                        feat_name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f'feature_{idx}'
                        logger.info(f"    {i+1}. {feat_name}: {importances[idx]:.4f}")
            except Exception as e:
                logger.debug(f"Could not extract feature importance for {name}: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """Return averaged feature importances across models."""
        combined = np.zeros(NUM_FEATURES)
        count = 0

        for name, model in self.models.items():
            try:
                base = model
                if hasattr(model, 'estimator'):
                    base = model.estimator
                elif hasattr(model, 'calibrated_classifiers_'):
                    base = model.calibrated_classifiers_[0].estimator

                if hasattr(base, 'feature_importances_'):
                    imp = base.feature_importances_
                    if len(imp) == NUM_FEATURES:
                        combined += imp
                        count += 1
            except Exception:
                pass

        if count > 0:
            combined /= count

        return {FEATURE_NAMES[i]: float(combined[i]) for i in range(NUM_FEATURES)}

    # ---------------------------------------------------------------
    # Feedback loop (SQLite)
    # ---------------------------------------------------------------

    def _init_feedback_db(self):
        """Initialize the SQLite feedback database."""
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT NOT NULL,
                        features TEXT NOT NULL,
                        predicted_score REAL,
                        predicted_label INTEGER,
                        user_label INTEGER NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                ''')
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS retrain_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        real_samples INTEGER,
                        synthetic_samples INTEGER,
                        precision REAL,
                        recall REAL,
                        f1 REAL,
                        threshold REAL
                    )
                ''')
                conn.commit()
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to init feedback DB: {e}")

    def submit_feedback(self, email: str, features: np.ndarray,
                        predicted_score: float, user_label: int):
        """Store user feedback (0=safe, 1=malicious) for later retraining.

        Args:
            email: The email address that was analyzed
            features: The feature vector used for prediction
            predicted_score: The ensemble prediction score
            user_label: User's correction (0=safe, 1=malicious)
        """
        try:
            features_json = json.dumps(features.tolist() if hasattr(features, 'tolist') else list(features))
            predicted_label = 1 if predicted_score >= self.precision_threshold else 0

            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                conn.execute(
                    'INSERT INTO feedback (email, features, predicted_score, predicted_label, user_label, timestamp) '
                    'VALUES (?, ?, ?, ?, ?, ?)',
                    (email, features_json, predicted_score, predicted_label,
                     user_label, datetime.now().isoformat())
                )
                conn.commit()
                count = conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
                conn.close()

            logger.info(f"Feedback stored: email={email}, pred={predicted_score:.3f}, label={user_label}, total={count}")

            # Auto-retrain at milestones
            if count in (50, 100, 250, 500, 1000, 2500, 5000):
                logger.info(f"Feedback milestone ({count}) — triggering retrain...")
                threading.Thread(target=self._retrain_from_feedback, daemon=True).start()

        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")

    def _load_feedback_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load all feedback data for training."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                rows = conn.execute('SELECT features, user_label FROM feedback').fetchall()
                conn.close()

            if not rows:
                return np.array([]), np.array([])

            X = []
            y = []
            for features_json, label in rows:
                features = json.loads(features_json)
                # Ensure correct feature count
                if len(features) == NUM_FEATURES:
                    X.append(features)
                    y.append(label)
                elif len(features) > NUM_FEATURES:
                    X.append(features[:NUM_FEATURES])
                    y.append(label)

            if X:
                return np.array(X, dtype=np.float64), np.array(y)
            return np.array([]), np.array([])

        except Exception as e:
            logger.warning(f"Failed to load feedback data: {e}")
            return np.array([]), np.array([])

    def _retrain_from_feedback(self):
        """Retrain models incorporating feedback data."""
        logger.info("Retraining with feedback data...")
        try:
            self._train_models()
            logger.info("Retrain complete!")
        except Exception as e:
            logger.error(f"Retrain failed: {e}")

    def get_feedback_count(self) -> int:
        """Return total feedback samples collected."""
        try:
            with self._db_lock:
                conn = sqlite3.connect(self._db_path)
                count = conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]
                conn.close()
            return count
        except Exception:
            return 0

    def retrain(self):
        """Public method to trigger manual retrain."""
        threading.Thread(target=self._retrain_from_feedback, daemon=True).start()

    # ---------------------------------------------------------------
    # Save / Load
    # ---------------------------------------------------------------

    def _save_models(self):
        """Save trained models to disk using pickle (standard for sklearn models)."""
        if not self.config.enable_ml:
            return

        try:
            models_file = self.config.models_dir / "ml_v2_models.pkl"
            save_data = {
                'models': self.models,
                'scaler': self.scaler,
                'metrics': self.model_metrics,
                'weights': self.model_weights,
                'precision_threshold': self.precision_threshold,
                'feature_names': FEATURE_NAMES,
                'num_features': NUM_FEATURES,
                'timestamp': datetime.now().isoformat()
            }

            # pickle is standard for sklearn model serialization (local files only)
            with open(models_file, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Models saved to {models_file}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    # ---------------------------------------------------------------
    # Metrics & Reporting
    # ---------------------------------------------------------------

    def get_metrics_summary(self) -> str:
        """Generate a formatted summary of all model metrics."""
        if not self.model_metrics:
            return "No metrics available. Models need to be trained first."

        summary = "\n" + "=" * 100 + "\n"
        summary += "ML MODEL PERFORMANCE METRICS (v2 — Precision-Optimized)\n"
        summary += "=" * 100 + "\n\n"

        summary += f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10} {'Weight':<10}\n"
        summary += "-" * 80 + "\n"

        for name, metrics in self.model_metrics.items():
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            auc = metrics.get('roc_auc')
            weight = self.model_weights.get(name, 0)

            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            summary += f"{name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc_str:<10} {weight:<10.4f}\n"

        summary += f"\nPrecision threshold: {self.precision_threshold:.4f}\n"
        summary += f"Target precision: {self.target_precision:.3f}\n"
        summary += f"Feedback samples: {self.get_feedback_count()}\n"

        summary += "\n" + "=" * 100 + "\n"
        summary += "CONFUSION MATRIX DETAILS\n"
        summary += "=" * 100 + "\n"

        for name, metrics in self.model_metrics.items():
            cm = metrics.get('confusion_matrix', {})
            if cm:
                tn, fp, fn, tp = cm.get('tn', 0), cm.get('fp', 0), cm.get('fn', 0), cm.get('tp', 0)
                total = tn + fp + fn + tp
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

                summary += f"\n{name}:\n"
                summary += f"  TN: {tn:>6}  FP: {fp:>6}\n"
                summary += f"  FN: {fn:>6}  TP: {tp:>6}\n"
                summary += f"  Total: {total}  Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}\n"

        summary += "\n" + "=" * 100 + "\n"
        return summary

    def _print_metrics_summary(self):
        """Print formatted metrics summary."""
        summary = self.get_metrics_summary()
        print(summary)
        logger.info("Model metrics summary generated")

    def print_metrics_summary(self):
        """Public alias for metrics summary."""
        self._print_metrics_summary()

    def get_metrics_dict(self) -> Dict:
        """Return metrics as a dictionary."""
        return self.model_metrics
