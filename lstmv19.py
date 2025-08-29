#!/usr/bin/env python3
"""
Advanced Battery Discharge Analysis System with Meta-Ensemble Comparison
This script performs comprehensive analysis of battery discharge data using multiple
deep learning architectures, then employs a meta-ensemble neural network to
compare and optimize predictions across all models.
"""
import os
import sys
import time
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Bidirectional, Dropout, Conv1D,
    MaxPooling1D, Input, Lambda,
    GlobalAveragePooling1D, LayerNormalization,
    MultiHeadAttention, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, median_absolute_error
)
from sklearn.exceptions import ConvergenceWarning
from pywt import wavedec
from tqdm import tqdm
import traceback
import math

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# Configure matplotlib for better visualizations
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 1.5,
    'grid.alpha': 0.3
})

# -------------------------------
# GPU Setup Function
# -------------------------------
def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"✅ GPU acceleration enabled with {len(gpus)} GPU(s)")
            return True
        except RuntimeError as e:
            logging.warning(f"⚠️ GPU setup failed: {e}")
            return False
    else:
        logging.info("ℹ️ No GPU available. Using CPU.")
        return False

# -------------------------------
# Custom Transformer Block
# -------------------------------
class TransformerBlock(tf.keras.layers.Layer):
    """Custom Transformer block for temporal sequence modeling."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate
        })
        return config

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# -------------------------------
# Physics-Informed Neural Network
# -------------------------------
class PINNModel(Model):
    """Physics-Informed Neural Network with custom training step."""
    def __init__(self, base_model, physics_weight=0.1):
        super(PINNModel, self).__init__()
        self.base_model = base_model
        self.physics_weight = physics_weight
        self.mse_loss = tf.keras.losses.MeanSquaredError()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'base_model': self.base_model,
            'physics_weight': self.physics_weight
        })
        return config

    def call(self, inputs):
        return self.base_model(inputs)
    
    def physics_constraints(self, inputs, outputs):
        """Calculate physics-based residuals for battery discharge."""
        # Extract the last timestep's time value
        time_idx = -1
        # Calculate time difference between last two timesteps
        dt = inputs[:, -1, time_idx] - inputs[:, -2, time_idx]
        dt = tf.maximum(dt, 1e-6)  # Ensure dt is not zero
        
        # Extract voltage and current values
        voltage_idx = 0
        v_last = inputs[:, -1, voltage_idx]
        current_idx = 2
        current = inputs[:, -1, current_idx]
        
        # Calculate expected voltage change based on current
        capacity = 1.0  # Assuming normalized capacity
        expected_dv = -current * dt / capacity
        expected_v = v_last + expected_dv
        
        # Calculate physics loss
        physics_loss = tf.reduce_mean(tf.square(outputs - expected_v))
        return physics_loss

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            data_loss = self.mse_loss(y, y_pred)
            physics_loss = self.physics_constraints(x, y_pred)
            total_loss = data_loss + self.physics_weight * physics_loss
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "physics_loss": physics_loss
        }

    def test_step(self, data):
        """Test step that only computes data loss (without physics loss)."""
        x, y = data
        y_pred = self(x, training=False)
        data_loss = self.mse_loss(y, y_pred)
        return {"loss": data_loss, "data_loss": data_loss}

# -------------------------------
# Meta-Ensemble Neural Network (FIXED)
# -------------------------------
class MetaEnsembleModel(Model):
    """Meta-Ensemble model that learns to optimally combine predictions from multiple models."""
    def __init__(self, num_models, sequence_length, feature_dim, uncertainty_aware=True):
        super(MetaEnsembleModel, self).__init__()
        self.uncertainty_aware = uncertainty_aware
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_models = num_models
        
        # Prediction processing branch
        self.conv1 = Conv1D(16, 3, activation='relu', padding='same')
        self.dropout1 = Dropout(0.1)
        self.conv2 = Conv1D(8, 3, activation='relu', padding='same')
        self.dropout2 = Dropout(0.1)
        
        # Uncertainty processing branch (if applicable)
        if uncertainty_aware:
            self.uncert_conv1 = Conv1D(16, 3, activation='relu', padding='same')
            self.uncert_dropout1 = Dropout(0.1)
            self.uncert_conv2 = Conv1D(8, 3, activation='relu', padding='same')
            self.uncert_dropout2 = Dropout(0.1)
        
        # LSTM layers
        self.bilstm1 = Bidirectional(LSTM(16, return_sequences=True))
        self.dropout3 = Dropout(0.2)
        self.bilstm2 = Bidirectional(LSTM(8))
        self.dropout4 = Dropout(0.2)
        
        # Output layers
        self.pred_output = Dense(1, name='prediction')
        self.confidence_output = Dense(1, activation='sigmoid', name='confidence')
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_models': self.num_models,
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'uncertainty_aware': self.uncertainty_aware
        })
        return config

    def call(self, inputs, training=False):
        if self.uncertainty_aware:
            pred_input, uncert_input = inputs
        else:
            pred_input = inputs
            
        # Process predictions
        x = self.conv1(pred_input)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        
        # Process uncertainties if applicable
        if self.uncertainty_aware:
            u = self.uncert_conv1(uncert_input)
            u = self.uncert_dropout1(u, training=training)
            u = self.uncert_conv2(u)
            u = self.uncert_dropout2(u, training=training)
            x = Concatenate()([x, u])
        
        # Process through LSTM layers
        x = self.bilstm1(x)
        x = self.dropout3(x, training=training)
        x = self.bilstm2(x)
        x = self.dropout4(x, training=training)
        
        # Generate outputs
        pred = self.pred_output(x)
        conf = self.confidence_output(x)
        
        return pred, conf
    
    def predict_with_uncertainty(self, model_predictions, uncertainty_estimates=None, n_samples=50):
        """Predict with Monte Carlo dropout to estimate uncertainty."""
        if self.uncertainty_aware and uncertainty_estimates is None:
            raise ValueError("Uncertainty estimates required for uncertainty-aware model")
        
        predictions = []
        for _ in range(n_samples):
            if self.uncertainty_aware:
                pred, _ = self([model_predictions, uncertainty_estimates], training=True)
            else:
                pred, _ = self(model_predictions, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_uncert = np.std(predictions, axis=0)
        return ensemble_pred, ensemble_uncert

# -------------------------------
# Battery Discharge Analyzer
# -------------------------------
class BatteryDischargeAnalyzer:
    """Main class for battery discharge data analysis with deep learning models."""
    def __init__(self, config=None):
        self.config = {
            'data_directory': '/content/drive/MyDrive/all_data_csv',
            'output_directory': '/content/drive/MyDrive/all_data_csv',
            'file_names': ['0.5cDA.csv', '1cDA.csv', '2cDA.csv', '3cDA.csv', '5cDA.csv'],
            'labels': ['0.5C', '1C', '2C', '3C', '5C'],
            'sequence_length': 15,
            'validation_split': 0.2,
            'batch_size': 32,
            'epochs': 120,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 8,
            'min_learning_rate': 5e-7,
            'bayesian_samples': 50,
            'wavelet': 'db4',
            'wavelet_level': 3,
            'rolling_window': 7,
            'gpu_enabled': True,
            'ensemble_epochs': 80,
            'ensemble_patience': 10,
            'min_ensemble_data': 20  # Minimum data points needed for meta-ensemble
        }
        if config:
            self.config.update(config)
        self.config['data_directory'] = self.config['data_directory'].strip()
        self.config['output_directory'] = self.config['output_directory'].strip()
        os.makedirs(self.config['output_directory'], exist_ok=True)
        log_dir = os.path.join(self.config['output_directory'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'analysis.log')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('BatteryAnalysis')
        self.gpu_available = self._setup_gpu()
        self.logger.info(f"Logger initialized. Log saved to: {log_file}")
        self.logger.info(f"GPU available: {self.gpu_available}")
        self.colors = {
            'Transformer': '#2ca02c',
            'Hybrid CNN-BiLSTM': '#9467bd',
            'Bayesian LSTM': '#1f77b4',
            'GRU': '#ff9896',
            'BiGRU': '#98df8a',
            'TCN': '#d6d6d6',
            'PINN': '#c5b0d5',
            'CNN-GRU': '#c49c94',
            'Ensemble': '#e377c2',
            'Actual': '#7f7f7f',
            'Predicted': '#ff7f0e',
            'Uncertainty': '#d62728'
        }
        self.model_names = [
            'Transformer', 'Hybrid CNN-BiLSTM', 'Bayesian LSTM',
            'GRU', 'BiGRU', 'TCN', 'PINN', 'CNN-GRU'
        ]
        self.all_metrics = {}
        self.all_predictions = {}
        self.all_times = {}
        self.all_actuals = {}
        self.model_history = {}
        self.ensemble_models = {}
    
    def _setup_gpu(self):
        if not self.config.get('gpu_enabled', True):
            self.logger.info("GPU explicitly disabled in config. Using CPU.")
            return False
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                with tf.device('/GPU:0'):
                    _ = tf.constant([1.0, 2.0, 3.0])
                self.logger.info(f"✅ GPU acceleration enabled with {len(gpus)} GPU(s)")
                return True
            except RuntimeError as e:
                self.logger.warning(f"⚠️ GPU setup failed: {e}")
                return False
        else:
            self.logger.info("ℹ️ No GPU available. Using CPU.")
            return False
    
    def time_to_seconds(self, time_str):
        try:
            if pd.isna(time_str) or time_str is None:
                return None
            parts = str(time_str).strip().split(':')
            if len(parts) == 3:
                hh, mm, ss = map(int, parts)
                return hh * 3600 + mm * 60 + ss
            elif len(parts) == 2:
                mm, ss = map(int, parts)
                return mm * 60 + ss
            else:
                return None
        except (ValueError, TypeError, AttributeError) as e:
            self.logger.warning(f"Time conversion error: {e} for '{time_str}'")
            return None
    
    def extract_wavelet_features(self, signal, wavelet=None, level=None):
        wavelet = wavelet or self.config['wavelet']
        level = level or self.config['wavelet_level']
        try:
            coeffs = wavedec(signal, wavelet, level=level)
            features = []
            for coeff in coeffs:
                if len(coeff) > 0:
                    features.extend([
                        np.mean(coeff),
                        np.std(coeff),
                        np.median(coeff),
                        np.percentile(coeff, 75) - np.percentile(coeff, 25),
                        np.max(coeff) - np.min(coeff),
                        np.sum(np.abs(np.diff(coeff)))
                    ])
            if len(coeffs) > 0 and len(coeffs[0]) > 0:
                features.extend([np.mean(coeffs[0]), np.std(coeffs[0])])
            return np.array(features)
        except Exception as e:
            self.logger.warning(f"Wavelet extraction failed: {e}")
            return np.zeros(10)
    
    def process_data(self, file_path, label):
        self.logger.info(f"Processing file: {file_path} with label: {label}")
        try:
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(data)} rows from {file_path}")
            
            # Check required columns
            required_cols = ['Time(hhh:mm:ss)', 'Voltage(mV)', 'Temperature (C)']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                self.logger.error(f"Missing columns: {missing}")
                return None
            
            # Convert time to seconds
            data['Time(s)'] = data['Time(hhh:mm:ss)'].apply(self.time_to_seconds)
            
            # Filter out invalid time entries
            mask = ~data['Time(s)'].isna()
            if mask.sum() < 20:
                self.logger.warning("Insufficient valid time data.")
                return None
            
            time_s = data.loc[mask, 'Time(s)'].values
            voltage = data.loc[mask, 'Voltage(mV)'].values / 1000.0  # Convert mV to V
            temp = data.loc[mask, 'Temperature (C)'].values
            
            if len(time_s) < 20:
                self.logger.warning("Not enough data points.")
                return None
            
            # Engineer features
            features = self._engineer_features(time_s, voltage, temp)
            if features is None or features.shape[0] == 0:
                return None
            
            # Normalize features
            normalized_features, scalers = self._normalize_features(features)
            
            # Prepare sequences
            X_seq, y_volt, y_temp, time_seq = self._prepare_sequences(normalized_features, time_s)
            if X_seq is None or len(X_seq) == 0:
                return None
            
            return X_seq, y_volt, y_temp, time_seq, scalers, features
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def _engineer_features(self, time, voltage, temperature):
        try:
            window = self.config['rolling_window']
            v_series = pd.Series(voltage)
            t_series = pd.Series(temperature)
            
            # Rolling statistics
            v_roll_mean = v_series.rolling(window=window, min_periods=1).mean().values
            v_roll_std = v_series.rolling(window=window, min_periods=1).std().fillna(0).values
            t_roll_mean = t_series.rolling(window=window, min_periods=1).mean().values
            t_roll_std = t_series.rolling(window=window, min_periods=1).std().fillna(0).values
            
            # Rate of change
            v_roc = np.gradient(voltage, time)
            t_roc = np.gradient(temperature, time)
            
            # Acceleration
            v_accel = np.gradient(v_roc, time)
            t_accel = np.gradient(t_roc, time)
            
            # Time features
            time_hrs = time / 3600
            time_mins = (time % 3600) / 60
            
            # Feature matrix
            feature_matrix = np.column_stack([
                voltage, temperature,
                np.gradient(time),
                v_roll_mean, v_roll_std,
                t_roll_mean, t_roll_std,
                v_roc, t_roc,
                v_accel, t_accel,
                time_hrs, time_mins
            ])
            
            return feature_matrix
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return None
    
    def _normalize_features(self, feature_matrix):
        scalers = {}
        normalized = np.zeros_like(feature_matrix, dtype=np.float32)
        
        for i in range(feature_matrix.shape[1]):
            # Skip if column is constant
            if np.std(feature_matrix[:, i]) < 1e-8:
                normalized[:, i] = feature_matrix[:, i]
                scalers[i] = None
                continue
            
            # Use RobustScaler for better outlier handling
            scaler = RobustScaler()
            normalized[:, i] = scaler.fit_transform(
                feature_matrix[:, i].reshape(-1, 1)
            ).flatten()
            scalers[i] = scaler
        
        return normalized, scalers
    
    def _prepare_sequences(self, features, time):
        seq_len = self.config['sequence_length']
        n_samples = len(features) - seq_len
        
        if n_samples <= 0:
            self.logger.warning("Not enough data for sequences.")
            return None, None, None, None
        
        X = np.zeros((n_samples, seq_len, features.shape[1]))
        y_volt = np.zeros(n_samples)
        y_temp = np.zeros(n_samples)
        t_seq = time[seq_len:]
        
        for i in range(n_samples):
            start = i
            end = i + seq_len
            X[i] = features[start:end]
            y_volt[i] = features[end, 0]  # Voltage is first feature
            y_temp[i] = features[end, 1]   # Temperature is second feature
        
        return X, y_volt, y_temp, t_seq
    
    def create_transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # Positional encoding function
        def add_pos_encoding(x):
            seq_len = tf.shape(x)[1]
            pos = tf.range(0, seq_len, dtype=tf.float32)[None, :, None]
            scale = tf.reduce_max(tf.abs(x), axis=[1, 2], keepdims=True)
            scale = tf.maximum(scale, 1e-8)
            pos = pos / scale
            return tf.concat([x, pos], axis=-1)
        
        x = Lambda(add_pos_encoding, name='positional_encoding')(inputs)
        x = Dense(64, activation='gelu')(x)
        x = Dropout(0.1)(x)
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, rate=0.1)(x)
        x = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128, rate=0.1)(x)
        
        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='gelu')(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(1)(x)
        return Model(inputs, outputs)
    
    def create_hybrid_cnn_bilstm_model(self, input_shape):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape, 
                  padding='same', kernel_regularizer=l2(1e-4)),
            MaxPooling1D(2),
            Dropout(0.2),
            Conv1D(128, 3, activation='relu', padding='same', 
                  kernel_regularizer=l2(1e-4)),
            MaxPooling1D(2),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True, 
                              kernel_regularizer=l2(1e-4))),
            Dropout(0.3),
            Bidirectional(LSTM(32, kernel_regularizer=l2(1e-4))),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def create_bayesian_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape, 
                kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            LSTM(32, kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def create_gru_model(self, input_shape):
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape, 
                kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            GRU(32, kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def create_bigru_model(self, input_shape):
        model = Sequential([
            Bidirectional(GRU(64, return_sequences=True, 
                             input_shape=input_shape, 
                             kernel_regularizer=l2(1e-4))),
            Dropout(0.2),
            Bidirectional(GRU(32, kernel_regularizer=l2(1e-4))),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def create_tcn_model(self, input_shape):
        inputs = Input(shape=input_shape)
        
        # First convolutional block
        x = Conv1D(64, 3, padding='causal', dilation_rate=1,
                  activation='relu', kernel_regularizer=l2(1e-4))(inputs)
        x = Dropout(0.2)(x)
        x = Conv1D(64, 3, padding='causal', dilation_rate=1,
                  activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.2)(x)
        res1 = x
        
        # Second convolutional block with residual connection
        x = Conv1D(128, 3, padding='causal', dilation_rate=2,
                  activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, 3, padding='causal', dilation_rate=2,
                  activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.2)(x)
        res2 = Conv1D(128, 1, padding='same')(res1)
        x = tf.keras.layers.Add()([x, res2])
        
        # Global pooling and dense layers
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs)
    
    def create_pinn_model(self, input_shape, base_model_type="Transformer"):
        if base_model_type == "Transformer":
            base_model = self.create_transformer_model(input_shape)
        elif base_model_type == "CNN-BiLSTM":
            base_model = self.create_hybrid_cnn_bilstm_model(input_shape)
        elif base_model_type == "GRU":
            base_model = self.create_bigru_model(input_shape)
        else:
            raise ValueError(f"Unknown base model type: {base_model_type}")
        
        pinn_model = PINNModel(base_model, physics_weight=0.1)
        return pinn_model
    
    def create_cnn_gru_model(self, input_shape):
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape, 
                  padding='same', kernel_regularizer=l2(1e-4)),
            MaxPooling1D(2),
            Dropout(0.2),
            Conv1D(128, 3, activation='relu', padding='same', 
                  kernel_regularizer=l2(1e-4)),
            MaxPooling1D(2),
            Dropout(0.2),
            GRU(64, return_sequences=True, kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            GRU(32, kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def create_meta_ensemble_model(self, sequence_length, num_models, uncertainty_aware=True):
        return MetaEnsembleModel(
            num_models=num_models,
            sequence_length=sequence_length,
            feature_dim=1,
            uncertainty_aware=uncertainty_aware
        )
    
    def evaluate_model(self, y_true, y_pred, y_std=None, label=""):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Handle NaN values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            self.logger.warning(f"No valid data points for evaluation of {label}")
            return {
                'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 
                'MDAE': np.nan, 'MAPE': np.nan,
                'Mean Error': np.nan, 'Std Error': np.nan
            }
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mdae = median_absolute_error(y_true, y_pred)
        
        # Handle division by zero in MAPE
        valid_mape = (np.abs(y_true) > 1e-8)
        if np.any(valid_mape):
            mape = np.mean(np.abs((y_true[valid_mape] - y_pred[valid_mape]) / y_true[valid_mape])) * 100
        else:
            mape = np.nan
        
        metrics = {
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MDAE': mdae, 'MAPE': mape,
            'Mean Error': np.mean(y_true - y_pred),
            'Std Error': np.std(y_true - y_pred)
        }
        
        if y_std is not None and len(y_std) > 0:
            y_std = np.array(y_std).flatten()[valid_mask]
            lower_bound = y_pred - 1.96 * y_std
            upper_bound = y_pred + 1.96 * y_std
            coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
            sharpness = np.mean(2 * 1.96 * y_std)
            metrics['Coverage'] = coverage
            metrics['Sharpness'] = sharpness
        
        self.logger.info(f"📊 {label} Evaluation:")
        for k, v in metrics.items():
            self.logger.info(f"  ✅ {k}: {v:.4f}" if not math.isnan(v) else f"  ✅ {k}: NaN")
        
        return metrics
    
    def train_model(self, X_train, y_train, X_val, y_val, model, model_name, target, label):
        self.logger.info(f"Training {model_name} for {target} on {label}...")
        
        # Compile the model if it's not a PINNModel
        if not isinstance(model, PINNModel):
            model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=Adam(0.001), loss='mse')
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', 
                         patience=self.config['early_stopping_patience'], 
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', 
                             factor=0.5, 
                             patience=self.config['reduce_lr_patience'], 
                             min_lr=self.config['min_learning_rate'])
        ]
        
        # Train the model
        start = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start
        
        # Get predictions
        if 'Bayesian' in model_name or 'GRU' in model_name or 'LSTM' in model_name:
            def predict_with_dropout(model, x, n_samples):
                """Perform Monte Carlo dropout for uncertainty estimation"""
                model.training = True
                preds = []
                for _ in tqdm(range(n_samples), desc=f"Bayesian inference ({model_name})"):
                    preds.append(model(x, training=True))
                model.training = False
                return np.array(preds)
            
            preds = predict_with_dropout(model, X_val, self.config['bayesian_samples'])
            y_pred = np.mean(preds, axis=0)
            y_std = np.std(preds, axis=0)
        else:
            y_pred = model.predict(X_val, verbose=0)
            y_std = None
        
        self.logger.info(f"Training completed in {training_time:.2f}s")
        return model, history, y_pred, y_std
    
    def train_meta_ensemble(self, model_predictions, y_actual, time_seq,
                       sequence_length, target, label):
        all_model_names = list(model_predictions.keys())
        num_models = len(all_model_names)
        self.logger.info(f"Training meta-ensemble for {target} with {num_models} models")
        
        # Find valid indices (after sequence_length)
        valid_indices = np.arange(sequence_length - 1, len(y_actual))
        n_valid = len(valid_indices)
        
        # Check if we have enough data for meta-ensemble
        min_required = max(self.config['min_ensemble_data'], self.config['batch_size'])
        if n_valid < min_required:
            self.logger.error(f"Not enough valid data points ({n_valid}) for meta-ensemble training. Need at least {min_required}.")
            return None
        
        # Initialize arrays for meta-ensemble input
        X_ensemble = np.zeros((n_valid, sequence_length, num_models))
        U_ensemble = np.zeros((n_valid, sequence_length, num_models))
        
        # Populate the meta-ensemble input arrays
        for i, model_name in enumerate(all_model_names):
            pred, std = model_predictions[model_name]
            
            # Handle cases where std is None
            if std is None:
                std = np.zeros_like(pred)
            
            for j_idx, j in enumerate(valid_indices):
                start_idx = j - sequence_length + 1
                
                # Handle cases where we don't have enough history
                if start_idx < 0:
                    # Pad with zeros at the beginning
                    pad_length = -start_idx
                    available_preds = pred[0:j+1]
                    available_stds = std[0:j+1]
                    
                    # Pad with zeros to make the sequence the correct length
                    padded_preds = np.pad(available_preds, (pad_length, 0), mode='constant')
                    padded_stds = np.pad(available_stds, (pad_length, 0), mode='constant')
                    
                    X_ensemble[j_idx, :, i] = padded_preds
                    U_ensemble[j_idx, :, i] = padded_stds
                else:
                    X_ensemble[j_idx, :, i] = pred[start_idx:j+1]
                    U_ensemble[j_idx, :, i] = std[start_idx:j+1]
        
        # Create meta-ensemble model
        meta_ensemble = self.create_meta_ensemble_model(
            sequence_length=sequence_length,
            num_models=num_models,
            uncertainty_aware=True
        )
        
        # Prepare targets
        y_ensemble = y_actual[valid_indices].reshape(-1, 1)
        
        # Split into train/validation sets
        split_idx = int(0.8 * n_valid)
        if split_idx < min_required or (n_valid - split_idx) < min_required:
            self.logger.warning("Not enough data for proper train/validation split in meta-ensemble")
            split_idx = max(min_required, n_valid // 2)
        
        X_train, X_val = X_ensemble[:split_idx], X_ensemble[split_idx:]
        U_train, U_val = U_ensemble[:split_idx], U_ensemble[split_idx:]
        y_train, y_val = y_ensemble[:split_idx], y_ensemble[split_idx:]
        
        # Verify sequence lengths
        assert X_train.shape[1] == sequence_length, f"X_train sequence length mismatch: expected {sequence_length}, got {X_train.shape[1]}"
        assert X_val.shape[1] == sequence_length, f"X_val sequence length mismatch: expected {sequence_length}, got {X_val.shape[1]}"
        assert U_train.shape[1] == sequence_length, f"U_train sequence length mismatch: expected {sequence_length}, got {U_train.shape[1]}"
        assert U_val.shape[1] == sequence_length, f"U_val sequence length mismatch: expected {sequence_length}, got {U_val.shape[1]}"
        
        # Set up callbacks
        callbacks = [
            EarlyStopping(monitor='val_prediction_loss',
                         patience=self.config['ensemble_patience'],
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_prediction_loss',
                             factor=0.5,
                             patience=max(1, self.config['ensemble_patience']//3))
        ]
        
        self.logger.info(f"Training meta-ensemble model for {target} on {label}...")
        
        # FIX: Compile the model before training
        meta_ensemble.compile(
            optimizer=Adam(0.001),
            loss={
                'prediction': 'mse',
                'confidence': 'binary_crossentropy'
            },
            loss_weights={
                'prediction': 1.0,
                'confidence': 0.1
            },
            metrics={'prediction': ['mae', 'mse']}
        )
        
        # Train the meta-ensemble
        history = meta_ensemble.fit(
            [X_train, U_train],
            {'prediction': y_train, 'confidence': np.ones_like(y_train)},
            validation_data=([X_val, U_val],
                            {'prediction': y_val, 'confidence': np.ones_like(y_val)}),
            epochs=self.config['ensemble_epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        return meta_ensemble
    
    def generate_ensemble_predictions(self, ensemble_model, model_predictions, time_seq):
        all_model_names = list(model_predictions.keys())
        num_models = len(all_model_names)
        sequence_length = self.config['sequence_length']
        
        # Find valid indices (after sequence_length)
        valid_indices = np.arange(sequence_length - 1, len(time_seq))
        n_valid = len(valid_indices)
        
        if n_valid <= 0:
            self.logger.error("Not enough data points to generate ensemble predictions")
            return np.full(len(time_seq), np.nan), np.full(len(time_seq), np.nan)
        
        # Initialize arrays for meta-ensemble input
        X_ensemble = np.zeros((n_valid, sequence_length, num_models))
        U_ensemble = np.zeros((n_valid, sequence_length, num_models))
        
        # Populate the meta-ensemble input arrays
        for i, model_name in enumerate(all_model_names):
            pred, std = model_predictions[model_name]
            
            # Handle cases where std is None
            if std is None:
                std = np.zeros_like(pred)
            
            for j_idx, j in enumerate(valid_indices):
                start_idx = j - sequence_length + 1
                
                # Handle cases where we don't have enough history
                if start_idx < 0:
                    # Pad with zeros at the beginning
                    pad_length = -start_idx
                    available_preds = pred[0:j+1]
                    available_stds = std[0:j+1]
                    
                    # Pad with zeros to make the sequence the correct length
                    padded_preds = np.pad(available_preds, (pad_length, 0), mode='constant')
                    padded_stds = np.pad(available_stds, (pad_length, 0), mode='constant')
                    
                    X_ensemble[j_idx, :, i] = padded_preds
                    U_ensemble[j_idx, :, i] = padded_stds
                else:
                    X_ensemble[j_idx, :, i] = pred[start_idx:j+1]
                    U_ensemble[j_idx, :, i] = std[start_idx:j+1]
        
        # Get ensemble predictions with uncertainty
        ensemble_pred, ensemble_uncert = ensemble_model.predict_with_uncertainty(
            X_ensemble, U_ensemble, n_samples=30
        )
        
        # Create full prediction arrays with NaNs for invalid indices
        full_pred = np.full(len(time_seq), np.nan)
        full_uncert = np.full(len(time_seq), np.nan)
        
        # Fill in valid predictions
        full_pred[valid_indices] = ensemble_pred.flatten()
        full_uncert[valid_indices] = ensemble_uncert.flatten()
        
        return full_pred, full_uncert
    
    def train_and_evaluate(self, file_path, label):
        result = self.process_data(file_path, label)
        if result is None:
            return None, None, None, None, None
        
        X_seq, y_volt, y_temp, time_plot, scalers, _ = result
        
        # Split data into train/validation
        split_idx = int((1 - self.config['validation_split']) * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_volt_train, y_volt_val = y_volt[:split_idx], y_volt[split_idx:]
        y_temp_train, y_temp_val = y_temp[:split_idx], y_temp[split_idx:]
        time_val = time_plot[split_idx:]
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.logger.info(f"Input shape: {input_shape}")
        
        # Initialize metrics and predictions dictionaries
        volt_metrics, volt_preds, volt_hist = {}, {}, {}
        temp_metrics, temp_preds, temp_hist = {}, {}, {}
        
        # Voltage training - Transformer
        model_transformer_v = self.create_transformer_model(input_shape)
        model_transformer_v, hist_transformer_v, pred_transformer_v, std_transformer_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_transformer_v, 'Transformer', 'Voltage', label)
        volt_hist['Transformer'] = hist_transformer_v
        
        # Scale back to original units
        scaler_v = scalers[0]
        y_actual_v = y_volt_val
        pred_actual_v = pred_transformer_v.flatten()
        std_actual_v = std_transformer_v.flatten() * scaler_v.scale_[0] if std_transformer_v is not None and scaler_v is not None else None
        
        metrics_transformer_v = self.evaluate_model(y_actual_v, pred_actual_v, std_actual_v, f"Transformer {label} Voltage")
        volt_metrics['Transformer'] = metrics_transformer_v
        volt_preds['Transformer'] = (pred_actual_v, std_actual_v)
        
        # Voltage training - Hybrid CNN-BiLSTM
        model_cnn_bilstm_v = self.create_hybrid_cnn_bilstm_model(input_shape)
        model_cnn_bilstm_v, hist_cnn_bilstm_v, pred_cnn_bilstm_v, std_cnn_bilstm_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_cnn_bilstm_v, 'Hybrid CNN-BiLSTM', 'Voltage', label)
        volt_hist['Hybrid CNN-BiLSTM'] = hist_cnn_bilstm_v
        
        pred_actual_v_cnn = pred_cnn_bilstm_v.flatten()
        std_actual_v_cnn = std_cnn_bilstm_v.flatten() * scaler_v.scale_[0] if std_cnn_bilstm_v is not None and scaler_v is not None else None
        metrics_cnn_bilstm_v = self.evaluate_model(y_actual_v, pred_actual_v_cnn, std_actual_v_cnn, f"Hybrid CNN-BiLSTM {label} Voltage")
        volt_metrics['Hybrid CNN-BiLSTM'] = metrics_cnn_bilstm_v
        volt_preds['Hybrid CNN-BiLSTM'] = (pred_actual_v_cnn, std_actual_v_cnn)
        
        # Voltage training - Bayesian LSTM
        model_bayes_v = self.create_bayesian_lstm_model(input_shape)
        model_bayes_v, hist_bayes_v, pred_bayes_v, std_bayes_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_bayes_v, 'Bayesian LSTM', 'Voltage', label)
        volt_hist['Bayesian LSTM'] = hist_bayes_v
        
        pred_actual_v_bayes = pred_bayes_v.flatten()
        std_actual_v_bayes = std_bayes_v.flatten() * scaler_v.scale_[0] if std_bayes_v is not None and scaler_v is not None else None
        metrics_bayes_v = self.evaluate_model(y_actual_v, pred_actual_v_bayes, std_actual_v_bayes, f"Bayesian LSTM {label} Voltage")
        volt_metrics['Bayesian LSTM'] = metrics_bayes_v
        volt_preds['Bayesian LSTM'] = (pred_actual_v_bayes, std_actual_v_bayes)
        
        # Voltage training - GRU
        model_gru_v = self.create_gru_model(input_shape)
        model_gru_v, hist_gru_v, pred_gru_v, std_gru_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_gru_v, 'GRU', 'Voltage', label)
        volt_hist['GRU'] = hist_gru_v
        
        pred_actual_v_gru = pred_gru_v.flatten()
        std_actual_v_gru = std_gru_v.flatten() * scaler_v.scale_[0] if std_gru_v is not None and scaler_v is not None else None
        metrics_gru_v = self.evaluate_model(y_actual_v, pred_actual_v_gru, std_actual_v_gru, f"GRU {label} Voltage")
        volt_metrics['GRU'] = metrics_gru_v
        volt_preds['GRU'] = (pred_actual_v_gru, std_actual_v_gru)
        
        # Voltage training - BiGRU
        model_bigru_v = self.create_bigru_model(input_shape)
        model_bigru_v, hist_bigru_v, pred_bigru_v, std_bigru_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_bigru_v, 'BiGRU', 'Voltage', label)
        volt_hist['BiGRU'] = hist_bigru_v
        
        pred_actual_v_bigru = pred_bigru_v.flatten()
        std_actual_v_bigru = std_bigru_v.flatten() * scaler_v.scale_[0] if std_bigru_v is not None and scaler_v is not None else None
        metrics_bigru_v = self.evaluate_model(y_actual_v, pred_actual_v_bigru, std_actual_v_bigru, f"BiGRU {label} Voltage")
        volt_metrics['BiGRU'] = metrics_bigru_v
        volt_preds['BiGRU'] = (pred_actual_v_bigru, std_actual_v_bigru)
        
        # Voltage training - TCN
        model_tcn_v = self.create_tcn_model(input_shape)
        model_tcn_v, hist_tcn_v, pred_tcn_v, std_tcn_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_tcn_v, 'TCN', 'Voltage', label)
        volt_hist['TCN'] = hist_tcn_v
        
        pred_actual_v_tcn = pred_tcn_v.flatten()
        std_actual_v_tcn = std_tcn_v.flatten() * scaler_v.scale_[0] if std_tcn_v is not None and scaler_v is not None else None
        metrics_tcn_v = self.evaluate_model(y_actual_v, pred_actual_v_tcn, std_actual_v_tcn, f"TCN {label} Voltage")
        volt_metrics['TCN'] = metrics_tcn_v
        volt_preds['TCN'] = (pred_actual_v_tcn, std_actual_v_tcn)
        
        # Voltage training - PINN
        model_pinn_v = self.create_pinn_model(input_shape, base_model_type="Transformer")
        model_pinn_v.compile(optimizer=Adam(0.001), loss='mse')
        
        callbacks = [
            EarlyStopping(monitor='val_loss', 
                         patience=self.config['early_stopping_patience'], 
                         restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', 
                             factor=0.5, 
                             patience=self.config['reduce_lr_patience'], 
                             min_lr=self.config['min_learning_rate'])
        ]
        
        start = time.time()
        history = model_pinn_v.fit(
            X_train, y_volt_train,
            validation_data=(X_val, y_volt_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start
        
        pred_pinn_v = model_pinn_v.predict(X_val, verbose=0)
        std_pinn_v = None
        volt_hist['PINN'] = history
        
        pred_actual_v_pinn = pred_pinn_v.flatten()
        std_actual_v_pinn = None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_pinn_v = self.evaluate_model(y_actual_v, pred_actual_v_pinn, std_actual_v_pinn, f"PINN {label} Voltage")
        volt_metrics['PINN'] = metrics_pinn_v
        volt_preds['PINN'] = (pred_actual_v_pinn, std_actual_v_pinn)
        
        # Voltage training - CNN-GRU
        model_cnn_gru_v = self.create_cnn_gru_model(input_shape)
        model_cnn_gru_v, hist_cnn_gru_v, pred_cnn_gru_v, std_cnn_gru_v = self.train_model(
            X_train, y_volt_train, X_val, y_volt_val, model_cnn_gru_v, 'CNN-GRU', 'Voltage', label)
        volt_hist['CNN-GRU'] = hist_cnn_gru_v
        
        pred_actual_v_cnn_gru = pred_cnn_gru_v.flatten()
        std_actual_v_cnn_gru = std_cnn_gru_v.flatten() * scaler_v.scale_[0] if std_cnn_gru_v is not None and scaler_v is not None else None
        metrics_cnn_gru_v = self.evaluate_model(y_actual_v, pred_actual_v_cnn_gru, std_actual_v_cnn_gru, f"CNN-GRU {label} Voltage")
        volt_metrics['CNN-GRU'] = metrics_cnn_gru_v
        volt_preds['CNN-GRU'] = (pred_actual_v_cnn_gru, std_actual_v_cnn_gru)
        
        # Temperature training - Transformer
        model_transformer_t = self.create_transformer_model(input_shape)
        model_transformer_t, hist_transformer_t, pred_transformer_t, std_transformer_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_transformer_t, 'Transformer', 'Temperature', label)
        temp_hist['Transformer'] = hist_transformer_t
        
        # Scale back to original units
        scaler_t = scalers[1]
        y_actual_t = y_temp_val
        pred_actual_t = pred_transformer_t.flatten()
        std_actual_t = std_transformer_t.flatten() * scaler_t.scale_[0] if std_transformer_t is not None and scaler_t is not None else None
        
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_transformer_t = self.evaluate_model(y_actual_t, pred_actual_t, std_actual_t, f"Transformer {label} Temperature")
        temp_metrics['Transformer'] = metrics_transformer_t
        temp_preds['Transformer'] = (pred_actual_t, std_actual_t)
        
        # Temperature training - Hybrid CNN-BiLSTM
        model_cnn_bilstm_t = self.create_hybrid_cnn_bilstm_model(input_shape)
        model_cnn_bilstm_t, hist_cnn_bilstm_t, pred_cnn_bilstm_t, std_cnn_bilstm_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_cnn_bilstm_t, 'Hybrid CNN-BiLSTM', 'Temperature', label)
        temp_hist['Hybrid CNN-BiLSTM'] = hist_cnn_bilstm_t
        
        pred_actual_t_cnn = pred_cnn_bilstm_t.flatten()
        std_actual_t_cnn = std_cnn_bilstm_t.flatten() * scaler_t.scale_[0] if std_cnn_bilstm_t is not None and scaler_t is not None else None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_cnn_bilstm_t = self.evaluate_model(y_actual_t, pred_actual_t_cnn, std_actual_t_cnn, f"Hybrid CNN-BiLSTM {label} Temperature")
        temp_metrics['Hybrid CNN-BiLSTM'] = metrics_cnn_bilstm_t
        temp_preds['Hybrid CNN-BiLSTM'] = (pred_actual_t_cnn, std_actual_t_cnn)
        
        # Temperature training - Bayesian LSTM
        model_bayes_t = self.create_bayesian_lstm_model(input_shape)
        model_bayes_t, hist_bayes_t, pred_bayes_t, std_bayes_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_bayes_t, 'Bayesian LSTM', 'Temperature', label)
        temp_hist['Bayesian LSTM'] = hist_bayes_t
        
        pred_actual_t_bayes = pred_bayes_t.flatten()
        std_actual_t_bayes = std_bayes_t.flatten() * scaler_t.scale_[0] if std_bayes_t is not None and scaler_t is not None else None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_bayes_t = self.evaluate_model(y_actual_t, pred_actual_t_bayes, std_actual_t_bayes, f"Bayesian LSTM {label} Temperature")
        temp_metrics['Bayesian LSTM'] = metrics_bayes_t
        temp_preds['Bayesian LSTM'] = (pred_actual_t_bayes, std_actual_t_bayes)
        
        # Temperature training - GRU
        model_gru_t = self.create_gru_model(input_shape)
        model_gru_t, hist_gru_t, pred_gru_t, std_gru_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_gru_t, 'GRU', 'Temperature', label)
        temp_hist['GRU'] = hist_gru_t
        
        pred_actual_t_gru = pred_gru_t.flatten()
        std_actual_t_gru = std_gru_t.flatten() * scaler_t.scale_[0] if std_gru_t is not None and scaler_t is not None else None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_gru_t = self.evaluate_model(y_actual_t, pred_actual_t_gru, std_actual_t_gru, f"GRU {label} Temperature")
        temp_metrics['GRU'] = metrics_gru_t
        temp_preds['GRU'] = (pred_actual_t_gru, std_actual_t_gru)
        
        # Temperature training - BiGRU
        model_bigru_t = self.create_bigru_model(input_shape)
        model_bigru_t, hist_bigru_t, pred_bigru_t, std_bigru_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_bigru_t, 'BiGRU', 'Temperature', label)
        temp_hist['BiGRU'] = hist_bigru_t
        
        pred_actual_t_bigru = pred_bigru_t.flatten()
        std_actual_t_bigru = std_bigru_t.flatten() * scaler_t.scale_[0] if std_bigru_t is not None and scaler_t is not None else None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_bigru_t = self.evaluate_model(y_actual_t, pred_actual_t_bigru, std_actual_t_bigru, f"BiGRU {label} Temperature")
        temp_metrics['BiGRU'] = metrics_bigru_t
        temp_preds['BiGRU'] = (pred_actual_t_bigru, std_actual_t_bigru)
        
        # Temperature training - TCN
        model_tcn_t = self.create_tcn_model(input_shape)
        model_tcn_t, hist_tcn_t, pred_tcn_t, std_tcn_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_tcn_t, 'TCN', 'Temperature', label)
        temp_hist['TCN'] = hist_tcn_t
        
        pred_actual_t_tcn = pred_tcn_t.flatten()
        std_actual_t_tcn = std_tcn_t.flatten() * scaler_t.scale_[0] if std_tcn_t is not None and scaler_t is not None else None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_tcn_t = self.evaluate_model(y_actual_t, pred_actual_t_tcn, std_actual_t_tcn, f"TCN {label} Temperature")
        temp_metrics['TCN'] = metrics_tcn_t
        temp_preds['TCN'] = (pred_actual_t_tcn, std_actual_t_tcn)
        
        # Temperature training - PINN
        model_pinn_t = self.create_pinn_model(input_shape, base_model_type="Transformer")
        model_pinn_t.compile(optimizer=Adam(0.001), loss='mse')
        
        history = model_pinn_t.fit(
            X_train, y_temp_train,
            validation_data=(X_val, y_temp_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        pred_pinn_t = model_pinn_t.predict(X_val, verbose=0)
        std_pinn_t = None
        temp_hist['PINN'] = history
        
        pred_actual_t_pinn = pred_pinn_t.flatten()
        std_actual_t_pinn = None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_pinn_t = self.evaluate_model(y_actual_t, pred_actual_t_pinn, std_actual_t_pinn, f"PINN {label} Temperature")
        temp_metrics['PINN'] = metrics_pinn_t
        temp_preds['PINN'] = (pred_actual_t_pinn, std_actual_t_pinn)
        
        # Temperature training - CNN-GRU
        model_cnn_gru_t = self.create_cnn_gru_model(input_shape)
        model_cnn_gru_t, hist_cnn_gru_t, pred_cnn_gru_t, std_cnn_gru_t = self.train_model(
            X_train, y_temp_train, X_val, y_temp_val, model_cnn_gru_t, 'CNN-GRU', 'Temperature', label)
        temp_hist['CNN-GRU'] = hist_cnn_gru_t
        
        pred_actual_t_cnn_gru = pred_cnn_gru_t.flatten()
        std_actual_t_cnn_gru = std_cnn_gru_t.flatten() * scaler_t.scale_[0] if std_cnn_gru_t is not None and scaler_t is not None else None
        # FIX: Fixed f-string syntax error - kept on single line
        metrics_cnn_gru_t = self.evaluate_model(y_actual_t, pred_actual_t_cnn_gru, std_actual_t_cnn_gru, f"CNN-GRU {label} Temperature")
        temp_metrics['CNN-GRU'] = metrics_cnn_gru_t
        temp_preds['CNN-GRU'] = (pred_actual_t_cnn_gru, std_actual_t_cnn_gru)
        
        # Train meta-ensemble model for voltage
        self.logger.info(f"Training meta-ensemble model for voltage on {label}...")
        ensemble_volt = self.train_meta_ensemble(
            volt_preds, y_actual_v, time_val,
            sequence_length=self.config['sequence_length'],
            target='Voltage',
            label=label
        )
        
        # Train meta-ensemble model for temperature
        self.logger.info(f"Training meta-ensemble model for temperature on {label}...")
        ensemble_temp = self.train_meta_ensemble(
            temp_preds, y_actual_t, time_val,
            sequence_length=self.config['sequence_length'],
            target='Temperature',
            label=label
        )
        
        # Store ensemble models
        if label not in self.ensemble_models:
            self.ensemble_models[label] = {}
        self.ensemble_models[label]['Voltage'] = ensemble_volt
        self.ensemble_models[label]['Temperature'] = ensemble_temp
        
        # Generate ensemble predictions
        if ensemble_volt is not None:
            ensemble_pred_volt, ensemble_uncert_volt = self.generate_ensemble_predictions(
                ensemble_volt, volt_preds, time_val)
        else:
            self.logger.warning(f"Meta-ensemble for voltage on {label} failed, using fallback")
            valid_preds = []
            for pred, std in volt_preds.values():
                if pred is not None:
                    valid_preds.append(pred)
            if valid_preds:
                ensemble_pred_volt = np.mean(valid_preds, axis=0)
                ensemble_uncert_volt = np.std(valid_preds, axis=0)
            else:
                ensemble_pred_volt = np.full_like(y_actual_v, np.nan)
                ensemble_uncert_volt = np.full_like(y_actual_v, np.nan)
        
        if ensemble_temp is not None:
            ensemble_pred_temp, ensemble_uncert_temp = self.generate_ensemble_predictions(
                ensemble_temp, temp_preds, time_val)
        else:
            self.logger.warning(f"Meta-ensemble for temperature on {label} failed, using fallback")
            valid_preds = []
            for pred, std in temp_preds.values():
                if pred is not None:
                    valid_preds.append(pred)
            if valid_preds:
                ensemble_pred_temp = np.mean(valid_preds, axis=0)
                ensemble_uncert_temp = np.std(valid_preds, axis=0)
            else:
                ensemble_pred_temp = np.full_like(y_actual_t, np.nan)
                ensemble_uncert_temp = np.full_like(y_actual_t, np.nan)
        
        # Evaluate ensemble model
        metrics_ensemble_volt = self.evaluate_model(
            y_actual_v, ensemble_pred_volt, ensemble_uncert_volt,
            f"Ensemble {label} Voltage")
        metrics_ensemble_temp = self.evaluate_model(
            y_actual_t, ensemble_pred_temp, ensemble_uncert_temp,
            f"Ensemble {label} Temperature")
        
        # Add ensemble to metrics and predictions
        volt_metrics['Ensemble'] = metrics_ensemble_volt
        temp_metrics['Ensemble'] = metrics_ensemble_temp
        volt_preds['Ensemble'] = (ensemble_pred_volt, ensemble_uncert_volt)
        temp_preds['Ensemble'] = (ensemble_pred_temp, ensemble_uncert_temp)
        
        self.model_history[label] = {'Voltage': volt_hist, 'Temperature': temp_hist}
        
        return {'Voltage': volt_metrics, 'Temperature': temp_metrics}, \
               {'Voltage': volt_preds, 'Temperature': temp_preds}, \
               time_val, \
               y_actual_v, \
               y_actual_t
    
    def save_results(self):
        self.logger.info("Saving results to disk...")
        results_dir = os.path.join(self.config['output_directory'], 'results')
        plots_dir = os.path.join(results_dir, 'plots')
        preds_dir = os.path.join(results_dir, 'predictions')
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(preds_dir, exist_ok=True)
        
        # Save Metrics to CSV
        metrics_data = []
        for label, metrics_dict in self.all_metrics.items():
            for target, models in metrics_dict.items():
                for model_name, metrics in models.items():
                    row = {
                        'Discharge_Rate': label,
                        'Target': target,
                        'Model': model_name,
                    }
                    row.update(metrics)
                    metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = os.path.join(results_dir, 'evaluation_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        self.logger.info(f"📊 Metrics saved to: {metrics_path}")
        
        # Save Predictions & Plots
        for label in self.all_metrics.keys():
            if label not in self.all_predictions:
                continue
            
            time_vals = self.all_times[label]
            actual_volt = self.all_actuals[label]['Voltage']
            actual_temp = self.all_actuals[label]['Temperature']
            
            for model_name in self.model_names + ['Ensemble']:
                if model_name not in self.all_predictions[label]['Voltage'] or \
                   model_name not in self.all_predictions[label]['Temperature']:
         
