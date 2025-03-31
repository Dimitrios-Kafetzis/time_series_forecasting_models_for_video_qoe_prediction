#!/usr/bin/env python3
"""
Filename: test_models.py
Author: Dimitrios Kafetzis
Creation Date: 2025-02-04
Description:
    This script loads a saved trained model and a reserved test dataset,
    computes predictions on the test data, and evaluates the model using
    regression metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE),
    and the R² score.

    It supports multiple dataset formats:
      1. Regular: One JSON file per 10-second interval with keys:
         "packet_loss_rate", "jitter", "throughput", "speed", "QoE", "timestamp",
         (and optionally additional temporal features such as "hour", "minute", "day_of_week").
      2. Augmented: One JSON file per 5‑second window with five 1‑second measurements
         (flattened into features) and aggregated QoE. Extra summary statistics may be
         included if the --use_stats flag is provided.

    It supports evaluation of models including:
         - Linear Regressor
         - Simple DNN
         - LSTM
         - GRU
         - Transformer
         - LSTM with Self-Attention
         - GRU with Self-Attention

    In addition to the evaluation metrics, this script measures the inference
    latency and estimates latencies on other hardware based on a scaling factor.
    
Usage Examples:
    Regular mode:
      $ python3 test_models.py --data_folder ./mock_dataset --model_file model_transformer.h5 --seq_length 5 --scaler_file scaler.save

    Testing Linear Regressor model:
      $ python3 test_models.py --data_folder ./mock_dataset --model_file linear_basic.h5 --seq_length 5 --scaler_file scaler.save
      
    Testing Simple DNN model:
      $ python3 test_models.py --data_folder ./mock_dataset --model_file dnn_with_elu.h5 --seq_length 5 --scaler_file scaler.save

    Augmented mode (without extra stats):
      $ python3 test_models.py --data_folder ./augmented_dataset --model_file model_lstm.h5 --seq_length 5 --scaler_file scaler.save --augmented

    Augmented mode (with extra stats):
      $ python3 test_models.py --data_folder ./augmented_dataset --model_file model_gru.h5 --seq_length 5 --scaler_file scaler.save --augmented --use_stats

    Testing models with self-attention:
      $ python3 test_models.py --data_folder ./augmented_dataset --model_file ./forecasting_models_v5/lstm_basic.h5 --seq_length 5 --scaler_file ./forecasting_models_v5/scaler.save --augmented

    To simulate inference latency on a different device (e.g., "xeon" or "jetson"):
      $ python3 test_models.py --data_folder ./augmented_dataset --model_file model_gru.h5 --seq_length 5 --scaler_file scaler.save --augmented --simulate_device xeon
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import argparse
import tensorflow as tf
from datetime import datetime
from time import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow.keras.backend as K
import sys

# Import functions from the forecasting models module
try:
    from timeseries_forecasting_models_v5 import (
        TransformerBlock, SelfAttention, 
        load_dataset_from_folder, load_augmented_dataset_from_folder,
        preprocess_dataframe, create_sequences
    )
    print("Successfully imported from timeseries_forecasting_models_v5")
except ImportError:
    try:
        from timeseries_forecasting_models_v2 import TransformerBlock, load_dataset_from_folder, load_augmented_dataset_from_folder
        print("Successfully imported from timeseries_forecasting_models_v2")
    except ImportError:
        print("Warning: Could not import from forecasting models modules, using built-in implementations...")

# Define the SelfAttention layer class for loading models with attention
class SelfAttention(tf.keras.layers.Layer):
    """
    Custom Self-Attention Layer
    This layer applies attention over the time steps of a sequence, allowing the model
    to focus on the most relevant parts of the time series for prediction.
    """
    def __init__(self, attention_units=128, **kwargs):
        self.attention_units = attention_units
        super(SelfAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # input_shape = (batch_size, time_steps, features)
        self.time_steps = input_shape[1]
        self.features = input_shape[2]
        
        # Dense layer to compute attention scores
        self.W_attention = self.add_weight(name='W_attention',
                                          shape=(self.features, self.attention_units),
                                          initializer='glorot_uniform',
                                          trainable=True)
        
        self.b_attention = self.add_weight(name='b_attention',
                                         shape=(self.attention_units,),
                                         initializer='zeros',
                                         trainable=True)
        
        # Context vector to compute attention weights
        self.u_attention = self.add_weight(name='u_attention',
                                          shape=(self.attention_units, 1),
                                          initializer='glorot_uniform',
                                          trainable=True)
        
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        
        # Step 1: Compute attention scores
        # (batch_size, time_steps, features) @ (features, attention_units) = (batch_size, time_steps, attention_units)
        score = tf.tanh(tf.tensordot(inputs, self.W_attention, axes=[[2], [0]]) + self.b_attention)
        
        # Step 2: Compute attention weights
        # (batch_size, time_steps, attention_units) @ (attention_units, 1) = (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u_attention, axes=[[2], [0]]), axis=1)
        
        # Step 3: Apply attention weights to input sequence
        # (batch_size, time_steps, 1) * (batch_size, time_steps, features) = (batch_size, time_steps, features)
        context_vector = attention_weights * inputs
        
        # Step 4: Sum over time dimension to get weighted representation
        # (batch_size, features)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'attention_units': self.attention_units,
        })
        return config

# Implementation of data loading functions in case they can't be imported
def load_dataset_from_folder_fallback(folder_path):
    """
    Load all JSON files from the folder (regular format) and return a DataFrame.
    Each JSON file is expected to have the keys:
      "packet_loss_rate", "jitter", "throughput", "speed", "QoE", "timestamp",
      and (optionally) additional temporal fields ("hour", "minute", "day_of_week").
    """
    data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                data.append(json_data)
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    df = pd.DataFrame(data_sorted)
    return df

def load_augmented_dataset_from_folder_fallback(folder_path, use_stats=False):
    """
    Load all JSON files from the folder (augmented format) and return a DataFrame.
    In each JSON file, the keys that are not "QoE" or "timestamp" represent 1-second measurements.
    These are sorted and flattened into feature columns f0, f1, ..., f19 (for 5 seconds × 4 features).
    If use_stats is True, additional statistics (mean, std, min, max for each feature) are computed.
    """
    data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            inner_keys = [k for k in json_data.keys() if k not in ["QoE", "timestamp"]]
            inner_keys = sorted(inner_keys)
            flat_features = []
            if use_stats:
                stats_features = {"packet_loss_rate": [], "jitter": [], "throughput": [], "speed": []}
            for key in inner_keys:
                entry = json_data[key]
                flat_features.extend([entry["packet_loss_rate"], entry["jitter"], entry["throughput"], entry["speed"]])
                if use_stats:
                    stats_features["packet_loss_rate"].append(entry["packet_loss_rate"])
                    stats_features["jitter"].append(entry["jitter"])
                    stats_features["throughput"].append(entry["throughput"])
                    stats_features["speed"].append(entry["speed"])
            sample = {}
            for i, val in enumerate(flat_features):
                sample[f"f{i}"] = val
            if use_stats:
                for feature in ["packet_loss_rate", "jitter", "throughput", "speed"]:
                    arr = np.array(stats_features[feature])
                    sample[f"{feature}_mean"] = float(np.mean(arr))
                    sample[f"{feature}_std"] = float(np.std(arr))
                    sample[f"{feature}_min"] = float(np.min(arr))
                    sample[f"{feature}_max"] = float(np.max(arr))
            sample["QoE"] = json_data["QoE"]
            sample["timestamp"] = json_data["timestamp"]
            data.append(sample)
    data_sorted = sorted(data, key=lambda x: x["timestamp"])
    df = pd.DataFrame(data_sorted)
    return df

def measure_inference_latency(model, sample_input, num_runs=100):
    # Warm-up run
    model.predict(sample_input)
    times = []
    for _ in range(num_runs):
        start_time = time()
        model.predict(sample_input)
        end_time = time()
        times.append(end_time - start_time)
    avg_latency = np.mean(times)
    return avg_latency

def preprocess_dataframe_fallback(df):
    """
    Convert columns to numeric types and convert 'timestamp' to a datetime object.
    """
    numeric_cols = [col for col in df.columns if col != 'timestamp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def create_sequences_fallback(df, seq_length=5, feature_cols=None, target_col='QoE'):
    """
    Build sequences of shape (seq_length, number_of_features) and corresponding targets.
    The target is the QoE value at the time step immediately after the sequence.
    """
    X, y = [], []
    for i in range(len(df) - seq_length):
        seq_X = df.iloc[i:i+seq_length][feature_cols].values
        seq_y = df.iloc[i+seq_length][target_col]
        X.append(seq_X)
        y.append(seq_y)
    return np.array(X), np.array(y)

def detect_model_type(model_file):
    """
    Detect the model type from the filename to provide additional information.
    """
    filename = os.path.basename(model_file).lower()
    
    if 'linear' in filename:
        return "Linear Regressor"
    elif 'dnn' in filename:
        return "Simple DNN"
    elif 'lstm' in filename:
        return "LSTM"
    elif 'gru' in filename:
        return "GRU"
    elif 'transformer' in filename:
        return "Transformer"
    else:
        return "Unknown"

def get_model_expected_shape(model):
    """
    Analyzes the model to determine its expected input shape
    """
    # Try to get the input shape from the model
    try:
        input_shape = model.layers[0].input_shape
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape
    except:
        return None

def inspect_model_layer_shapes(model):
    """
    Print the input and output shapes of each layer in the model for debugging
    """
    print("\nModel Layer Shapes:")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'input_shape') and hasattr(layer, 'output_shape'):
            print(f"Layer {i}: {layer.name}")
            print(f"  Input shape: {layer.input_shape}")
            print(f"  Output shape: {layer.output_shape}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing JSON files.")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the saved trained model file.")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Sequence length (number of time steps used as input).")
    parser.add_argument("--scaler_file", type=str, required=True,
                        help="Path to the saved scaler file.")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Proportion of data to reserve for testing (default: 0.2).")
    # Options for augmented dataset mode and extra statistical features.
    parser.add_argument("--augmented", action="store_true",
                        help="Indicate that the dataset is in augmented mode (each file covers 5 seconds with 1-second granularity).")
    parser.add_argument("--use_stats", action="store_true",
                        help="(Only valid with --augmented) Include extra statistical features computed from the inner 5-second window.")
    # New argument to simulate latency on other devices.
    parser.add_argument("--simulate_device", type=str, default="current",
                        help="Simulate inference latency for a given device. Options: current, xeon, jetson")
    # New argument for detailed information about model structure
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information about the model and results")
    args = parser.parse_args()

    # Print information about the test
    print(f"Testing model: {args.model_file}")
    model_type = detect_model_type(args.model_file)
    print(f"Detected model type: {model_type}")

    # Try to use the imported functions, fall back to local implementations if needed
    try:
        if args.augmented:
            print("Loading augmented dataset from:", args.data_folder)
            df = load_augmented_dataset_from_folder(args.data_folder, use_stats=args.use_stats)
        else:
            print("Loading regular dataset from:", args.data_folder)
            df = load_dataset_from_folder(args.data_folder)
    except NameError:
        print("Using fallback data loading functions")
        if args.augmented:
            print("Loading augmented dataset from:", args.data_folder)
            df = load_augmented_dataset_from_folder_fallback(args.data_folder, use_stats=args.use_stats)
        else:
            print("Loading regular dataset from:", args.data_folder)
            df = load_dataset_from_folder_fallback(args.data_folder)

    # In augmented mode, features are all columns except 'QoE' and 'timestamp'
    if args.augmented:
        feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
    else:
        feature_cols = ['packet_loss_rate', 'jitter', 'throughput', 'speed']

    # Define normalization columns (features + target).
    norm_cols = feature_cols + ["QoE"]
    
    # Preprocess the DataFrame
    try:
        df = preprocess_dataframe(df)
    except NameError:
        df = preprocess_dataframe_fallback(df)
    
    # Load the scaler (fitted during training) and apply the transformation.
    scaler = joblib.load(args.scaler_file)
    df[norm_cols] = scaler.transform(df[norm_cols])

    # Create sequences from the data.
    try:
        X, y = create_sequences(df, seq_length=args.seq_length, feature_cols=feature_cols, target_col='QoE')
    except NameError:
        X, y = create_sequences_fallback(df, seq_length=args.seq_length, feature_cols=feature_cols, target_col='QoE')
        
    print("Total sequences:", X.shape[0])

    # Reserve the last portion of the sequences for testing.
    test_size = int(len(X) * args.test_ratio)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    print("Test samples:", X_test.shape[0])

    # Load the trained model with custom objects dictionary
    # Add SelfAttention to the custom_objects dictionary for loading self-attention models
    try:
        # Try to import TransformerBlock from the module
        custom_objects = {"TransformerBlock": TransformerBlock, "SelfAttention": SelfAttention}
    except NameError:
        # Fall back to only SelfAttention if TransformerBlock is not available
        custom_objects = {"SelfAttention": SelfAttention}
    
    # Load the model
    print(f"Loading model from {args.model_file}...")
    try:
        model = tf.keras.models.load_model(args.model_file, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
    
    # Print model summary if verbose
    if args.verbose:
        print("\nModel Summary:")
        model.summary()
        inspect_model_layer_shapes(model)

    # Check input shape compatibility
    expected_shape = get_model_expected_shape(model)
    if expected_shape:
        expected_features = expected_shape[-1]
        actual_features = X_test.shape[-1]
        
        if expected_features != actual_features:
            print(f"\nWARNING: Input shape mismatch!")
            print(f"Model expects {expected_features} features, but data has {actual_features} features")
            print(f"This usually happens when the model was trained with or without --use_stats")
            print(f"Current --use_stats setting: {args.use_stats}")
            print("You may need to run the test with the correct --use_stats setting for this model")
            print("\nExiting test to prevent invalid results.")
            sys.exit(1)

    # Evaluate the model on the test set.
    print("\nEvaluating model...")
    try:
        test_loss = model.evaluate(X_test, y_test, verbose=1)
        predictions = model.predict(X_test, verbose=1)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nEvaluation Metrics:")
    print("Test Loss:", test_loss)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # Convert scaled metrics back to original scale, if desired
    if args.verbose:
        # Create dummy arrays to invert the scaling
        dummy_y_test = np.zeros((len(y_test), len(norm_cols)))
        dummy_y_test[:, -1] = y_test
        
        dummy_pred = np.zeros((len(predictions), len(norm_cols)))
        dummy_pred[:, -1] = predictions.flatten()
        
        # Invert scaling
        y_test_orig = scaler.inverse_transform(dummy_y_test)[:, -1]
        pred_orig = scaler.inverse_transform(dummy_pred)[:, -1]
        
        # Calculate metrics in original scale
        mse_orig = mean_squared_error(y_test_orig, pred_orig)
        mae_orig = mean_absolute_error(y_test_orig, pred_orig)
        
        print("\nMetrics in Original Scale:")
        print(f"Mean Squared Error (MSE): {mse_orig:.4f}")
        print(f"Mean Absolute Error (MAE): {mae_orig:.4f}")
        
        # Print range of values for context
        print(f"Original Target Range: {np.min(y_test_orig):.4f} to {np.max(y_test_orig):.4f}")

    # --- Inference Latency Measurement ---
    print("\nMeasuring inference latency...")
    # Choose a random sample from the test set.
    sample_input = X_test[np.random.randint(0, X_test.shape[0])].reshape(1, args.seq_length, len(feature_cols))
    
    try:
        avg_latency = measure_inference_latency(model, sample_input, num_runs=100)
        # Convert latency to milliseconds.
        avg_latency_ms = avg_latency * 1000
        print(f"Average Inference Latency on Current System: {avg_latency_ms:.2f} ms")

        # Simulate latency for other devices based on scaling factors.
        # You can adjust these scaling factors as needed.
        scaling_factors = {
            "current": 1.0,
            "xeon": 0.8,   # 20% faster than current
            "jetson": 5.0  # 5 times slower than current
        }
        if args.simulate_device in scaling_factors:
            estimated_latency = avg_latency_ms * scaling_factors[args.simulate_device]
            print(f"Estimated Inference Latency on {args.simulate_device.capitalize()} Device: {estimated_latency:.2f} ms")
        else:
            print(f"Unknown simulate_device '{args.simulate_device}'. Using current latency.")
    except Exception as e:
        print(f"Error measuring inference latency: {str(e)}")
        avg_latency_ms = float('nan')

    # Provide model-specific interpretation if applicable
    if model_type == "Linear Regressor" and args.verbose:
        try:
            # For linear models, try to extract and display feature weights
            # This assumes the model has a structure where weights are accessible
            weights = model.layers[-1].get_weights()[0]
            bias = model.layers[-1].get_weights()[1]
            
            print("\nLinear Model Weights (Top 10 by magnitude):")
            
            # Create feature names based on sequence structure
            flattened_feature_names = []
            for t in range(args.seq_length):
                for feature in feature_cols:
                    flattened_feature_names.append(f"{feature}_t-{args.seq_length-t}")
            
            # Sort weights by absolute magnitude
            feature_importance = sorted(zip(flattened_feature_names, weights.flatten()), 
                                      key=lambda x: abs(x[1]), reverse=True)
            
            for feature, weight in feature_importance[:10]:
                print(f"{feature}: {weight:.4f}")
                
            print(f"Bias term: {bias[0]:.4f}")
        except Exception as e:
            print(f"Could not extract weights from linear model: {str(e)}")

if __name__ == "__main__":
    main()