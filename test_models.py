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
         - LSTM
         - GRU
         - Transformer
         - Linear Regressor

    In addition to the evaluation metrics, this script measures the inference
    latency and estimates latencies on other hardware based on a scaling factor.
    
Usage Examples:
    Regular mode:
      $ python3 test_models.py --data_folder ./mock_dataset --model_file model_transformer.h5 --seq_length 5 --scaler_file scaler.save

    Augmented mode (without extra stats):
      $ python3 test_models.py --data_folder ./augmented_dataset --model_file model_lstm.h5 --seq_length 5 --scaler_file scaler.save --augmented

    Augmented mode (with extra stats):
      $ python3 test_models.py --data_folder ./augmented_dataset --model_file model_gru.h5 --seq_length 5 --scaler_file scaler.save --augmented --use_stats

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

# Import functions from the forecasting models module
from timeseries_forecasting_models_v2 import TransformerBlock, load_dataset_from_folder, load_augmented_dataset_from_folder

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
    args = parser.parse_args()

    # Load data using the appropriate loader.
    if args.augmented:
        print("Loading augmented dataset from:", args.data_folder)
        df = load_augmented_dataset_from_folder(args.data_folder, use_stats=args.use_stats)
        # In augmented mode, features are all columns except 'QoE' and 'timestamp'
        feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
    else:
        print("Loading regular dataset from:", args.data_folder)
        df = load_dataset_from_folder(args.data_folder)
        feature_cols = ['packet_loss_rate', 'jitter', 'throughput', 'speed']

    # Define normalization columns (features + target).
    norm_cols = feature_cols + ["QoE"]
    # Load the scaler (fitted during training) and apply the transformation.
    scaler = joblib.load(args.scaler_file)
    df[norm_cols] = scaler.transform(df[norm_cols])

    # Ensure the data is sorted by timestamp.
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Create sequences from the data.
    X, y = [], []
    for i in range(len(df) - args.seq_length):
        seq_X = df.iloc[i:i+args.seq_length][feature_cols].values
        seq_y = df.iloc[i+args.seq_length]["QoE"]
        X.append(seq_X)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    print("Total sequences:", X.shape[0])

    # Reserve the last portion of the sequences for testing.
    test_size = int(len(X) * args.test_ratio)
    X_test = X[-test_size:]
    y_test = y[-test_size:]
    print("Test samples:", X_test.shape[0])

    # Load the trained model.
    custom_objects = {"TransformerBlock": TransformerBlock}
    model = tf.keras.models.load_model(args.model_file, custom_objects=custom_objects)

    # Evaluate the model on the test set.
    test_loss = model.evaluate(X_test, y_test)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\nEvaluation Metrics:")
    print("Test Loss (MSE):", test_loss)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    # --- Inference Latency Measurement ---
    # Choose a random sample from the test set.
    sample_input = X_test[np.random.randint(0, X_test.shape[0])].reshape(1, args.seq_length, len(feature_cols))
    avg_latency = measure_inference_latency(model, sample_input, num_runs=100)
    # Convert latency to milliseconds.
    avg_latency_ms = avg_latency * 1000
    print(f"\nAverage Inference Latency on Current System: {avg_latency_ms:.2f} ms")

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

if __name__ == "__main__":
    main()
