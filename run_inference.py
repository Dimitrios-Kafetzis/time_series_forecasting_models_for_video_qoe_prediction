#!/usr/bin/env python3
"""
Filename: run_inference.py
Author: Dimitrios Kafetzis
Creation Date: 2025-02-04
Description:
    This script loads a saved trained model and scaler, loads a new inference JSON file,
    and performs inference by constructing the required input sequence.
    It supports two dataset formats:
      - Standard: one JSON file per 10‑second timepoint with flat structure.
      - Augmented: one JSON file per 5‑second window containing per‑second sub‐records and
                   an aggregated QoE (the overall timestamp is that of the last second in the window).

Usage Examples:
    Standard mode:
      $ python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json \
          --data_folder ./mock_dataset --seq_length 5 --model_file model_transformer.h5 --scaler_file scaler.save

    Augmented mode:
      $ python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json \
          --data_folder ./augmented_dataset --seq_length 5 --model_file model_transformer.h5 --scaler_file scaler.save --augmented
"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Import the custom TransformerBlock (needed when loading the model)
from timeseries_forecasting_models import TransformerBlock

def load_dataset_from_folder(folder_path):
    """
    Load all JSON files from the folder assuming standard format:
      {
          "packet_loss_rate": <float>,
          "jitter": <float>,
          "throughput": <float>,
          "speed": <float>,
          "QoE": <float>,
          "timestamp": "YYYYMMDDHHMMSS"
      }
    Returns a DataFrame.
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

def load_augmented_dataset_from_folder(folder_path):
    """
    Load all JSON files from the folder assuming augmented format.
    Each JSON file is expected to have a structure like:
    {
      "<timestamp1>": { "packet_loss_rate": <float>, "jitter": <float>, "throughput": <float>, "speed": <float> },
      "<timestamp2>": { ... },
      ...
      "<timestampN>": { ... },
      "QoE": <float>,         # aggregated QoE for the 5-second window
      "timestamp": "YYYYMMDDHHMMSS"  # overall timestamp (last second in the window)
    }
    For each file, this function aggregates the per-second values (using mean) for the 4 features,
    and returns a DataFrame with columns: "packet_loss_rate", "jitter", "throughput", "speed", "QoE", "timestamp".
    """
    data = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            # Separate aggregated fields and per-second sub-records.
            aggregated = {}
            per_second_values = {"packet_loss_rate": [], "jitter": [], "throughput": [], "speed": []}
            for key, value in json_data.items():
                if key in ["QoE", "timestamp"]:
                    aggregated[key] = value
                else:
                    # Assume each other key holds a dict with the 4 features.
                    if isinstance(value, dict):
                        for feat in per_second_values.keys():
                            if feat in value:
                                per_second_values[feat].append(value[feat])
            # Compute aggregated (mean) for each feature if available.
            for feat, values in per_second_values.items():
                if values:
                    aggregated[feat] = round(float(np.mean(values)), 2)
                else:
                    aggregated[feat] = None
            data.append(aggregated)
    data_sorted = sorted(data, key=lambda x: x['timestamp'])
    df = pd.DataFrame(data_sorted)
    return df

def load_inference_file(file_path, augmented=False):
    """
    Loads a new inference JSON file.
    If augmented is False, expects standard structure.
    If augmented is True, expects augmented structure and computes aggregated features.
    Returns a dictionary with keys: "packet_loss_rate", "jitter", "throughput", "speed".
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    if not augmented:
        # Standard mode: simply return the 4 feature values.
        return {
            "packet_loss_rate": data["packet_loss_rate"],
            "jitter": data["jitter"],
            "throughput": data["throughput"],
            "speed": data["speed"]
        }
    else:
        # Augmented mode: compute aggregated features from per-second sub-records.
        per_second_values = {"packet_loss_rate": [], "jitter": [], "throughput": [], "speed": []}
        for key, value in data.items():
            if key in ["QoE", "timestamp"]:
                continue
            if isinstance(value, dict):
                for feat in per_second_values.keys():
                    if feat in value:
                        per_second_values[feat].append(value[feat])
        aggregated = {}
        for feat, values in per_second_values.items():
            if values:
                aggregated[feat] = round(float(np.mean(values)), 2)
            else:
                aggregated[feat] = None
        return aggregated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_file", type=str, required=True,
                        help="Path to the new inference JSON file.")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing the training dataset (to retrieve previous records).")
    parser.add_argument("--seq_length", type=int, default=5,
                        help="Sequence length (number of time steps used as input).")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to the saved trained model file.")
    parser.add_argument("--scaler_file", type=str, required=True,
                        help="Path to the saved scaler file.")
    parser.add_argument("--augmented", action="store_true",
                        help="Flag to indicate that the dataset is in augmented mode (5-second window with 1-second granularity).")
    args = parser.parse_args()

    # Load the saved model and scaler.
    model = tf.keras.models.load_model(args.model_file,
                                       custom_objects={"TransformerBlock": TransformerBlock})
    scaler = joblib.load(args.scaler_file)

    # Depending on the mode, load the training dataset appropriately.
    if args.augmented:
        df = load_augmented_dataset_from_folder(args.data_folder)
    else:
        df = load_dataset_from_folder(args.data_folder)
    
    # Ensure the dataframe is sorted by timestamp.
    df.sort_values("timestamp", inplace=True)
    
    # Define feature columns (we always use the 4 features).
    feature_cols = ['packet_loss_rate', 'jitter', 'throughput', 'speed']
    
    # For inference, we need the last (seq_length - 1) records from the training dataset.
    last_records = df.iloc[-(args.seq_length - 1):][feature_cols].values

    # Load the new inference file (aggregate if in augmented mode).
    new_record_features = load_inference_file(args.inference_file, augmented=args.augmented)
    new_data = np.array([[new_record_features["packet_loss_rate"],
                           new_record_features["jitter"],
                           new_record_features["throughput"],
                           new_record_features["speed"]]])
    # The scaler was fitted on 5 columns (4 features + QoE); add a dummy for QoE.
    dummy = np.zeros((new_data.shape[0], 1))
    new_record_full = np.hstack([new_data, dummy])
    new_record_scaled = scaler.transform(new_record_full)[:, :4]

    # Form the full input sequence by stacking the last records and the new record.
    sequence = np.vstack([last_records, new_record_scaled])
    sequence = sequence.reshape(1, args.seq_length, len(feature_cols))

    # Predict the QoE (prediction is in normalized scale).
    predicted_qoe_scaled = model.predict(sequence)
    
    # To convert the predicted QoE back to the original scale, create a dummy array.
    dummy_array = np.zeros((1, 5))
    dummy_array[0, -1] = predicted_qoe_scaled[0, 0]
    predicted_qoe = scaler.inverse_transform(dummy_array)[0, -1]
    
    print("Predicted QoE:", predicted_qoe)

if __name__ == "__main__":
    main()
