#!/usr/bin/env python3
"""
Filename: generate_mock_dataset.py
Author: Dimitrios Kafetzis (Updated version)
Creation Date: 2025-02-04
Description:
    This script generates a mock dataset of JSON files representing time series data.
    Each JSON file corresponds to a timepoint and follows this format:
    
    {
        "QoE": <float>,
        "timestamp": <int>,
        "<timestamp_1>": {
            "throughput": <float>,
            "packets_lost": <float>,
            "packet_loss_rate": <float>,
            "jitter": <float>,
            "speed": <float>
        },
        "<timestamp_2>": { ... },
        "<timestamp_3>": { ... },
        "<timestamp_4>": { ... },
        "<timestamp_5>": { ... }
    }

    The script supports three modes:
        - "dataset": Generates files with complete data (including QoE).
        - "inference": Generates files with the QoE field set to null (for use as unknown inputs during inference).
        - "augmented": The original mode is maintained for backward compatibility, but now all modes use the
                      10-second window with 2-second interval format by default.
    
    The synthetic data generation has been enhanced with time-dependent variations to simulate
    peak traffic conditions.

Usage Examples:
    1. Generate a training dataset of 100 JSON files (10-second window per file):
       $ python3 generate_mock_dataset.py --output_folder ./mock_dataset --mode dataset --num_points 100 --start_timestamp 20250130114158

    2. Generate an inference file (with unknown QoE):
       $ python3 generate_mock_dataset.py --output_folder ./inference_inputs --mode inference --num_points 1 --start_timestamp 20250204123000

    3. Generate an augmented dataset with 100 files (each covering 10 seconds with 2-second granularity):
       $ python3 generate_mock_dataset.py --output_folder ./augmented_dataset --mode augmented --num_points 100 --start_timestamp 20250130114158
"""

import os
import json
import argparse
import random
from datetime import datetime, timedelta
import numpy as np

def generate_data_point(timestamp, mode="dataset"):
    """
    Generates a single data point as a dictionary.
    
    In 'dataset' mode, all fields are populated (including QoE).
    In 'inference' mode, the QoE field is set to None.
    
    The generated values simulate an urban mobile (4G/5G) scenario using realistic ranges:
      - packet_loss_rate: Percentage between 0 and 5.
      - packets_lost: Absolute number between 0 and 5.
      - jitter: Milliseconds between 15 and 30.
      - throughput: Mbps between 600 and 1600.
      - speed: km/h between 0 and 60.
      
    To simulate peak traffic conditions (typically during rush hours, e.g., 7-9 AM and 5-8 PM):
      - An extra penalty (peak_penalty) is applied to the QoE.
      - Additional jitter (extra_jitter) is added to the base jitter value.
      
    QoE Calculation:
      The QoE (Quality of Experience) is computed on a scale of 0-100.
    
    Returns:
        A dictionary containing the generated features.
    """
    # Extract hour for peak traffic simulation
    hour = timestamp.hour

    # Determine if current time falls within peak traffic hours.
    if 7 <= hour < 10 or 17 <= hour < 20:
        peak_penalty = 5.0  # Extra penalty to simulate degraded quality during rush hours.
        extra_jitter = random.uniform(0, 5)  # Introduce additional jitter (up to 5 ms) during peak hours.
    else:
        peak_penalty = 0.0
        extra_jitter = 0.0

    # Generate basic network and mobility features using realistic random ranges.
    packet_loss_rate = round(random.uniform(0, 5), 2)  # Packet loss rate between 0% and 5%.
    packets_lost = round(random.uniform(0, 5), 1)  # Number of packets lost between 0 and 5.
    jitter = round(random.uniform(15, 30), 3) + extra_jitter  # Base jitter with added extra jitter during peak hours.
    throughput = round(random.uniform(600, 1600), 1)  # Throughput between 600 and 1600 Mbps.
    speed = round(random.uniform(0, 60), 0)  # Speed between 0 and 60 km/h.

    # Compute QoE on a scale of 0-100
    noise = random.uniform(-1.0, 1.0)
    qoe_value = (
        95.0
        - (packet_loss_rate * 4.0)
        - (jitter * 0.1)
        + (((throughput - 600) / 1000) * 5.0)
        - (speed * 0.05)
        - peak_penalty
        + noise
    )
    # Clip the QoE value to the range [0, 100] and round to 6 decimal places.
    qoe_value = round(max(0, min(qoe_value, 100)), 6)
    
    # Build the final data point dictionary with all features.
    data_point = {
        "throughput": throughput,
        "packets_lost": packets_lost,
        "packet_loss_rate": packet_loss_rate,
        "jitter": jitter,
        "speed": speed
    }
    return data_point, qoe_value

def save_augmented_data(augmented_data, output_folder):
    """
    Saves an augmented data dictionary as a JSON file.
    The filename is based on the 'timestamp' field.
    """
    filename = f"{augmented_data['timestamp']}.json"
    file_path = os.path.join(output_folder, filename)
    with open(file_path, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    print(f"Saved {file_path}")

def compute_statistics(data_points):
    """
    Computes summary statistics for the generated dataset.
    Returns a dictionary of statistics.
    """
    stats = {}
    num_files = len(data_points)
    stats['total_files'] = num_files
    timestamps = [dp['timestamp'] for dp in data_points]
    start_time = min(timestamps)
    end_time = max(timestamps)
    stats['start_timestamp'] = start_time
    stats['end_timestamp'] = end_time
    
    # Convert timestamps to datetime for time span calculation
    start_dt = datetime.strptime(str(start_time), "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(str(end_time), "%Y%m%d%H%M%S")
    stats['time_span'] = str(end_dt - start_dt)
    
    combined = {"throughput": [], "packets_lost": [], "packet_loss_rate": [], "jitter": [], "speed": []}
    qoe_values = []
    
    for dp in data_points:
        if dp.get("QoE") is not None:
            qoe_values.append(dp["QoE"])
        
        for key, value in dp.items():
            if key in ["QoE", "timestamp"]:
                continue
            sub_dp = value
            for feature in combined.keys():
                combined[feature].append(sub_dp[feature])
    
    for feature, values in combined.items():
        arr = np.array(values)
        feature_stats = {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            '25th_percentile': float(np.percentile(arr, 25)),
            '75th_percentile': float(np.percentile(arr, 75)),
            'coefficient_of_variation': float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else None
        }
        stats[feature] = feature_stats
    
    if qoe_values:
        arr = np.array(qoe_values)
        qoe_stats = {
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'std': float(np.std(arr)),
            '25th_percentile': float(np.percentile(arr, 25)),
            '75th_percentile': float(np.percentile(arr, 75)),
            'coefficient_of_variation': float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else None
        }
        stats["QoE"] = qoe_stats
    
    return stats

def estimate_disk_usage(output_folder):
    """
    Estimates the total disk usage (in bytes) of all JSON files in the output folder.
    """
    total_size = 0
    for file in os.listdir(output_folder):
        if file.endswith('.json'):
            file_path = os.path.join(output_folder, file)
            total_size += os.path.getsize(file_path)
    return total_size

def generate_files(output_folder, start_timestamp, num_points, mode):
    """
    Generates a series of JSON files for all modes.
    Each file corresponds to a 10-second window with 2-second intervals.
    Returns the list of generated data points.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    data_points = []
    start_time = start_timestamp
    
    # Generate one file every 10 seconds
    for i in range(num_points):
        window_data = {}
        qoe_values = []
        
        # For each window, generate data points every 2 seconds
        for j in range(5):
            current_timestamp = start_time + timedelta(seconds=i*10 + j*2)
            timestamp_key = int(current_timestamp.strftime("%Y%m%d%H%M%S"))
            
            dp, qoe = generate_data_point(current_timestamp, mode=mode)
            qoe_values.append(qoe)
            window_data[str(timestamp_key)] = dp
        
        # Calculate aggregated QoE or set to None for inference mode
        if mode != "inference":
            aggregated_qoe = sum(qoe_values) / len(qoe_values)
            window_data["QoE"] = aggregated_qoe
        else:
            window_data["QoE"] = None
        
        # Set the overall timestamp (the last timestamp in the window)
        final_timestamp = start_time + timedelta(seconds=i*10 + 8)
        window_data["timestamp"] = int(final_timestamp.strftime("%Y%m%d%H%M%S"))
        
        data_points.append(window_data)
        save_augmented_data(window_data, output_folder)
    
    return data_points

def print_statistics(stats, disk_usage, num_points, mode):
    """
    Prints a summary of the statistics in a human-readable format.
    """
    print("\n--- Dataset Generation Summary ---")
    print(f"Total files generated: {stats['total_files']}")
    print(f"Start timestamp: {stats['start_timestamp']}")
    print(f"End timestamp: {stats['end_timestamp']}")
    print(f"Time span: {stats['time_span']}")
    hours = (num_points - 1) * 10 / 3600.0
    print(f"Estimated duration (10-second windows): {hours:.2f} hours")
    
    print("\nFeature Statistics:")
    for key, value in stats.items():
        if key in ['total_files', 'start_timestamp', 'end_timestamp', 'time_span']:
            continue
        print(f"\n{key}:")
        for stat_key, stat_value in value.items():
            print(f"  {stat_key}: {stat_value}")
    
    size_in_kb = disk_usage / 1024.0
    size_in_mb = size_in_kb / 1024.0
    print("\nEstimated Disk Usage of JSON Files:")
    print(f"  {disk_usage} bytes ({size_in_kb:.2f} KB, {size_in_mb:.2f} MB)")
    
    print("\nGeneration Settings Recap:")
    print("  - One file per 10-second window with data points every 2 seconds")
    print(f"  - Total points (files): {num_points}")
    print("  - Mode:", "Dataset (with QoE values)" if mode == "dataset" else ("Inference (QoE is None)" if mode == "inference" else "Augmented"))
    
    print("\nDiversity Indicators (Coefficient of Variation):")
    for feature in ["throughput", "packets_lost", "packet_loss_rate", "jitter", "speed"] + (["QoE"] if mode in ["dataset", "augmented"] else []):
        if feature in stats:
            cv = stats[feature].get('coefficient_of_variation', None)
            if cv is not None:
                print(f"  {feature}: {cv:.4f}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate a mock dataset of JSON files with different modes."
    )
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the folder where JSON files will be stored.")
    parser.add_argument("--start_timestamp", type=str, default=None,
                        help="Starting timestamp in YYYYMMDDHHMMSS format. If not provided, the current time is used.")
    parser.add_argument("--num_points", type=int, default=100,
                        help="Number of JSON files to generate.")
    parser.add_argument("--mode", type=str, choices=["dataset", "inference", "augmented"],
                        default="dataset",
                        help="Mode to run: 'dataset' (with QoE), 'inference' (QoE is None), or 'augmented' (backward compatibility).")
    args = parser.parse_args()
    
    if args.start_timestamp:
        try:
            start_time = datetime.strptime(args.start_timestamp, "%Y%m%d%H%M%S")
        except ValueError:
            print("Error: start_timestamp must be in YYYYMMDDHHMMSS format.")
            return
    else:
        start_time = datetime.now()
        print("No start_timestamp provided. Using current time:", start_time.strftime("%Y%m%d%H%M%S"))
    
    # All modes now use the same file generation function
    data_points = generate_files(args.output_folder, start_time, args.num_points, args.mode)
    
    stats = compute_statistics(data_points)
    disk_usage = estimate_disk_usage(args.output_folder)
    print_statistics(stats, disk_usage, args.num_points, args.mode)
    
if __name__ == "__main__":
    main()