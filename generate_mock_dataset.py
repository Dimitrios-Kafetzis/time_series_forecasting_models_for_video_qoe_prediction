#!/usr/bin/env python3
"""
Filename: generate_mock_dataset.py
Author: Dimitrios Kafetzis
Creation Date: 2025-02-04
Description:
    This script generates a mock dataset of JSON files representing time series data.
    Each JSON file corresponds to a timepoint (with different granularities, depending on mode)
    and follows one of the two formats:
    
    Standard modes ("dataset" and "inference"):
      {
          "packet_loss_rate": <float>,
          "jitter": <float>,
          "throughput": <float>,
          "speed": <float>,
          "QoE": <float> or null,
          "timestamp": "YYYYMMDDHHMMSS",
          "hour": <int>,
          "minute": <int>,
          "day_of_week": <int>
      }
      
    Augmented mode ("augmented"):
      {
          "<timestamp_1>": {
              "packet_loss_rate": <float>,
              "jitter": <float>,
              "throughput": <float>,
              "speed": <float>
          },
          "<timestamp_2>": { ... },
          "<timestamp_3>": { ... },
          "<timestamp_4>": { ... },
          "<timestamp_5>": { ... },
          "QoE": <aggregated QoE over the 5 seconds (0 to 1)>,
          "timestamp": "<timestamp of the 5th second in the window>"
      }

    The script supports three modes:
        - "dataset": Generates files with complete data (including QoE) for training/testing.
        - "inference": Generates files with the QoE field set to null (for use as unknown inputs during inference).
        - "augmented": Generates files for a 5‑second window with per‑second metrics and an aggregated QoE.
    
    The synthetic data generation has been enhanced with time-dependent variations to simulate
    peak traffic conditions. Additional temporal features (hour, minute, day_of_week) are also included.

Usage Examples:
    1. Generate a training dataset of 100 JSON files (10-second interval per file):
       $ python3 generate_mock_dataset.py --output_folder ./mock_dataset --mode dataset --num_points 100 --start_timestamp 20250130114158

    2. Generate an inference file (with unknown QoE):
       $ python3 generate_mock_dataset.py --output_folder ./inference_inputs --mode inference --num_points 1 --start_timestamp 20250204123000

    3. Generate an augmented dataset with 100 files (each covering 5 seconds with 1-second granularity):
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
      - jitter: Milliseconds between 20 and 100.
      - throughput: Mbps between 10 and 100.
      - speed: km/h between 20 and 60.
      
    Additional temporal features are extracted from the given timestamp:
      - hour: The hour of the day (0-23).
      - minute: The minute of the hour (0-59).
      - day_of_week: The day of the week (Monday=0, Sunday=6).
      
    To simulate peak traffic conditions (typically during rush hours, e.g., 7-9 AM and 5-8 PM):
      - An extra penalty (peak_penalty) is applied to the QoE.
      - Additional jitter (extra_jitter) is added to the base jitter value.
      
    QoE Calculation:
      The QoE (Quality of Experience) is computed as follows:
        1. Start with a base quality of 0.95, representing near-ideal conditions.
        2. Subtract a penalty for packet loss (0.04 per percentage point).
        3. Subtract a penalty for jitter (0.003 per millisecond), including any extra jitter during peak times.
        4. Add a bonus for throughput; throughput is normalized such that 10 Mbps gives 0 bonus and 100 Mbps gives up to 0.3 bonus.
        5. Subtract a penalty for speed; speeds higher than 20 km/h incur a penalty scaled by 0.05.
        6. Subtract an additional fixed penalty (peak_penalty) during peak hours.
        7. Add a small random noise in the range [-0.02, 0.02] to simulate measurement variability.
      Finally, the QoE value is clipped to the range [0, 1] and rounded to two decimal places.
    
    Returns:
        A dictionary containing the generated features, QoE (or None in inference mode),
        a timestamp, and additional temporal features.
    """
    # Extract temporal features from the timestamp.
    hour = timestamp.hour
    minute = timestamp.minute
    day_of_week = timestamp.weekday()  # Monday=0, Sunday=6

    # Determine if current time falls within peak traffic hours.
    if 7 <= hour < 10 or 17 <= hour < 20:
        peak_penalty = 0.05  # Extra penalty to simulate degraded quality during rush hours.
        extra_jitter = random.uniform(0, 5)  # Introduce additional jitter (up to 5 ms) during peak hours.
    else:
        peak_penalty = 0.0
        extra_jitter = 0.0

    # Generate basic network and mobility features using realistic random ranges.
    packet_loss_rate = round(random.uniform(0, 5), 2)  # Packet loss rate between 0% and 5%.
    jitter = round(random.uniform(20, 100), 2) + extra_jitter  # Base jitter with added extra jitter during peak hours.
    throughput = round(random.uniform(10, 100), 2)  # Throughput between 10 and 100 Mbps.
    speed = round(random.uniform(20, 60), 2)  # Speed between 20 and 60 km/h.

    # Compute QoE using a formula inspired by VMAF:
    # - Start with a base value of 0.95.
    # - Subtract penalties based on packet loss and jitter.
    # - Add a bonus based on normalized throughput.
    # - Subtract a penalty based on speed.
    # - Subtract an extra peak penalty during rush hours.
    # - Add a small random noise to simulate variability.
    noise = random.uniform(-0.02, 0.02)
    qoe_value = (
        0.95
        - (packet_loss_rate * 0.04)
        - (jitter * 0.003)
        + (((throughput - 10) / 90) * 0.3)
        - (((speed - 20) / 40) * 0.05)
        - peak_penalty
        + noise
    )
    # Clip the QoE value to the range [0, 1] and round to 2 decimal places.
    qoe_value = round(max(0, min(qoe_value, 1)), 2)
    
    # Build the final data point dictionary with all features.
    data_point = {
        "packet_loss_rate": packet_loss_rate,
        "jitter": round(jitter, 2),
        "throughput": throughput,
        "speed": speed,
        "QoE": qoe_value if mode == "dataset" else None,
        "timestamp": timestamp.strftime("%Y%m%d%H%M%S"),
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week
    }
    return data_point

def save_data_point(data_point, output_folder):
    """
    Saves a single data point dictionary as a JSON file.
    The filename is based on the timestamp (YYYYMMDDHHMMSS.json).
    """
    filename = f"{data_point['timestamp']}.json"
    file_path = os.path.join(output_folder, filename)
    with open(file_path, 'w') as f:
        json.dump(data_point, f, indent=4)
    print(f"Saved {file_path}")

def save_augmented_data(augmented_data, output_folder):
    """
    Saves an augmented data dictionary as a JSON file.
    The filename is based on the aggregated 'timestamp' field.
    """
    filename = f"{augmented_data['timestamp']}.json"
    file_path = os.path.join(output_folder, filename)
    with open(file_path, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    print(f"Saved {file_path}")

def compute_statistics(data_points, mode):
    """
    Computes summary statistics for the generated dataset.
    Returns a dictionary of statistics.
    
    For 'dataset' and 'inference' modes, each element in data_points is a flat dictionary.
    For 'augmented' mode, each element is a dictionary that contains several per-second sub-records.
    """
    stats = {}
    if mode != "augmented":
        num_points = len(data_points)
        stats['total_files'] = num_points
        timestamps = [datetime.strptime(dp['timestamp'], "%Y%m%d%H%M%S") for dp in data_points]
        start_time = min(timestamps)
        end_time = max(timestamps)
        stats['start_timestamp'] = start_time.strftime("%Y%m%d%H%M%S")
        stats['end_timestamp'] = end_time.strftime("%Y%m%d%H%M%S")
        stats['time_span'] = str(end_time - start_time)
        features = ["packet_loss_rate", "jitter", "throughput", "speed"]
        if mode == "dataset":
            features.append("QoE")
        for feature in features:
            values = [dp[feature] for dp in data_points if dp[feature] is not None]
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
    else:
        num_files = len(data_points)
        stats['total_files'] = num_files
        timestamps = [datetime.strptime(dp['timestamp'], "%Y%m%d%H%M%S") for dp in data_points]
        start_time = min(timestamps)
        end_time = max(timestamps)
        stats['start_timestamp'] = start_time.strftime("%Y%m%d%H%M%S")
        stats['end_timestamp'] = end_time.strftime("%Y%m%d%H%M%S")
        stats['time_span'] = str(end_time - start_time)
        combined = { "packet_loss_rate": [], "jitter": [], "throughput": [], "speed": [] }
        aggregated_qoe = []
        for dp in data_points:
            for key, sub_dp in dp.items():
                if key in ["QoE", "timestamp"]:
                    continue
                for feature in combined.keys():
                    combined[feature].append(sub_dp[feature])
            if dp.get("QoE") is not None:
                aggregated_qoe.append(dp["QoE"])
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
        if aggregated_qoe:
            arr = np.array(aggregated_qoe)
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
    Generates a series of JSON files for modes "dataset" and "inference".
    Each file corresponds to a 10-second interval.
    Returns the list of generated data points.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    data_points = []
    current_timestamp = start_timestamp
    for i in range(num_points):
        data_point = generate_data_point(current_timestamp, mode=mode)
        data_points.append(data_point)
        save_data_point(data_point, output_folder)
        current_timestamp += timedelta(seconds=10)
    return data_points

def generate_augmented_files(output_folder, start_timestamp, num_windows):
    """
    Generates a series of JSON files in augmented mode.
    Each file corresponds to a 5-second window with 1-second granularity.
    Returns a list of augmented data dictionaries.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    augmented_data_points = []
    for i in range(num_windows):
        window_data = {}
        qoe_values = []
        for j in range(5):
            current_timestamp = start_timestamp + timedelta(seconds=i*5 + j)
            dp = generate_data_point(current_timestamp, mode="dataset")
            qoe_values.append(dp["QoE"])
            sub_record = {
                "packet_loss_rate": dp["packet_loss_rate"],
                "jitter": dp["jitter"],
                "throughput": dp["throughput"],
                "speed": dp["speed"]
            }
            key = dp["timestamp"]
            window_data[key] = sub_record
        aggregated_qoe = round(sum(qoe_values) / len(qoe_values), 2)
        window_data["QoE"] = aggregated_qoe
        overall_timestamp = start_timestamp + timedelta(seconds=i*5 + 4)
        window_data["timestamp"] = overall_timestamp.strftime("%Y%m%d%H%M%S")
        augmented_data_points.append(window_data)
        save_augmented_data(window_data, output_folder)
    return augmented_data_points

def print_statistics(stats, disk_usage, num_points, mode):
    """
    Prints a summary of the statistics in a human‐readable format.
    """
    print("\n--- Dataset Generation Summary ---")
    print(f"Total files generated: {stats['total_files']}")
    print(f"Start timestamp: {stats['start_timestamp']}")
    print(f"End timestamp: {stats['end_timestamp']}")
    print(f"Time span: {stats['time_span']}")
    if mode in ["dataset", "inference"]:
        hours = (num_points - 1) * 10 / 3600.0
        print(f"Estimated duration (10-second intervals): {hours:.2f} hours")
    elif mode == "augmented":
        hours = (num_points - 1) * 5 / 3600.0
        print(f"Estimated duration (5-second windows): {hours:.2f} hours")
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
    if mode in ["dataset", "inference"]:
        print("  - One file per 10-second interval")
    elif mode == "augmented":
        print("  - One file per 5-second window with per-second granularity")
    print(f"  - Total points (files): {num_points}")
    print("  - Mode:", "Dataset (with QoE values)" if mode == "dataset" else ("Inference (QoE is None)" if mode == "inference" else "Augmented"))
    print("\nDiversity Indicators (Coefficient of Variation):")
    for feature in ["packet_loss_rate", "jitter", "throughput", "speed"] + (["QoE"] if mode in ["dataset", "augmented"] else []):
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
                        help="Number of JSON files (or windows, for augmented mode) to generate.")
    parser.add_argument("--mode", type=str, choices=["dataset", "inference", "augmented"],
                        default="dataset",
                        help="Mode to run: 'dataset' (with QoE), 'inference' (QoE is None), or 'augmented' (5-second window with per-second granularity).")
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
    
    if args.mode == "augmented":
        data_points = generate_augmented_files(args.output_folder, start_time, args.num_points)
    else:
        data_points = generate_files(args.output_folder, start_time, args.num_points, args.mode)
    
    stats = compute_statistics(data_points, args.mode)
    disk_usage = estimate_disk_usage(args.output_folder)
    print_statistics(stats, disk_usage, args.num_points, args.mode)
    
if __name__ == "__main__":
    main()
