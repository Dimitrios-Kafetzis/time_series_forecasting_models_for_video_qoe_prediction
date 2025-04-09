#!/usr/bin/env python3
"""
Filename: prepare_validation_data.py
Description:
    This script prepares a validation dataset by creating pairs of files:
    1. Original files containing the ground truth QoE values
    2. Modified files with QoE values removed (for inference)
    
    The script supports the new augmented dataset format (10-second windows with 
    2-second intervals) as well as the legacy format.

Usage Examples:
    Basic usage:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset

    Specify a specific subset (percentage) of files to process:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --sample_ratio 0.2

    Create samples using a specified seed for reproducibility:
      $ python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --sample_ratio 0.3 --random_seed 42

    Create a validation dataset from legacy format files:
      $ python3 prepare_validation_data.py --input_folder ./old_dataset --output_folder ./validation_dataset --legacy_format
"""

import os
import json
import argparse
import random
import shutil
from datetime import datetime
import sys

def setup_directories(base_output_folder):
    """Create the necessary directory structure for validation data."""
    # Create directory for files with ground truth
    ground_truth_dir = os.path.join(base_output_folder, "ground_truth")
    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)
    
    # Create directory for files without QoE (for inference)
    inference_dir = os.path.join(base_output_folder, "inference")
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)
        
    return ground_truth_dir, inference_dir

def create_inference_copy(file_path, target_path, legacy_format=False):
    """
    Create a copy of the JSON file with QoE values removed.
    Returns the original QoE value for reference.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Get the original QoE value before removing
        original_qoe = data.get('QoE')
        
        # Replace QoE with None (representing unknown)
        data['QoE'] = None
        
        # Write the modified JSON to the target path
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        return original_qoe
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def validate_json_structure(file_path, legacy_format=False):
    """
    Validate if the JSON file has the expected structure for validation.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check for required fields
        if 'QoE' not in data or 'timestamp' not in data:
            return False
        
        # If QoE is null or None, file is not valid for validation
        if data['QoE'] is None:
            return False
            
        # For new format (default), check for timestamp data structure
        if not legacy_format:
            # Check that there are timestamp keys with nested data
            timestamp_keys = [k for k in data.keys() if k not in ['QoE', 'timestamp']]
            if not timestamp_keys:
                return False
                
            # Check nested structure on first timestamp
            first_ts = timestamp_keys[0]
            required_fields = ['throughput', 'packets_lost', 'packet_loss_rate', 'jitter', 'speed']
            for field in required_fields:
                if field not in data[first_ts]:
                    return False
                    
        return True
    except Exception as e:
        print(f"Error validating file {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Prepare validation dataset by creating pairs of files.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to folder containing original JSON files.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to output folder for validation dataset.')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help='Fraction of files to include in validation set (default: 1.0 = all files).')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for reproducible sampling (default: None).')
    parser.add_argument('--legacy_format', action='store_true',
                        help='Indicate that the dataset is in legacy format.')
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.random_seed is not None:
        random.seed(args.random_seed)
    
    # Create output directories
    ground_truth_dir, inference_dir = setup_directories(args.output_folder)
    
    # Create a metadata file to store ground truth QoE values
    metadata_path = os.path.join(args.output_folder, "validation_metadata.json")
    metadata = {
        "creation_timestamp": datetime.now().strftime("%Y%m%d%H%M%S"),
        "files": {},
        "format": "legacy" if args.legacy_format else "new",
        "sample_ratio": args.sample_ratio,
        "random_seed": args.random_seed
    }
    
    # Get list of all JSON files
    all_files = [f for f in os.listdir(args.input_folder) if f.endswith('.json')]
    
    # Sample files based on ratio
    if args.sample_ratio < 1.0:
        num_files = int(len(all_files) * args.sample_ratio)
        selected_files = random.sample(all_files, num_files)
    else:
        selected_files = all_files
    
    print(f"Preparing validation dataset with {len(selected_files)} files")
    
    valid_file_count = 0
    invalid_file_count = 0
    
    # Process each file
    for filename in selected_files:
        src_path = os.path.join(args.input_folder, filename)
        
        # Check if file has valid structure for validation
        if validate_json_structure(src_path, args.legacy_format):
            # Copy original file to ground truth directory
            gt_path = os.path.join(ground_truth_dir, filename)
            shutil.copy2(src_path, gt_path)
            
            # Create modified version with QoE removed
            inf_path = os.path.join(inference_dir, filename)
            qoe_value = create_inference_copy(src_path, inf_path, args.legacy_format)
            
            # Store original QoE value in metadata
            if qoe_value is not None:
                metadata["files"][filename] = {
                    "ground_truth_qoe": qoe_value
                }
                valid_file_count += 1
        else:
            invalid_file_count += 1
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Print summary
    print(f"Validation dataset preparation complete:")
    print(f"  - Valid files processed: {valid_file_count}")
    print(f"  - Invalid files skipped: {invalid_file_count}")
    print(f"  - Ground truth files saved to: {ground_truth_dir}")
    print(f"  - Inference files saved to: {inference_dir}")
    print(f"  - Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()