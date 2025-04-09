#!/usr/bin/env python3
"""
Filename: validate_models.py
Description:
    This script performs controlled validation experiments on time series forecasting models.
    It takes:
    1. A validation dataset prepared by prepare_validation_data.py
    2. One or more trained models
    3. The associated scaler
    
    It runs inference on the QoE-less files, compares predictions with ground truth,
    calculates various metrics, and generates visualizations of model performance.
    
    The script supports validating multiple models at once and generating
    comparative reports and visualizations.

Usage Examples:
    Validate a single model:
      $ python3 validate_models.py --validation_folder ./validation_dataset --model_file model_lstm.h5 --scaler_file scaler.save --output_dir ./validation_results

    Validate multiple models and create comparison plots:
      $ python3 validate_models.py --validation_folder ./validation_dataset --model_dir ./forecasting_models_v5 --scaler_file scaler.save --output_dir ./validation_results

    Validate models with specific sequence length:
      $ python3 validate_models.py --validation_folder ./validation_dataset --model_dir ./forecasting_models_v5 --scaler_file scaler.save --output_dir ./validation_results --seq_length 5

    Validate models with statistical features enabled:
      $ python3 validate_models.py --validation_folder ./validation_dataset --model_dir ./forecasting_models_v5 --scaler_file scaler.save --output_dir ./validation_results --use_stats
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from datetime import datetime
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score, 
                           median_absolute_error, explained_variance_score)

# Define this function here to avoid dependency issues
def detect_model_type(model_path):
    """
    Detect the model type from the filename to provide additional information.
    """
    filename = os.path.basename(model_path).lower()
    
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

# Try to import functions from the forecasting models module
try:
    from timeseries_forecasting_models_v5 import (
        TransformerBlock, SelfAttention, 
        load_dataset_from_folder, load_augmented_dataset_from_folder,
        preprocess_dataframe, create_sequences
    )
    print("Successfully imported from timeseries_forecasting_models_v5")
except ImportError as e:
    print(f"Import error from timeseries_forecasting_models_v5: {str(e)}")
    try:
        from test_models import (
            TransformerBlock, SelfAttention,
            load_augmented_dataset_from_folder_fallback as load_augmented_dataset_from_folder,
            preprocess_dataframe_fallback as preprocess_dataframe,
            create_sequences_fallback as create_sequences
        )
        print("Successfully imported from test_models")
    except ImportError as e:
        print(f"Import error from test_models: {str(e)}")
        print("Error: Could not import required functions. Please ensure the files timeseries_forecasting_models_v5.py or test_models.py are in the current directory.")
        sys.exit(1)

def debug_print(message):
    """Helper function to print debug messages with timestamps"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {message}")

def load_validation_dataset(validation_folder):
    """Load the validation dataset and metadata."""
    debug_print(f"Looking for validation data in: {validation_folder}")
    
    ground_truth_dir = os.path.join(validation_folder, "ground_truth")
    inference_dir = os.path.join(validation_folder, "inference")
    metadata_path = os.path.join(validation_folder, "validation_metadata.json")
    
    # Check if directories exist
    if not os.path.exists(ground_truth_dir):
        debug_print(f"Error: ground_truth directory not found at {ground_truth_dir}")
        sys.exit(1)
        
    if not os.path.exists(inference_dir):
        debug_print(f"Error: inference directory not found at {inference_dir}")
        sys.exit(1)
        
    if not os.path.exists(metadata_path):
        debug_print(f"Error: validation_metadata.json not found at {metadata_path}")
        sys.exit(1)
    
    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        debug_print(f"Metadata loaded with {len(metadata.get('files', {}))} validation files")
    except Exception as e:
        debug_print(f"Error reading metadata file: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    return ground_truth_dir, inference_dir, metadata

def load_and_preprocess_validation_files(inference_dir, seq_length, feature_cols, scaler, legacy_format=False, use_stats=False):
    """Load and preprocess validation files for inference."""
    debug_print(f"Loading files from {inference_dir} with use_stats={use_stats}, legacy_format={legacy_format}")
    
    # Load all the inference files
    try:
        if legacy_format:
            df = load_augmented_dataset_from_folder(inference_dir, use_stats=use_stats, new_format=False)
        else:
            df = load_augmented_dataset_from_folder(inference_dir, use_stats=use_stats, new_format=True)
        debug_print(f"Loaded dataframe with shape: {df.shape}")
    except Exception as e:
        debug_print(f"Error loading files: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Preprocess the data
    try:
        df = preprocess_dataframe(df)
        debug_print("Preprocessing completed")
    except Exception as e:
        debug_print(f"Error in preprocessing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Get feature columns if not provided
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ["QoE", "timestamp"]]
        debug_print(f"Generated feature columns: {len(feature_cols)} columns")
    
    # Define normalization columns
    norm_cols = feature_cols + ["QoE"]
    
    # Apply scaler transformation
    try:
        df[norm_cols] = scaler.transform(df[norm_cols])
        debug_print("Normalization applied")
    except Exception as e:
        debug_print(f"Error applying normalization: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Add a normalized timestamp column for matching
    try:
        df['timestamp_str'] = df['timestamp'].astype(str)
        df['timestamp_norm'] = df['timestamp_str'].apply(lambda x: ''.join(filter(str.isdigit, x)))
        debug_print(f"Added normalized timestamp column. Sample: {df['timestamp_norm'].head().tolist()}")
    except Exception as e:
        debug_print(f"Error creating normalized timestamp: {str(e)}")
    
    return df, norm_cols, feature_cols

def process_file_timestamps(filenames):
    """
    Extract timestamps from filenames if they follow a specific pattern.
    Used to ensure proper ordering of files.
    """
    # Assuming filenames have timestamp structure e.g., 20250402131554.json
    timestamps = []
    for file in filenames:
        try:
            # Extract timestamp from filename
            ts = int(file.split('.')[0])
            timestamps.append((file, ts))
        except (ValueError, IndexError):
            # If filename doesn't contain a valid timestamp, use zero
            timestamps.append((file, 0))
    
    # Sort by timestamp
    return [t[0] for t in sorted(timestamps, key=lambda x: x[1])]

def generate_file_sequences(df, filenames, metadata, seq_length, feature_cols):
    """
    Generate sequences for each file, associate with ground truth, and keep track of file identity.
    """
    debug_print(f"Generating sequences for {len(filenames)} files with seq_length={seq_length}")
    X = []
    ground_truth = []
    file_mapping = []
    
    # Debug dataframe timestamps
    debug_print(f"DataFrame timestamp column type: {df['timestamp'].dtype}")
    debug_print(f"First few timestamps in DataFrame: {df['timestamp'].head().tolist()}")
    
    # Create normalized timestamps for comparison
    # First normalize the DataFrame timestamps by removing all non-numeric characters
    df_normalized_ts = []
    for ts in df['timestamp'].astype(str):
        # Remove all non-digit characters (spaces, colons, dashes, etc.)
        normalized = ''.join(filter(str.isdigit, ts))
        df_normalized_ts.append(normalized)
    
    debug_print(f"Normalized DataFrame timestamps example: {df_normalized_ts[:3]}")
    
    # Process each file
    processed_count = 0
    skipped_count = 0
    for i, filename in enumerate(filenames):
        # Skip if file is not in metadata
        if filename not in metadata["files"]:
            skipped_count += 1
            continue
            
        # Get ground truth QoE for this file
        gt_qoe = metadata["files"][filename]["ground_truth_qoe"]
        
        try:
            # Extract timestamp from filename (assuming format like 20250402131554.json)
            file_ts = filename.split('.')[0]
            
            # Check if this normalized timestamp appears in our normalized DataFrame timestamps
            if file_ts in df_normalized_ts:
                # Find the index of this timestamp
                matching_idx = df_normalized_ts.index(file_ts)
                
                # Now create a sequence ending with this index
                # We need seq_length rows to form a complete sequence
                if matching_idx >= seq_length - 1:
                    seq_start = matching_idx - (seq_length - 1)
                    seq_X = df.iloc[seq_start:matching_idx+1][feature_cols].values
                    
                    # Only use if we have a complete sequence
                    if len(seq_X) == seq_length:
                        X.append(seq_X)
                        ground_truth.append(gt_qoe)
                        file_mapping.append(filename)
                        processed_count += 1
                        debug_print(f"Successfully processed file {filename}")
                    else:
                        skipped_count += 1
                        debug_print(f"Skipping file {filename} - incomplete sequence (length={len(seq_X)})")
                else:
                    skipped_count += 1
                    debug_print(f"Skipping file {filename} - not enough prior data for sequence (idx={matching_idx})")
            else:
                skipped_count += 1
                debug_print(f"Skipping file {filename} - no matching timestamp found")
                debug_print(f"  File timestamp: {file_ts}")
                debug_print(f"  Example normalized DataFrame timestamps: {df_normalized_ts[:3]}")
        except Exception as e:
            debug_print(f"Error processing file {filename}: {str(e)}")
            traceback.print_exc()
            skipped_count += 1
            continue
    
    debug_print(f"Sequence generation complete: {processed_count} processed, {skipped_count} skipped")
    
    # If no sequences were generated, try a completely different approach
    if len(X) == 0:
        debug_print("WARNING: No sequences generated. Trying direct sequence creation...")
        
        try:
            # Create sequences directly without timestamp matching
            # Use sequence_length consecutive rows for X
            X_direct = []
            y_direct = []
            file_mapping_direct = []
            
            # For each file in the metadata
            for filename, file_data in metadata["files"].items():
                gt_qoe = file_data["ground_truth_qoe"]
                
                # Create a sequence from the last seq_length rows
                if len(df) >= seq_length:
                    seq_X = df.iloc[-seq_length:][feature_cols].values
                    X_direct.append(seq_X)
                    y_direct.append(gt_qoe)
                    file_mapping_direct.append(filename)
                    debug_print(f"Created direct sequence for file {filename}")
            
            debug_print(f"Direct sequence approach generated {len(X_direct)} sequences")
            
            if len(X_direct) > 0:
                return np.array(X_direct), np.array(y_direct), file_mapping_direct
        except Exception as e:
            debug_print(f"Direct sequence approach failed: {str(e)}")
            traceback.print_exc()
    
    return np.array(X), np.array(ground_truth), file_mapping

def load_models(model_args):
    """
    Load one or more models based on provided arguments.
    Returns a dictionary of model objects.
    """
    debug_print("Starting to load models...")
    models = {}
    
    # Set up custom objects for model loading
    custom_objects = {"TransformerBlock": TransformerBlock, "SelfAttention": SelfAttention}
    
    # Case 1: Single model file
    if model_args.model_file:
        model_path = model_args.model_file
        debug_print(f"Loading single model from {model_path}")
        try:
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            model_name = os.path.basename(model_path)
            models[model_name] = {
                "model": model, 
                "type": detect_model_type(model_path)
            }
            debug_print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            debug_print(f"Error loading model from {model_path}: {str(e)}")
            traceback.print_exc()
    
    # Case 2: Directory with multiple models
    elif model_args.model_dir:
        model_dir = model_args.model_dir
        debug_print(f"Loading models from directory: {model_dir}")
        
        if not os.path.exists(model_dir):
            debug_print(f"Error: Model directory {model_dir} not found")
            return {}
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
        debug_print(f"Found {len(model_files)} .h5 files in directory")
        
        for filename in model_files:
            model_path = os.path.join(model_dir, filename)
            debug_print(f"Loading model: {filename}")
            try:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
                models[filename] = {
                    "model": model, 
                    "type": detect_model_type(model_path)
                }
                debug_print(f"Successfully loaded model: {filename}")
            except Exception as e:
                debug_print(f"Error loading model from {model_path}: {str(e)}")
                traceback.print_exc()
    
    debug_print(f"Finished loading models, found {len(models)} valid models")
    return models

def run_inference(model, X, scaler, norm_cols):
    """
    Run inference on sequences and convert scaled predictions back to original values.
    """
    # Get predictions (scaled values)
    predictions_scaled = model.predict(X)
    
    # Create dummy arrays to invert the scaling
    dummy_pred = np.zeros((len(predictions_scaled), len(norm_cols)))
    dummy_pred[:, -1] = predictions_scaled.flatten()
    
    # Invert scaling to get actual QoE values
    predictions = scaler.inverse_transform(dummy_pred)[:, -1]
    
    return predictions

def calculate_metrics(ground_truth, predictions):
    """
    Calculate various evaluation metrics for the predictions.
    """
    metrics = {
        "mse": mean_squared_error(ground_truth, predictions),
        "rmse": np.sqrt(mean_squared_error(ground_truth, predictions)),
        "mae": mean_absolute_error(ground_truth, predictions),
        "medae": median_absolute_error(ground_truth, predictions),
        "r2": r2_score(ground_truth, predictions),
        "explained_variance": explained_variance_score(ground_truth, predictions),
        "mean_error": np.mean(predictions - ground_truth),
        "max_error": np.max(np.abs(predictions - ground_truth)),
        "min_error": np.min(np.abs(predictions - ground_truth))
    }
    
    return metrics

def generate_plots(ground_truth, predictions, file_mapping, model_name, output_dir):
    """
    Generate various visualization plots for model performance.
    """
    # Create output directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create a DataFrame for easier plotting
    results_df = pd.DataFrame({
        'Ground Truth': ground_truth,
        'Prediction': predictions,
        'Error': predictions - ground_truth,
        'File': file_mapping
    })
    
    # 1. Scatter plot: Ground Truth vs. Predictions
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(x='Ground Truth', y='Prediction', data=results_df, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(results_df['Ground Truth'].min(), results_df['Prediction'].min())
    max_val = max(results_df['Ground Truth'].max(), results_df['Prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'{model_name}: Ground Truth vs Predictions', fontsize=15)
    plt.xlabel('Ground Truth QoE', fontsize=12)
    plt.ylabel('Predicted QoE', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_scatter.png'), dpi=300)
    plt.close()
    
    # 2. Error Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['Error'], bins=30, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{model_name}: Prediction Error Distribution', fontsize=15)
    plt.xlabel('Prediction Error (Predicted - Ground Truth)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_error_dist.png'), dpi=300)
    plt.close()
    
    # 3. Error vs. Ground Truth (to check for bias patterns)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Ground Truth', y='Error', data=results_df, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name}: Error vs Ground Truth', fontsize=15)
    plt.xlabel('Ground Truth QoE', fontsize=12)
    plt.ylabel('Prediction Error', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_error_vs_truth.png'), dpi=300)
    plt.close()
    
    # 4. Top 10 largest errors
    top_errors = results_df.copy()
    top_errors['Abs_Error'] = np.abs(top_errors['Error'])
    top_errors = top_errors.sort_values('Abs_Error', ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(top_errors)), top_errors['Abs_Error'])
    plt.xticks(range(len(top_errors)), top_errors['File'], rotation=90)
    
    # Add file names as abbreviated labels
    plt.xticks(range(len(top_errors)), [f[:10] + '...' if len(f) > 10 else f for f in top_errors['File']], rotation=90)
    
    # Add error values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.title(f'{model_name}: Top 10 Files with Largest Prediction Errors', fontsize=15)
    plt.xlabel('File', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{model_name}_top_errors.png'), dpi=300)
    plt.close()
    
    return results_df

def generate_comparative_plots(all_results, output_dir):
    """
    Generate plots comparing different models.
    """
    if len(all_results) <= 1:
        return
        
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Create a combined DataFrame for comparison
    combined_df = pd.DataFrame()
    for model_name, results in all_results.items():
        model_df = pd.DataFrame({
            'Model': model_name,
            'MAE': results["metrics"]["mae"],
            'RMSE': results["metrics"]["rmse"],
            'R²': results["metrics"]["r2"],
            'Mean Error': results["metrics"]["mean_error"]
        }, index=[0])
        combined_df = pd.concat([combined_df, model_df])
    
    # Reset index
    combined_df.reset_index(drop=True, inplace=True)
    
    # 1. Bar chart of MAE and RMSE
    plt.figure(figsize=(14, 8))
    metrics_df = pd.melt(combined_df, id_vars=['Model'], value_vars=['MAE', 'RMSE'], 
                        var_name='Metric', value_name='Value')
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df)
    plt.title('MAE and RMSE Comparison Across Models', fontsize=15)
    plt.xticks(rotation=45)
    plt.ylabel('Value (lower is better)', fontsize=12)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison_error_metrics.png'), dpi=300)
    plt.close()
    
    # 2. Bar chart of R² Score
    plt.figure(figsize=(14, 6))
    sns.barplot(x='Model', y='R²', data=combined_df)
    plt.title('R² Score Comparison Across Models', fontsize=15)
    plt.xticks(rotation=45)
    plt.ylabel('R² (higher is better)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison_r2.png'), dpi=300)
    plt.close()
    
    # 3. Mean Error comparison (bias)
    plt.figure(figsize=(14, 6))
    sns.barplot(x='Model', y='Mean Error', data=combined_df)
    plt.title('Mean Error Comparison Across Models', fontsize=15)
    plt.xticks(rotation=45)
    plt.ylabel('Mean Error (closer to zero is better)', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison_bias.png'), dpi=300)
    plt.close()
    
    # 4. Prediction Error Distributions (combined)
    plt.figure(figsize=(14, 8))
    for model_name, results in all_results.items():
        results_df = results["results_df"]
        sns.kdeplot(results_df['Error'], label=model_name)
    
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution Comparison Across Models', fontsize=15)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Model')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'model_comparison_error_dist.png'), dpi=300)
    plt.close()

def generate_validation_report(all_results, metadata, args, output_dir):
    """
    Generate a comprehensive validation report with metrics and findings.
    """
    # Create report path
    report_path = os.path.join(output_dir, "validation_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("=======================================================\n")
        f.write("       TIME SERIES FORECASTING MODELS VALIDATION       \n")
        f.write(f"       Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}       \n")
        f.write("=======================================================\n\n")
        
        # Validation dataset information
        f.write("VALIDATION DATASET INFORMATION\n")
        f.write("-----------------------------\n")
        f.write(f"Dataset format: {metadata.get('format', 'unknown')}\n")
        f.write(f"Sample ratio: {metadata.get('sample_ratio', 'unknown')}\n")
        f.write(f"Random seed: {metadata.get('random_seed', 'None')}\n")
        f.write(f"Creation timestamp: {metadata.get('creation_timestamp', 'unknown')}\n")
        f.write(f"Number of validation files: {len(metadata.get('files', {}))}\n\n")
        
        # Validation parameters
        f.write("VALIDATION PARAMETERS\n")
        f.write("--------------------\n")
        f.write(f"Sequence length: {args.seq_length}\n")
        f.write(f"Statistical features: {'Enabled' if args.use_stats else 'Disabled'}\n")
        f.write(f"Original validation folder: {args.validation_folder}\n\n")
        
        # Results for each model
        f.write("MODEL VALIDATION RESULTS\n")
        f.write("----------------------\n\n")
        
        for model_name, results in all_results.items():
            metrics = results["metrics"]
            f.write(f"Model: {model_name}\n")
            f.write(f"Type: {results.get('model_type', 'Unknown')}\n")
            f.write("\nPerformance Metrics:\n")
            f.write(f"  MSE: {metrics['mse']:.6f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
            f.write(f"  MAE: {metrics['mae']:.6f}\n")
            f.write(f"  Median Absolute Error: {metrics['medae']:.6f}\n")
            f.write(f"  R² Score: {metrics['r2']:.6f}\n")
            f.write(f"  Explained Variance: {metrics['explained_variance']:.6f}\n")
            f.write(f"  Mean Error (Bias): {metrics['mean_error']:.6f}\n")
            f.write(f"  Max Absolute Error: {metrics['max_error']:.6f}\n")
            f.write(f"  Min Absolute Error: {metrics['min_error']:.6f}\n")
            
            # Files with largest errors
            top_errors = results["top_errors"]
            f.write("\nTop 5 Files with Largest Errors:\n")
            for i, (idx, row) in enumerate(top_errors.iterrows()):
                f.write(f"  {i+1}. File: {row['File']}\n")
                f.write(f"     Ground Truth: {row['Ground Truth']:.6f}\n")
                f.write(f"     Prediction: {row['Prediction']:.6f}\n")
                f.write(f"     Error: {row['Error']:.6f}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
        
        # Model comparison (if multiple models)
        if len(all_results) > 1:
            f.write("MODEL COMPARISON\n")
            f.write("--------------\n")
            f.write("Models ranked by R² Score (higher is better):\n")
            
            # Sort models by R² score
            models_by_r2 = sorted(
                [(name, results["metrics"]["r2"]) for name, results in all_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (name, r2) in enumerate(models_by_r2):
                f.write(f"{i+1}. {name}: {r2:.6f}\n")
            
            f.write("\nModels ranked by RMSE (lower is better):\n")
            
            # Sort models by RMSE
            models_by_rmse = sorted(
                [(name, results["metrics"]["rmse"]) for name, results in all_results.items()],
                key=lambda x: x[1]
            )
            
            for i, (name, rmse) in enumerate(models_by_rmse):
                f.write(f"{i+1}. {name}: {rmse:.6f}\n")
            
            f.write("\nModels ranked by MAE (lower is better):\n")
            
            # Sort models by MAE
            models_by_mae = sorted(
                [(name, results["metrics"]["mae"]) for name, results in all_results.items()],
                key=lambda x: x[1]
            )
            
            for i, (name, mae) in enumerate(models_by_mae):
                f.write(f"{i+1}. {name}: {mae:.6f}\n")
                
            f.write("\nModels ranked by bias (absolute mean error, lower is better):\n")
            
            # Sort models by absolute mean error (bias)
            models_by_bias = sorted(
                [(name, abs(results["metrics"]["mean_error"])) for name, results in all_results.items()],
                key=lambda x: x[1]
            )
            
            for i, (name, bias) in enumerate(models_by_bias):
                f.write(f"{i+1}. {name}: {bias:.6f}\n")
                
        f.write("\n=======================================================\n")
        f.write("End of Validation Report\n")
        
    print(f"Validation report generated: {report_path}")
    return report_path

def export_results_csv(all_results, output_dir):
    """
    Export validation results to CSV format for further analysis.
    """
    # Create a DataFrame for the summary metrics
    metrics_rows = []
    for model_name, results in all_results.items():
        metrics = results["metrics"]
        row = {
            "Model": model_name,
            "Type": results.get("model_type", "Unknown"),
            "MSE": metrics["mse"],
            "RMSE": metrics["rmse"],
            "MAE": metrics["mae"],
            "MedianAE": metrics["medae"],
            "R2": metrics["r2"],
            "ExplainedVariance": metrics["explained_variance"],
            "MeanError": metrics["mean_error"],
            "MaxError": metrics["max_error"],
            "MinError": metrics["min_error"]
        }
        metrics_rows.append(row)
    
    # Create and save the summary DataFrame
    summary_df = pd.DataFrame(metrics_rows)
    summary_path = os.path.join(output_dir, "validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Create and save detailed prediction results for each model
    for model_name, results in all_results.items():
        details_df = results["results_df"]
        details_path = os.path.join(output_dir, f"{model_name}_detailed_results.csv")
        details_df.to_csv(details_path, index=False)
    
    print(f"Results exported to CSV in {output_dir}")

def main():
    debug_print("Starting validate_models.py script")
    
    # Add at the beginning of main()
    pd.set_option('display.max_colwidth', None)
    pd.options.display.float_format = '{:.0f}'.format

    parser = argparse.ArgumentParser(description='Validate time series forecasting models with ground truth data.')
    parser.add_argument('--validation_folder', type=str, required=True,
                        help='Path to validation dataset folder created by prepare_validation_data.py.')
    
    # Model options (either a single model or a directory of models)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model_file', type=str,
                            help='Path to a single model file (.h5) to validate.')
    model_group.add_argument('--model_dir', type=str,
                            help='Path to directory containing multiple model files to validate and compare.')
    
    parser.add_argument('--scaler_file', type=str, required=True,
                        help='Path to the saved scaler file used for normalization.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for validation results and plots.')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='Sequence length used by the model(s).')
    
    # Dataset options
    parser.add_argument('--use_stats', action='store_true',
                        help='Include extra statistical features (must match model training configuration).')
    parser.add_argument('--legacy_format', action='store_true',
                        help='Use legacy format for validation data.')
    
    debug_print("Parsing arguments...")
    args = parser.parse_args()
    debug_print(f"Arguments parsed: {vars(args)}")
    
    debug_print("Creating output directory...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        debug_print(f"Created output directory: {args.output_dir}")
    else:
        debug_print(f"Output directory already exists: {args.output_dir}")
    
    debug_print("Loading validation dataset...")
    try:
        ground_truth_dir, inference_dir, metadata = load_validation_dataset(args.validation_folder)
        debug_print(f"Validation dataset loaded successfully")
    except Exception as e:
        debug_print(f"Error loading validation dataset: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Detect legacy format from metadata (or use argument override)
    legacy_format = args.legacy_format or metadata.get("format", "new") == "legacy"
    if legacy_format:
        debug_print("Using legacy format for data processing")
    
    # Load scaler
    debug_print(f"Loading scaler from {args.scaler_file}")
    try:
        scaler = joblib.load(args.scaler_file)
        debug_print("Scaler loaded successfully")
    except Exception as e:
        debug_print(f"Error loading scaler: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    # Load models
    models = load_models(args)
    if not models:
        debug_print("No valid models found to validate")
        sys.exit(1)
    
    debug_print(f"Loaded {len(models)} model(s) for validation")
    
    # Load and preprocess the inference files
    debug_print("Loading and preprocessing validation files...")
    df, norm_cols, feature_cols = load_and_preprocess_validation_files(
        inference_dir, 
        args.seq_length, 
        None,  
        scaler, 
        legacy_format=legacy_format, 
        use_stats=args.use_stats
    )
    
    # Create a list of filenames ordered by timestamp
    debug_print("Sorting validation files by timestamp...")
    filenames = process_file_timestamps([f for f in os.listdir(inference_dir) if f.endswith('.json')])
    debug_print(f"Found {len(filenames)} validation files to process")
    
    # Generate sequences from files
    debug_print("Generating sequences for validation...")
    X, ground_truth, file_mapping = generate_file_sequences(df, filenames, metadata, args.seq_length, feature_cols)
    
    debug_print(f"Generated {len(X)} validation sequences")
    
    if len(X) == 0:
        debug_print("Error: No valid sequences could be generated. Check sequence length and file count.")
        sys.exit(1)
    
    if len(X) == 0:
        debug_print("Attempting final fallback method...")
        try:
            # Create synthetic sequences using sliding windows
            X_synthetic = []
            y_synthetic = []
            file_mapping_synthetic = []
            
            # Get all ground truth QoE values
            all_qoe_values = [metadata["files"][fname]["ground_truth_qoe"] for fname in metadata["files"]]
            avg_qoe = sum(all_qoe_values) / len(all_qoe_values) if all_qoe_values else 0
            debug_print(f"Average QoE from metadata: {avg_qoe}")
            
            # Create one sequence per file using a sliding window approach
            for i in range(len(df) - args.seq_length + 1):
                window = df.iloc[i:i+args.seq_length][feature_cols].values
                X_synthetic.append(window)
                
                # Map to a ground truth value if possible, or use average
                if i < len(filenames) and filenames[i] in metadata["files"]:
                    qoe = metadata["files"][filenames[i]]["ground_truth_qoe"]
                    filename = filenames[i]
                else:
                    qoe = avg_qoe
                    filename = f"synthetic_{i}.json"
                
                y_synthetic.append(qoe)
                file_mapping_synthetic.append(filename)
            
            debug_print(f"Created {len(X_synthetic)} synthetic sequences as final fallback")
            X = np.array(X_synthetic)
            ground_truth = np.array(y_synthetic)
            file_mapping = file_mapping_synthetic
        except Exception as e:
            debug_print(f"Final fallback method failed: {str(e)}")
            traceback.print_exc()

    # Run validation for each model
    all_results = {}
    
    for model_name, model_info in models.items():
        debug_print(f"\nValidating model: {model_name}")
        model = model_info["model"]
        model_type = model_info["type"]
        
        # Run inference
        debug_print(f"Running inference with model: {model_name}")
        try:
            predictions = run_inference(model, X, scaler, norm_cols)
            debug_print(f"Inference complete, obtained {len(predictions)} predictions")
        except Exception as e:
            debug_print(f"Error during inference: {str(e)}")
            traceback.print_exc()
            continue
        
        # Calculate metrics
        debug_print("Calculating metrics...")
        metrics = calculate_metrics(ground_truth, predictions)
        debug_print(f"Metrics calculated: MSE={metrics['mse']:.6f}, RMSE={metrics['rmse']:.6f}, R²={metrics['r2']:.6f}")
        
        # Generate plots
        debug_print("Generating visualization plots...")
        try:
            results_df = generate_plots(ground_truth, predictions, file_mapping, model_name, args.output_dir)
            debug_print("Plots generated successfully")
        except Exception as e:
            debug_print(f"Error generating plots: {str(e)}")
            traceback.print_exc()
            results_df = pd.DataFrame({
                'Ground Truth': ground_truth,
                'Prediction': predictions,
                'Error': predictions - ground_truth,
                'File': file_mapping
            })
        
        # Store top errors for reporting
        top_errors = results_df.copy()
        top_errors['Abs_Error'] = np.abs(top_errors['Error'])
        top_errors = top_errors.sort_values('Abs_Error', ascending=False).head(5)
        
        # Store results
        all_results[model_name] = {
            "predictions": predictions,
            "metrics": metrics,
            "results_df": results_df,
            "top_errors": top_errors,
            "model_type": model_type
        }
        
        # Print summary metrics
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R² Score: {metrics['r2']:.6f}")
    
    # If multiple models, generate comparative plots
    if len(all_results) > 1:
        debug_print("Generating comparative analysis...")
        try:
            generate_comparative_plots(all_results, args.output_dir)
            debug_print("Comparative plots generated successfully")
        except Exception as e:
            debug_print(f"Error generating comparative plots: {str(e)}")
            traceback.print_exc()
    
    # Generate the validation report
    debug_print("Generating validation report...")
    try:
        report_path = generate_validation_report(all_results, metadata, args, args.output_dir)
        debug_print(f"Validation report generated: {report_path}")
    except Exception as e:
        debug_print(f"Error generating validation report: {str(e)}")
        traceback.print_exc()
    
    # Export results to CSV
    debug_print("Exporting results to CSV...")
    try:
        export_results_csv(all_results, args.output_dir)
        debug_print("Results exported to CSV successfully")
    except Exception as e:
        debug_print(f"Error exporting results to CSV: {str(e)}")
        traceback.print_exc()
    
    debug_print("\nValidation complete! Results saved to: " + args.output_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)