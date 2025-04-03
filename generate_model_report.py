#!/usr/bin/env python3
"""
generate_model_report.py

This script takes the CSV results file produced by test_all_models.sh
and generates a summary report with model rankings.

Usage:
    python3 generate_model_report.py results.csv report.txt
"""

import sys
import pandas as pd
import numpy as np

def generate_report(csv_path, report_path):
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return

    # Check if we have valid data
    if len(df) == 0:
        print("No data found in CSV file")
        return

    # Convert null values to NaN
    df = df.replace("null", np.nan)
    
    # Convert metrics to numeric
    for col in ['MSE', 'MAE', 'R2', 'Inference Latency (ms)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing values
    valid_df = df.dropna()
    
    # If we have no valid data, exit
    if len(valid_df) == 0:
        with open(report_path, 'a') as f:
            f.write("\n\nSUMMARY REPORT\n")
            f.write("=============\n\n")
            f.write("No valid model evaluation results found.\n")
        return

    # Open the report file in append mode
    with open(report_path, 'a') as f:
        f.write("\n\nSUMMARY REPORT\n")
        f.write("=============\n\n")
        
        # Generate model type breakdown
        f.write("Models by Type:\n")
        type_counts = df['Model Type'].value_counts()
        for model_type, count in type_counts.items():
            f.write(f"- {model_type}: {count} models\n")
        f.write("\n")

        # Create rankings based on different metrics
        f.write("Model Rankings:\n\n")
        
        # MSE Ranking (lower is better)
        if 'MSE' in valid_df.columns:
            f.write("Top 5 Models by MSE (lower is better):\n")
            mse_ranking = valid_df.sort_values('MSE').head(5)
            for i, (_, row) in enumerate(mse_ranking.iterrows(), 1):
                f.write(f"{i}. {row['Model']} ({row['Model Type']}) - MSE: {row['MSE']:.6f}\n")
            f.write("\n")
        
        # MAE Ranking (lower is better)
        if 'MAE' in valid_df.columns:
            f.write("Top 5 Models by MAE (lower is better):\n")
            mae_ranking = valid_df.sort_values('MAE').head(5)
            for i, (_, row) in enumerate(mae_ranking.iterrows(), 1):
                f.write(f"{i}. {row['Model']} ({row['Model Type']}) - MAE: {row['MAE']:.6f}\n")
            f.write("\n")
        
        # R2 Ranking (higher is better)
        if 'R2' in valid_df.columns:
            f.write("Top 5 Models by R² Score (higher is better):\n")
            r2_ranking = valid_df.sort_values('R2', ascending=False).head(5)
            for i, (_, row) in enumerate(r2_ranking.iterrows(), 1):
                f.write(f"{i}. {row['Model']} ({row['Model Type']}) - R²: {row['R2']:.6f}\n")
            f.write("\n")
        
        # Latency Ranking (lower is better)
        if 'Inference Latency (ms)' in valid_df.columns:
            f.write("Top 5 Models by Inference Latency (lower is better):\n")
            latency_ranking = valid_df.sort_values('Inference Latency (ms)').head(5)
            for i, (_, row) in enumerate(latency_ranking.iterrows(), 1):
                f.write(f"{i}. {row['Model']} ({row['Model Type']}) - Latency: {row['Inference Latency (ms)']:.2f} ms\n")
            f.write("\n")

        # Best Overall Model (based on a composite score)
        f.write("Best Overall Models (ranked by composite score):\n")
        
        # Create a composite score: normalize each metric to 0-1 range and then combine them
        # For MSE and MAE: lower is better, so we invert these scores
        # For R2: higher is better
        # For Latency: lower is better, so we invert this score
        
        valid_metrics_df = valid_df.copy()
        
        # Only include columns that are present and have valid data
        metrics_to_use = []
        
        if 'MSE' in valid_metrics_df.columns and not valid_metrics_df['MSE'].isna().all():
            metrics_to_use.append('MSE')
            min_mse = valid_metrics_df['MSE'].min()
            max_mse = valid_metrics_df['MSE'].max()
            if min_mse != max_mse:
                valid_metrics_df['MSE_score'] = 1 - ((valid_metrics_df['MSE'] - min_mse) / (max_mse - min_mse))
            else:
                valid_metrics_df['MSE_score'] = 1
        
        if 'MAE' in valid_metrics_df.columns and not valid_metrics_df['MAE'].isna().all():
            metrics_to_use.append('MAE')
            min_mae = valid_metrics_df['MAE'].min()
            max_mae = valid_metrics_df['MAE'].max()
            if min_mae != max_mae:
                valid_metrics_df['MAE_score'] = 1 - ((valid_metrics_df['MAE'] - min_mae) / (max_mae - min_mae))
            else:
                valid_metrics_df['MAE_score'] = 1
        
        if 'R2' in valid_metrics_df.columns and not valid_metrics_df['R2'].isna().all():
            metrics_to_use.append('R2')
            min_r2 = valid_metrics_df['R2'].min()
            max_r2 = valid_metrics_df['R2'].max()
            if min_r2 != max_r2:
                valid_metrics_df['R2_score'] = (valid_metrics_df['R2'] - min_r2) / (max_r2 - min_r2)
            else:
                valid_metrics_df['R2_score'] = 1
        
        if 'Inference Latency (ms)' in valid_metrics_df.columns and not valid_metrics_df['Inference Latency (ms)'].isna().all():
            metrics_to_use.append('Inference Latency (ms)')
            min_latency = valid_metrics_df['Inference Latency (ms)'].min()
            max_latency = valid_metrics_df['Inference Latency (ms)'].max()
            if min_latency != max_latency:
                valid_metrics_df['Latency_score'] = 1 - ((valid_metrics_df['Inference Latency (ms)'] - min_latency) / (max_latency - min_latency))
            else:
                valid_metrics_df['Latency_score'] = 1
        
        # Explain which metrics were used for the composite score
        f.write(f"(Composite score based on normalized: {', '.join(metrics_to_use)})\n")
        
        # Calculate the composite score
        score_columns = [col for col in valid_metrics_df.columns if col.endswith('_score')]
        if score_columns:
            valid_metrics_df['composite_score'] = valid_metrics_df[score_columns].mean(axis=1)
            
            # Rank by composite score
            composite_ranking = valid_metrics_df.sort_values('composite_score', ascending=False).head(5)
            for i, (_, row) in enumerate(composite_ranking.iterrows(), 1):
                metrics_info = []
                if 'MSE' in metrics_to_use:
                    metrics_info.append(f"MSE: {row['MSE']:.6f}")
                if 'MAE' in metrics_to_use:
                    metrics_info.append(f"MAE: {row['MAE']:.6f}")
                if 'R2' in metrics_to_use:
                    metrics_info.append(f"R²: {row['R2']:.6f}")
                if 'Inference Latency (ms)' in metrics_to_use:
                    metrics_info.append(f"Latency: {row['Inference Latency (ms)']:.2f} ms")
                
                metrics_str = ", ".join(metrics_info)
                f.write(f"{i}. {row['Model']} ({row['Model Type']}) - Score: {row['composite_score']:.4f} ({metrics_str})\n")
        else:
            f.write("Could not calculate composite scores due to missing metrics\n")
        
        f.write("\n")
        
        # Model type statistics - average performance by model type
        f.write("Average Performance by Model Type:\n")
        model_type_stats = valid_metrics_df.groupby('Model Type').mean()
        
        for model_type in model_type_stats.index:
            f.write(f"\n{model_type}:\n")
            if 'MSE' in metrics_to_use:
                f.write(f"  Avg MSE: {model_type_stats.loc[model_type, 'MSE']:.6f}\n")
            if 'MAE' in metrics_to_use:
                f.write(f"  Avg MAE: {model_type_stats.loc[model_type, 'MAE']:.6f}\n")
            if 'R2' in metrics_to_use:
                f.write(f"  Avg R²: {model_type_stats.loc[model_type, 'R2']:.6f}\n")
            if 'Inference Latency (ms)' in metrics_to_use:
                f.write(f"  Avg Latency: {model_type_stats.loc[model_type, 'Inference Latency (ms)']:.2f} ms\n")
        
        f.write("\n")
        f.write("=============\n")
        f.write("End of Report\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_model_report.py results.csv report.txt")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    report_path = sys.argv[2]
    
    generate_report(csv_path, report_path)
    print("Report generation complete!")