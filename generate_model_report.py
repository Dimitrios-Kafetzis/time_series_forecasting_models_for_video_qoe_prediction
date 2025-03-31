#!/usr/bin/env python3
"""
generate_model_report.py

This script processes the CSV file of model evaluation results and generates
a formatted summary section for the evaluation report, including model rankings
by different metrics.

Usage:
    python3 generate_model_report.py <results_csv> <report_file>
"""

import sys
import pandas as pd
import numpy as np
from tabulate import tabulate

def generate_summary(csv_file, report_file):
    """
    Generate a summary of model performance and append it to the report file.
    
    Args:
        csv_file: Path to the CSV file containing model evaluation results
        report_file: Path to the report file where the summary will be appended
    """
    # Load the results
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Check if we have data
    if len(df) == 0:
        print("No data found in the CSV file")
        return
    
    # Convert metrics to numeric types
    numeric_cols = ['MSE', 'MAE', 'R2', 'Inference Latency (ms)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create a summary section to append to the report
    with open(report_file, 'a') as f:
        f.write("=======================================================\n")
        f.write("                    SUMMARY REPORT                     \n")
        f.write("=======================================================\n\n")
        
        # Overall performance table
        f.write("Model Performance Rankings:\n\n")
        
        # Best models by different metrics
        metrics = {
            'MSE': {'best': 'min', 'description': 'Mean Squared Error (lower is better)'},
            'MAE': {'best': 'min', 'description': 'Mean Absolute Error (lower is better)'},
            'R2': {'best': 'max', 'description': 'R² Score (higher is better)'},
            'Inference Latency (ms)': {'best': 'min', 'description': 'Inference Latency in ms (lower is better)'}
        }
        
        for metric, info in metrics.items():
            f.write(f"Top 5 Models by {info['description']}:\n")
            
            # Sort based on whether lower or higher is better
            ascending = info['best'] == 'min'
            top_models = df.sort_values(by=metric, ascending=ascending).head(5)
            
            # Format the table
            table = []
            for i, (_, row) in enumerate(top_models.iterrows()):
                table.append([
                    i+1, 
                    row['Model'], 
                    row['Model Type'], 
                    f"{row[metric]:.6f}"
                ])
            
            # Write the table
            headers = ["Rank", "Model", "Type", metric]
            f.write(tabulate(table, headers=headers, tablefmt="grid"))
            f.write("\n\n")
        
        # Group performance by model type
        f.write("Average Performance by Model Type:\n\n")
        model_type_groups = df.groupby('Model Type').agg({
            'MSE': 'mean',
            'MAE': 'mean',
            'R2': 'mean',
            'Inference Latency (ms)': 'mean'
        }).reset_index()
        
        # Format the table
        type_table = []
        for _, row in model_type_groups.iterrows():
            type_table.append([
                row['Model Type'],
                f"{row['MSE']:.6f}",
                f"{row['MAE']:.6f}",
                f"{row['R2']:.6f}",
                f"{row['Inference Latency (ms)']:.2f}"
            ])
        
        # Write the table
        type_headers = ["Model Type", "Avg MSE", "Avg MAE", "Avg R²", "Avg Latency (ms)"]
        f.write(tabulate(type_table, headers=type_headers, tablefmt="grid"))
        f.write("\n\n")
        
        # Overall best model considering all metrics
        f.write("Overall Best Model:\n\n")
        
        # Normalize metrics to 0-1 range for fair comparison
        for col in ['MSE', 'MAE', 'Inference Latency (ms)']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col} (normalized)'] = 1 - ((df[col] - min_val) / (max_val - min_val))
            else:
                df[f'{col} (normalized)'] = 1.0
        
        # For R2, higher is better, so normalization is different
        min_val = df['R2'].min()
        max_val = df['R2'].max()
        if max_val > min_val:
            df['R2 (normalized)'] = (df['R2'] - min_val) / (max_val - min_val)
        else:
            df['R2 (normalized)'] = 1.0
        
        # Calculate overall score
        # We'll weight accuracy metrics higher than latency
        df['Overall Score'] = (
            df['MSE (normalized)'] * 0.3 + 
            df['MAE (normalized)'] * 0.3 + 
            df['R2 (normalized)'] * 0.3 + 
            df['Inference Latency (ms) (normalized)'] * 0.1
        )
        
        # Get the best overall model
        best_model = df.loc[df['Overall Score'].idxmax()]
        
        f.write(f"Based on a weighted combination of all metrics, the best overall model is:\n")
        f.write(f"Model: {best_model['Model']}\n")
        f.write(f"Type: {best_model['Model Type']}\n")
        f.write(f"MSE: {best_model['MSE']:.6f}\n")
        f.write(f"MAE: {best_model['MAE']:.6f}\n")
        f.write(f"R² Score: {best_model['R2']:.6f}\n")
        f.write(f"Inference Latency: {best_model['Inference Latency (ms)']:.2f} ms\n")
        f.write(f"Overall Score: {best_model['Overall Score']:.4f} (higher is better)\n\n")
        
        # Complete model ranking
        f.write("Complete Model Ranking (based on overall score):\n\n")
        ranked_models = df.sort_values(by='Overall Score', ascending=False)
        
        # Format the table
        ranking_table = []
        for i, (_, row) in enumerate(ranked_models.iterrows()):
            ranking_table.append([
                i+1,
                row['Model'],
                row['Model Type'],
                f"{row['MSE']:.6f}",
                f"{row['MAE']:.6f}",
                f"{row['R2']:.6f}",
                f"{row['Inference Latency (ms)']:.2f}",
                f"{row['Overall Score']:.4f}"
            ])
        
        # Write the table
        ranking_headers = ["Rank", "Model", "Type", "MSE", "MAE", "R²", "Latency (ms)", "Score"]
        f.write(tabulate(ranking_table, headers=ranking_headers, tablefmt="grid"))
        f.write("\n\n")
        
        # Add timestamp and footer
        import datetime
        f.write(f"Report generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=======================================================\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_model_report.py <results_csv> <report_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    report_file = sys.argv[2]
    
    generate_summary(csv_file, report_file)
    print(f"Summary report generated and appended to {report_file}")