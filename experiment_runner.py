#!/usr/bin/env python3
"""
Filename: experiment_runner.py
Author: Dimitrios Kafetzis
Creation Date: 2025-02-04
Description:
    This script automates the experimentation process for time series forecasting models.
    It runs the `timeseries_forecasting_models.py` script with various configurations
    for each model type (LSTM, GRU, Transformer), captures the output (final test loss),
    and logs the configuration along with the results into a CSV file.
    
    At the end, you can analyze the CSV file to determine the best configuration for
    each model type.
Usage Example:
    $ python3 experiment_runner.py --data_folder ./mock_dataset --epochs 20 --batch_size 16
"""

"""

Final proposition for the best configuration per model:

>Fine Tuning results for the 1000points dataset (120225):
------------------------------
 model_type  seq_length  hidden_units  num_layers  dropout_rate  learning_rate bidirectional  num_heads  ff_dim  test_loss
        gru          10         100.0         2.0           0.2         0.0005          True        NaN     NaN   0.102298
       lstm          10         100.0         2.0           0.2         0.0005         False        NaN     NaN   0.099032
transformer           5           NaN         NaN           0.2         0.0010           NaN        2.0   128.0   0.108765
---------------------------------------------------------------------------------------------------------------------------

>Fine Tuning results for the 100000points dataset (160225):
------------------------------
 model_type  seq_length  hidden_units  num_layers  dropout_rate  learning_rate bidirectional  num_heads  ff_dim  test_loss
        gru          10          50.0         2.0           0.2         0.0010         False        NaN     NaN   0.042484
       lstm          10         100.0         2.0           0.2         0.0010         False        NaN     NaN   0.042484
transformer          10           NaN         NaN           0.2         0.0005           NaN        4.0   128.0   0.042481

>Fine Tuning results for the 100000points augmented dataset (200225):
 model_type  seq_length  hidden_units  num_layers  dropout_rate  learning_rate bidirectional  num_heads  ff_dim  test_loss
        gru          10         100.0         2.0           0.3         0.0005         False        NaN     NaN   0.016783
       lstm          10          50.0         2.0           0.2         0.0005          True        NaN     NaN   0.016784
transformer          10           NaN         NaN           0.2         0.0010           NaN        2.0   128.0   0.016787


"""

import os
import csv
import subprocess
import itertools
import argparse
import pandas as pd

def run_experiment(cmd):
    """
    Runs a command using subprocess and returns the output.
    We assume that timeseries_forecasting_models.py prints the final test loss in a line starting with "Test Loss:".
    """
    print("Running command:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        output = result.stdout
        # Look for a line starting with "Test Loss:" and extract the numeric value.
        test_loss = None
        for line in output.splitlines():
            if line.startswith("Test Loss:"):
                try:
                    test_loss = float(line.split(":")[1].strip())
                except Exception as e:
                    test_loss = None
        return test_loss, output
    except subprocess.CalledProcessError as e:
        print("Error running command:", e.stderr)
        return None, e.stderr

def analyze_results(results_file='experiment_results.csv', output_file='best_results.csv'):
    """
    Analyze the experiment results stored in results_file and print
    (and optionally save) the best configuration (lowest test_loss)
    for each model type.
    """
    try:
        # Load the results CSV
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return

    # Ensure that the 'test_loss' column is numeric
    df['test_loss'] = pd.to_numeric(df['test_loss'], errors='coerce')
    
    # Drop rows where test_loss could not be converted
    df = df.dropna(subset=['test_loss'])

    # Group by model_type and get the index of the row with the lowest test_loss per group
    best_idx = df.groupby('model_type')['test_loss'].idxmin()
    best_df = df.loc[best_idx].reset_index(drop=True)
    
    print("\nFinal proposition for the best configuration per model:")
    # Select the columns you want to display; adjust as needed.
    display_cols = ['model_type', 'seq_length', 'hidden_units', 'num_layers',
                    'dropout_rate', 'learning_rate', 'bidirectional', 'num_heads', 'ff_dim', 'test_loss']
    print(best_df[display_cols].to_string(index=False))

    # Optionally, save the best configurations to a separate CSV file.
    try:
        best_df.to_csv(output_file, index=False)
        print(f"\nBest configurations saved to: {output_file}")
    except Exception as e:
        print(f"Error saving best configurations: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Path to folder containing JSON files for training/testing.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs for each run.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--output_csv", type=str, default="experiment_results.csv", help="CSV file to store results.")
    args = parser.parse_args()

    # Define the hyperparameter grid for each model type.
    model_types = ["lstm", "gru", "transformer"]

    # For simplicity, we define a small grid.
    seq_lengths = [5, 10] 
    hidden_units = [50, 100]  # Only relevant for LSTM/GRU
    num_layers = [1, 2]
    dropout_rates = [0.2, 0.3]
    learning_rates = [0.001, 0.0005]
    bidirectional_options = [False, True]  # Only for LSTM/GRU

    # For Transformer-specific parameters:
    num_heads_options = [2, 4]
    ff_dims = [64, 128]

    # Build a list of all configurations as dictionaries.
    experiments = []
    for model_type in model_types:
        if model_type in ["lstm", "gru"]:
            grid = list(itertools.product(seq_lengths, hidden_units, num_layers, dropout_rates, learning_rates, bidirectional_options))
            for (seq_length, hu, nl, dr, lr, bidir) in grid:
                config = {
                    "model_type": model_type,
                    "seq_length": seq_length,
                    "hidden_units": hu,
                    "num_layers": nl,
                    "dropout_rate": dr,
                    "learning_rate": lr,
                    "bidirectional": bidir
                }
                experiments.append(config)
        elif model_type == "transformer":
            grid = list(itertools.product(seq_lengths, dropout_rates, learning_rates, num_heads_options, ff_dims))
            for (seq_length, dr, lr, nh, ffd) in grid:
                config = {
                    "model_type": model_type,
                    "seq_length": seq_length,
                    "dropout_rate": dr,
                    "learning_rate": lr,
                    "num_heads": nh,
                    "ff_dim": ffd
                }
                experiments.append(config)
    
    # Prepare CSV file header.
    header = ["model_type", "seq_length", "hidden_units", "num_layers", "dropout_rate", "learning_rate", "bidirectional",
              "num_heads", "ff_dim", "test_loss", "run_output"]

    # Open the CSV file for writing results.
    with open(args.output_csv, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        # Loop over each experiment.
        for config in experiments:
            # Build the command to run timeseries_forecasting_models.py with these parameters.
            cmd = ["python3", "timeseries_forecasting_models.py",
                   "--data_folder", args.data_folder,
                   "--epochs", str(args.epochs),
                   "--batch_size", str(args.batch_size),
                   "--model_type", config["model_type"],
                   "--seq_length", str(config["seq_length"]),
                   "--dropout_rate", str(config["dropout_rate"]),
                   "--learning_rate", str(config["learning_rate"])]
            # Add parameters based on model type.
            if config["model_type"] in ["lstm", "gru"]:
                cmd.extend(["--hidden_units", str(config["hidden_units"]),
                            "--num_layers", str(config["num_layers"])])
                if config["bidirectional"]:
                    cmd.append("--bidirectional")
                # For consistency, we add dummy transformer params.
                cmd.extend(["--num_heads", "2", "--ff_dim", "64"])
            elif config["model_type"] == "transformer":
                cmd.extend(["--num_heads", str(config["num_heads"]),
                            "--ff_dim", str(config["ff_dim"])])
                # For consistency, add dummy RNN params.
                cmd.extend(["--hidden_units", "50", "--num_layers", "1"])
            # Print current configuration.
            print("Configuration:", config)
            # Run the experiment.
            test_loss, output = run_experiment(cmd)
            # Log the configuration and result.
            row = {
                "model_type": config["model_type"],
                "seq_length": config["seq_length"],
                "hidden_units": config.get("hidden_units", ""),
                "num_layers": config.get("num_layers", ""),
                "dropout_rate": config["dropout_rate"],
                "learning_rate": config["learning_rate"],
                "bidirectional": config.get("bidirectional", ""),
                "num_heads": config.get("num_heads", ""),
                "ff_dim": config.get("ff_dim", ""),
                "test_loss": test_loss,
                "run_output": output.replace("\n", " | ")
            }
            writer.writerow(row)
            csvfile.flush()  # Write results immediately
            print("Finished configuration with Test Loss:", test_loss)
    
    print("Experimentation completed. Results saved to", args.output_csv)

    # After experiments, run the analysis step:
    analyze_results()

if __name__ == "__main__":
    main()
