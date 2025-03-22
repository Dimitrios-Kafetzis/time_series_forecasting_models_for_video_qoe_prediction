# Time Series Forecasting Models for Video QoE Prediction

This repository provides a comprehensive framework for forecasting video Quality of Experience (QoE) based on network and mobility metrics. It includes several deep learning architectures (LSTM, GRU, Transformer) alongside utilities for dataset generation, hyperparameter tuning, model evaluation, and inference.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Mock Datasets](#generating-mock-datasets)
  - [Running Experiments](#running-experiments)
  - [Inference](#inference)
  - [Testing and Evaluation](#testing-and-evaluation)
  - [Automated Hyperparameter Tuning](#automated-hyperparameter-tuning)
- [Models](#models)
- [Distinction Between Timeseries Forecasting Model Versions](#distinction-between-timeseries-forecasting-model-versions)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Video streaming applications require constant monitoring of network conditions to maintain high user Quality of Experience (QoE). This project implements time series forecasting models using TensorFlow and Keras to predict QoE based on network metrics like packet loss, jitter, throughput, and speed. The project supports both standard and augmented dataset formats, enabling robust experimentation and real-world inference scenarios.

## Features

- **Multiple Model Architectures:**
  - LSTM
  - GRU
  - Transformer
  - Linear Regressor (Baseline)

- **Experimentation Pipelines:**
  Two experiment runners (experiment_runner.py and experiment_runner_v2.py) automate hyperparameter grid search and log results into CSV files for analysis.

- **Synthetic Dataset Generation:**
  The generate_mock_dataset.py script creates JSON datasets in three modes:
  - **Dataset Mode:** Complete training/testing data with QoE values.
  - **Inference Mode:** JSON files with QoE set to null for inference testing.
  - **Augmented Mode:** 5‑second windows with per‑second measurements and aggregated QoE, with optional statistical features.

- **Inference Support:**
  The run_inference.py script loads a saved model and scaler, processes new JSON input (supporting both standard and augmented formats), and outputs a predicted QoE value.

- **Model Evaluation and Testing:**
  The test_models.py script evaluates models using regression metrics (MSE, MAE, R²) and measures inference latency, with options to simulate performance on different hardware (e.g., Xeon, Jetson).

- **Automated Hyperparameter Tuning:**
  The project supports automated tuning for LSTM, GRU, and Transformer models via Keras Tuner.

## Directory Structure

The repository is organized as follows:

├── experiment_runner.py              # Experiment driver for standard dataset experiments  
├── experiment_runner_v2.py           # Enhanced experiment runner with augmented dataset support  
├── generate_mock_dataset.py          # Script to generate synthetic JSON datasets (dataset, inference, augmented)  
├── run_inference.py                  # Inference script for new JSON inputs using saved models  
├── test_models.py                    # Script for evaluating models and measuring inference latency  
├── timeseries_forecasting_models_v2.py  # Model definitions, training, and hyperparameter tuning (v2)  
├── timeseries_forecasting_models_v3.py  # Updated model definitions and training pipeline (v3)  
├── model_gru.h5                      # Pre-trained GRU model  
├── model_linear.h5                   # Pre-trained Linear Regressor model  
├── model_lstm.h5                     # Pre-trained LSTM model  
├── model_transformer.h5              # Pre-trained Transformer model  
├── scaler.save                       # Saved scaler for feature normalization  
└── inference_inputs/                 # Folder containing JSON files for inference (e.g., 20250204123000.json)

## Installation

1. **System Requirements:**
   - Ubuntu 20.04+ (or compatible)
   - Python 3.8+

2. **Set Up a Virtual Environment (Recommended):**
```bash
python3 -m venv venv source venv/bin/activate
```

3. **Install Dependencies:**
```bash
pip install --upgrade pip pip install numpy pandas tensorflow joblib matplotlib scikit-learn keras_tuner
```

## Usage

### Generating Mock Datasets

Generate synthetic JSON datasets using:
```bash
python3 generate_mock_dataset.py --output_folder <folder_path> --mode <mode> --num_points <N> --start_timestamp <YYYYMMDDHHMMSS>
```

**Examples:**

- **Dataset Mode (Training/Testing):**
```bash
python3 generate_mock_dataset.py --output_folder ./mock_dataset --mode dataset --num_points 100 --start_timestamp 20250130114158
```

- **Inference Mode:**
```bash
python3 generate_mock_dataset.py --output_folder ./inference_inputs --mode inference --num_points 1 --start_timestamp 20250204123000
```

- **Augmented Mode (with per‑second granularity):**
```bash
python3 generate_mock_dataset.py --output_folder ./augmented_dataset --mode augmented --num_points 100 --start_timestamp 20250130114158
```

### Running Experiments

Two experiment runners are provided:

- **Standard Experiment Runner:**
```bash
python3 experiment_runner.py --data_folder ./mock_dataset --epochs 20 --batch_size 16
```

- **Enhanced Experiment Runner (Augmented Dataset Support):**
```bash
python3 experiment_runner_v2.py --data_folder ./augmented_dataset --epochs 20 --batch_size 16 --augmented
```

Both scripts iterate over a grid of hyperparameter configurations, log the results to CSV files, and analyze them to determine the best configuration based on test loss.

### Inference

Run the inference script to predict QoE for new input data:

- **Standard Mode:**
```bash
python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json --data_folder ./mock_dataset --seq_length 5 --model_file model_transformer.h5 --scaler_file scaler.save
```

- **Augmented Mode:**
```bash
python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json --data_folder ./augmented_dataset --seq_length 5 --model_file model_transformer.h5 --scaler_file scaler.save --augmented
```

### Testing and Evaluation

Evaluate your trained models using:
```bash
python3 test_models.py --data_folder <folder_path> --model_file <model_file> --seq_length 5 --scaler_file scaler.save [--augmented] [--use_stats] [--simulate_device <device>]
```

**Example:**
```bash
python3 test_models.py --data_folder ./augmented_dataset --model_file model_gru.h5 --seq_length 5 --scaler_file scaler.save --augmented --use_stats --simulate_device jetson
```

This script reports evaluation metrics such as MSE, MAE, R², and measures inference latency.

### Automated Hyperparameter Tuning

Both timeseries_forecasting_models_v2.py and timeseries_forecasting_models_v3.py support automated hyperparameter tuning via Keras Tuner.

For example:
```bash
python3 timeseries_forecasting_models_v2.py --data_folder ./mock_dataset --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --tune --max_trials 10 --tune_epochs 20
```
Or with augmented datasets:
```bash
python3 timeseries_forecasting_models_v2.py --data_folder ./augmented_dataset --model_type transformer --seq_length 5 --epochs 20 --batch_size 16 --augmented --use_stats --tune --max_trials 10 --tune_epochs 20
```

## Models

Pre-trained model files included:
- **GRU:** model_gru.h5
- **Linear Regressor:** model_linear.h5
- **LSTM:** model_lstm.h5
- **Transformer:** model_transformer.h5

These can be used directly for inference or serve as starting points for further training.

## Distinction Between Timeseries Forecasting Model Versions

There are two versions of the model definition scripts:

- **timeseries_forecasting_models_v2.py:**
  - Implements LSTM, GRU, Transformer, and a Linear Regressor baseline.
  - Uses Mean Squared Error (MSE) as the loss function.
  - Contains a simpler architecture without explicit temporal pooling.
  - Provides support for automated hyperparameter tuning via Keras Tuner.

- **timeseries_forecasting_models_v3.py:**
  - Focuses on LSTM, GRU, and Transformer models (linear regressor is not included).
  - Implements architectural enhancements such as forcing all recurrent layers to output sequences and applying GlobalAveragePooling1D to aggregate temporal information.
  - Uses the log-cosh loss function for potentially smoother convergence.
  - Also supports hyperparameter tuning via Keras Tuner.
  
Choose the version that best suits your experimental needs or to compare performance differences.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes.
4. Push your branch and open a Pull Request.

Ensure your code follows the project’s style and includes relevant tests.

## License

This project is licensed under the MIT License.

## Contact

For questions, feedback, or collaboration, please contact:
Dimitrios Kafetzis  (dimitrioskafetzis@gmail.com and kafetzis@aueb.gr)
