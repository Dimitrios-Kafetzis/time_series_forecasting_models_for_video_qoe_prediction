# Time Series Forecasting Models for Video QoE Prediction

This repository provides a comprehensive framework for forecasting video Quality of Experience (QoE) based on network and mobility metrics. It includes several deep learning architectures (Linear Regressor, DNN, LSTM, GRU, Transformer) alongside utilities for dataset generation, hyperparameter tuning, model evaluation, and inference.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Mock Datasets](#generating-mock-datasets)
  - [Running Experiments](#running-experiments)
  - [Training All Models](#training-all-models)
  - [Testing All Models](#testing-all-models)
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
  - Linear Regressor (Baseline)
  - Simple DNN (Deep Neural Network)
  - LSTM (with and without self-attention)
  - GRU (with and without self-attention)
  - Transformer
 
- **Experimentation Pipelines:**
  Two experiment runners (experiment_runner.py and experiment_runner_v2.py) automate hyperparameter grid search and log results into CSV files for analysis.

- **Automated Model Training:**
  The train_all_models_v5.sh script automates training of multiple model variants with different configurations.

- **Comprehensive Model Evaluation:**
  The test_all_models.sh script evaluates all trained models and generates comparative performance reports.

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
  The project supports automated tuning for all model types via Keras Tuner.

## Directory Structure

The repository is organized as follows:

├── experiment_runner.py              # Experiment driver for standard dataset experiments  
├── experiment_runner_v2.py           # Enhanced experiment runner with augmented dataset support  
├── generate_mock_dataset.py          # Script to generate synthetic JSON datasets (dataset, inference, augmented)  
├── run_inference.py                  # Inference script for new JSON inputs using saved models  
├── test_models.py                    # Script for evaluating models and measuring inference latency  
├── timeseries_forecasting_models_v2.py  # Model definitions, training, and hyperparameter tuning (v2)  
├── timeseries_forecasting_models_v3.py  # Updated model definitions and training pipeline (v3)  
├── timeseries_forecasting_models_v5.py  # Latest model implementations with Linear, DNN, and self-attention models (v5)  
├── train_all_models_v5.sh            # Shell script to automate training of all model variants  
├── test_all_models.sh                # Shell script to test all models and generate comparison reports  
├── forecasting_models_v5/            # Directory containing trained model files  
│   ├── model_evaluation_report.txt   # Comprehensive model evaluation report  
│   ├── model_evaluation_results.csv  # CSV with evaluation metrics for all models  
│   ├── linear_basic.h5               # Pre-trained Linear model (basic)  
│   ├── linear_with_l1_reg.h5         # Pre-trained Linear model with L1 regularization  
│   ├── linear_with_l2_reg.h5         # Pre-trained Linear model with L2 regularization  
│   ├── linear_with_elastic_net.h5    # Pre-trained Linear model with ElasticNet regularization  
│   ├── dnn_basic.h5                  # Pre-trained DNN model (basic)  
│   ├── dnn_deep.h5                   # Pre-trained DNN with deeper architecture  
│   ├── dnn_with_elu.h5               # Pre-trained DNN with ELU activation  
│   ├── dnn_with_high_dropout.h5      # Pre-trained DNN with higher dropout  
│   ├── lstm_basic.h5                 # Pre-trained LSTM with self-attention  
│   ├── lstm_deep.h5                  # Pre-trained LSTM with multiple layers  
│   ├── lstm_wide.h5                  # Pre-trained LSTM with more hidden units  
│   ├── lstm_bidirectional.h5         # Pre-trained bidirectional LSTM  
│   ├── lstm_with_stats.h5            # Pre-trained LSTM with statistical features  
│   ├── gru_basic.h5                  # Pre-trained GRU with self-attention  
│   ├── gru_deep.h5                   # Pre-trained GRU with multiple layers  
│   ├── gru_wide.h5                   # Pre-trained GRU with more hidden units  
│   ├── gru_bidirectional.h5          # Pre-trained bidirectional GRU  
│   ├── gru_with_stats.h5             # Pre-trained GRU with statistical features  
│   ├── transformer_basic.h5          # Pre-trained Transformer (basic)  
│   ├── transformer_more_heads.h5     # Pre-trained Transformer with more attention heads  
│   ├── transformer_large_ff.h5       # Pre-trained Transformer with larger feed-forward dim  
│   ├── transformer_low_dropout.h5    # Pre-trained Transformer with low dropout  
│   ├── transformer_with_stats.h5     # Pre-trained Transformer with statistical features  
│   └── scaler.save                   # Saved scaler for feature normalization  
└── inference_inputs/                 # Folder containing JSON files for inference (e.g., 20250204123000.json)

## Installation

1. **System Requirements:**
   - Ubuntu 20.04+ (or compatible)
   - Python 3.8+
   - TensorFlow 2.x
   - CUDA-compatible GPU recommended for faster training

2. **Set Up a Virtual Environment (Recommended):**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies:**
```bash
pip install --upgrade pip
pip install numpy pandas tensorflow joblib matplotlib scikit-learn keras_tuner
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

### Training All Models

To train a comprehensive set of model variants with different architectures and configurations, use the train_all_models_v5.sh script:

```bash
./train_all_models_v5.sh
```

This script trains 21 different model configurations:
- 4 Linear Regressor variants (basic, L1 regularization, L2 regularization, ElasticNet)
- 4 Simple DNN variants (basic, deep, with ELU activation, with high dropout)
- 5 LSTM variants (basic, deep, wide, bidirectional, with stats)
- 5 GRU variants (basic, deep, wide, bidirectional, with stats)
- 5 Transformer variants (basic, more heads, large feed-forward, low dropout, with stats)

All models are saved to the forecasting_models_v5 directory with appropriate naming conventions.

### Testing All Models

To evaluate and compare all trained models, use the test_all_models.sh script:

```bash
./test_all_models.sh
```

This script:
1. Tests each model in the forecasting_models_v5 directory
2. Computes evaluation metrics (MSE, MAE, R² Score)
3. Measures inference latency for each model
4. Generates a comprehensive evaluation report (model_evaluation_report.txt)
5. Creates a CSV file with all results (model_evaluation_results.csv)

The report includes model rankings by each metric and identifies the best overall model based on a weighted combination of all metrics.

### Inference

Run the inference script to predict QoE for new input data:

- **Standard Mode:**
```bash
python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json --data_folder ./mock_dataset --seq_length 5 --model_file forecasting_models_v5/lstm_with_stats.h5 --scaler_file forecasting_models_v5/scaler.save
```

- **Augmented Mode with Statistical Features:**
```bash
python3 run_inference.py --inference_file ./inference_inputs/20250204123000.json --data_folder ./augmented_dataset --seq_length 5 --model_file forecasting_models_v5/lstm_with_stats.h5 --scaler_file forecasting_models_v5/scaler.save --augmented --use_stats
```

### Testing and Evaluation

Evaluate your trained models using:
```bash
python3 test_models.py --data_folder <folder_path> --model_file <model_file> --seq_length 5 --scaler_file forecasting_models_v5/scaler.save [--augmented] [--use_stats] [--simulate_device <device>]
```

**Example:**
```bash
python3 test_models.py --data_folder ./augmented_dataset --model_file forecasting_models_v5/gru_with_stats.h5 --seq_length 5 --scaler_file forecasting_models_v5/scaler.save --augmented --use_stats --simulate_device jetson
```

This script reports evaluation metrics such as MSE, MAE, R², and measures inference latency.

### Automated Hyperparameter Tuning

The timeseries_forecasting_models_v5.py supports automated hyperparameter tuning for all model types via Keras Tuner:

```bash
python3 timeseries_forecasting_models_v5.py --data_folder ./augmented_dataset --model_type lstm --seq_length 5 --epochs 20 --batch_size 16 --augmented --use_stats --tune --max_trials 10 --tune_epochs 20
```

This allows you to find optimal hyperparameters for any model type (linear, dnn, lstm, gru, transformer).

## Models

The latest version (v5) includes the following model types:

1. **Linear Regressor Models**: 
   - Basic implementation flattens input sequences and applies a single Dense layer
   - Variants with L1, L2, and ElasticNet regularization

2. **Simple DNN Models**:
   - Flatten input sequence and apply multiple Dense layers
   - Configurable activation functions, depth, and dropout rates

3. **LSTM Models with Self-Attention**:
   - Replace traditional pooling with self-attention mechanism
   - Variants with different depths, widths, and bidirectional configurations

4. **GRU Models with Self-Attention**:
   - Similar to LSTM but using GRU cells
   - Also implements self-attention for temporal feature aggregation

5. **Transformer Models**:
   - Implements transformer architecture for sequence modeling
   - Configurable attention heads, feed-forward dimensions, and dropout

## Distinction Between Timeseries Forecasting Model Versions

There are four versions of the model definition scripts:

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

- **timeseries_forecasting_models_v4.py:**
  - Builds upon v3 by replacing GlobalAveragePooling1D with a custom SelfAttention mechanism in the LSTM and GRU models.
  - Implements the SelfAttention layer class for enhanced temporal feature aggregation.
  - Maintains the three core model architectures: LSTM, GRU, and Transformer.
  - Uses log-cosh loss function and supports automated hyperparameter tuning.
  - Allows for bidirectional RNNs combined with the self-attention mechanism.
  
- **timeseries_forecasting_models_v5.py:**
  - Most comprehensive implementation with five model types: Linear Regressor, Simple DNN, LSTM, GRU, and Transformer.
  - Extends the SelfAttention approach from v4 to all recurrent models.
  - Adds multiple linear regressor variants with different regularization strategies (L1, L2, ElasticNet).
  - Implements configurable DNN models with various architectures, activations, and dropout rates.
  - Uses log-cosh loss for most models (except linear regressors which use MSE).
  - Enhanced hyperparameter tuning with model-specific parameter spaces.
  - Full support for statistical features with the `--use_stats` flag.
  
Choose the version that best suits your experimental needs or to compare performance differences.

### Model Validation

The repository includes a validation system to perform controlled experiments on trained models using known ground truth data. This allows for more realistic assessment of model performance beyond traditional test set evaluation.

#### Preparing a Validation Dataset

First, create a validation dataset with pairs of files (with/without QoE values) using:

```bash
python3 prepare_validation_data.py --input_folder ./real_dataset --output_folder ./validation_dataset --sample_ratio 0.2 --random_seed 42"
```

Options:
- `--input_folder`: Path to original dataset with ground truth QoE values
- `--output_folder`: Where to save the validation dataset
- `--sample_ratio`: Fraction of files to include in validation set (default: 1.0)
- `--random_seed`: Random seed for reproducible sampling
- `--legacy_format`: Use if your dataset is in legacy format

#### Validating Models

Run controlled validation experiments on one or more models:

```bash
python3 validate_models.py --validation_folder ./validation_dataset --model_dir ./forecasting_models_v5 --scaler_file ./forecasting_models_v5/scaler.save --output_dir ./validation_results --seq_length 5 --use_stats"
```

For a single model:
```bash
python3 validate_models.py --validation_folder ./validation_dataset --model_file ./forecasting_models_v5/model_lstm.h5 --scaler_file ./forecasting_models_v5/scaler.save --output_dir ./validation_results_lstm"
```

Options:
- `--validation_folder`: Path to validation dataset created by prepare_validation_data.py
- `--model_dir`: Directory containing all models to validate
- `--model_file`: Path to a single model file for individual validation
- `--scaler_file`: Path to the scaler file
- `--output_dir`: Where to save validation results and visualizations
- `--seq_length`: Sequence length used by the model(s)
- `--use_stats`: Include if models were trained with statistical features
- `--legacy_format`: Use if validation dataset is in legacy format

#### Integrated Testing and Validation

Use the enhanced test_all_models.sh script to run both testing and validation:

```bash
./test_all_models.sh --validate --prepare-validation --validation-sample 0.2"
```

Options:
- `--validate`: Enable validation after regular testing
- `--prepare-validation`: Create validation dataset before testing
- `--validation-sample <ratio>`: Sampling ratio for validation dataset
- `--validation-seed <num>`: Random seed for reproducible validation
- `--validation-folder <path>`: Custom validation dataset location
- `--validation-output <path>`: Custom validation results location

#### Validation Outputs

The validation process generates:

1. **Validation report** (validation_report.txt):
   - Comprehensive metrics for each model
   - Rankings based on various metrics (RMSE, MAE, R², bias)
   - Detailed analysis of largest errors

2. **Visualizations**:
   - Scatter plots of predictions vs ground truth
   - Error distribution histograms
   - Error vs ground truth plots
   - Top files with largest errors
   - Comparative visualizations across models

3. **CSV files** for further analysis:
   - validation_summary.csv: Summary metrics for all models
   - model_detailed_results.csv: File-by-file predictions and errors

## License

This project is licensed under the MIT License.

## Contact

For questions, feedback, or collaboration, please contact:
Dimitrios Kafetzis  (dimitrioskafetzis@gmail.com and kafetzis@aueb.gr)