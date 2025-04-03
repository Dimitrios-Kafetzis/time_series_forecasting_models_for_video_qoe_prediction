#!/bin/bash
# train_all_models_v5.sh
# Script to train all time series forecasting models with various configurations
# using the augmented dataset with the new format (10-second windows with 2-second intervals).
# UPDATED: All models now use the new dataset format by default and the --use_stats flag for feature consistency

# Define the output directory for models
OUTPUT_DIR=~/Impact-xG_prediction_model/forecasting_models_v5
# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Define the path to your augmented dataset
AUGMENTED_DATASET="./real_dataset"  # Change this to your actual dataset path

# Define common parameters
SEQ_LENGTH=5
EPOCHS=20
BATCH_SIZE=16

# Log file for training outputs
LOG_FILE="$OUTPUT_DIR/training_log.txt"

echo "Starting training of all models at $(date)" | tee "$LOG_FILE"
echo "Models will be saved in $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "All models will be trained with the new dataset format (default) and --use_stats flag enabled" | tee -a "$LOG_FILE"

# Function to run training with specified parameters
train_model() {
    local model_type=$1
    local config_name=$2
    local extra_args=$3
    
    echo "" | tee -a "$LOG_FILE"
    echo "=====================================================" | tee -a "$LOG_FILE"
    echo "Training $model_type model with config: $config_name" | tee -a "$LOG_FILE"
    echo "Starting at $(date)" | tee -a "$LOG_FILE"
    echo "=====================================================" | tee -a "$LOG_FILE"
    
    # Construct the complete command with --use_stats for all models
    # The new format is now used by default when --augmented is specified
    local cmd="python3 timeseries_forecasting_models_v5.py \
                --data_folder $AUGMENTED_DATASET \
                --model_type $model_type \
                --seq_length $SEQ_LENGTH \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --augmented \
                --use_stats \
                $extra_args"
    
    echo "Command: $cmd" | tee -a "$LOG_FILE"
    
    # Execute the command and tee output to log file
    eval "$cmd 2>&1 | tee -a $LOG_FILE"
    
    # The model will be saved directly to the output directory by the Python script
    # but we can rename it with a more descriptive name
    local original_model="$OUTPUT_DIR/model_${model_type}_with_attention.h5"
    local renamed_model="$OUTPUT_DIR/${model_type}_${config_name}.h5"
    
    # Wait a moment to ensure the file is completely written
    sleep 2
    
    # Check if the original model exists before attempting to rename
    if [ -f "$original_model" ]; then
        mv "$original_model" "$renamed_model"
        echo "Model renamed to: $renamed_model" | tee -a "$LOG_FILE"
    else
        echo "Warning: Original model file not found at $original_model" | tee -a "$LOG_FILE"
        # List directory contents for debugging
        echo "Directory contents:" | tee -a "$LOG_FILE"
        ls -la "$OUTPUT_DIR" | tee -a "$LOG_FILE"
    fi
    
    echo "Completed at $(date)" | tee -a "$LOG_FILE"
    echo "=====================================================" | tee -a "$LOG_FILE"
}

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

# 1. Train Linear Models
# 1.1 Basic Linear Model
train_model "linear" "basic" ""

# 1.2 Linear Model with L1 Regularization
train_model "linear" "with_l1_reg" "--l1_reg 0.01"

# 1.3 Linear Model with L2 Regularization
train_model "linear" "with_l2_reg" "--l2_reg 0.01"

# 1.4 Linear Model with both L1 and L2 Regularization (ElasticNet)
train_model "linear" "with_elastic_net" "--l1_reg 0.005 --l2_reg 0.005"

# 2. Train DNN Models
# 2.1 Basic DNN with default configuration
train_model "dnn" "basic" ""

# 2.2 DNN with deeper architecture
train_model "dnn" "deep" "--hidden_layers 128,64,32"

# 2.3 DNN with ELU activation
train_model "dnn" "with_elu" "--activation elu"

# 2.4 DNN with higher dropout for regularization
train_model "dnn" "with_high_dropout" "--dropout_rate 0.4"

# 3. Train LSTM Models
# 3.1 Basic LSTM with self-attention
train_model "lstm" "basic" "--attention_units 128"

# 3.2 LSTM with more layers
train_model "lstm" "deep" "--num_layers 3 --attention_units 128"

# 3.3 LSTM with more hidden units
train_model "lstm" "wide" "--hidden_units 100 --attention_units 128"

# 3.4 Bidirectional LSTM
train_model "lstm" "bidirectional" "--bidirectional --attention_units 128"

# 3.5 LSTM with explicit stats naming (for consistency)
train_model "lstm" "with_stats" "--attention_units 128"

# 4. Train GRU Models
# 4.1 Basic GRU with self-attention
train_model "gru" "basic" "--attention_units 128"

# 4.2 GRU with more layers
train_model "gru" "deep" "--num_layers 3 --attention_units 128"

# 4.3 GRU with more hidden units
train_model "gru" "wide" "--hidden_units 100 --attention_units 128"

# 4.4 Bidirectional GRU
train_model "gru" "bidirectional" "--bidirectional --attention_units 128"

# 4.5 GRU with explicit stats naming (for consistency)
train_model "gru" "with_stats" "--attention_units 128"

# 5. Train Transformer Models
# 5.1 Basic Transformer
train_model "transformer" "basic" ""

# 5.2 Transformer with more heads
train_model "transformer" "more_heads" "--num_heads 4"

# 5.3 Transformer with larger feed-forward dimension
train_model "transformer" "large_ff" "--ff_dim 128"

# 5.4 Transformer with lower dropout
train_model "transformer" "low_dropout" "--dropout_rate 0.05"

# 5.5 Transformer with explicit stats naming (for consistency)
train_model "transformer" "with_stats" ""

# The scaler file should already be in the output directory
# No need to copy it, just verify it exists
if [ -f "$OUTPUT_DIR/scaler.save" ]; then
    echo "Scaler file exists at $OUTPUT_DIR/scaler.save" | tee -a "$LOG_FILE"
else
    echo "Warning: Scaler file not found at $OUTPUT_DIR/scaler.save" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"
echo "All models trained successfully!" | tee -a "$LOG_FILE"
echo "Models are saved in $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Training completed at $(date)" | tee -a "$LOG_FILE"
echo "=====================================================" | tee -a "$LOG_FILE"