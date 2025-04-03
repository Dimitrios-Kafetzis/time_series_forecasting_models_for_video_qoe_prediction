#!/bin/bash
# test_all_models.sh
# Script to run testing on all models in the forecasting_models_v5 directory
# and generate a comprehensive evaluation report
# UPDATED: Now uses the new dataset format by default (10-second windows with 2-second intervals)

# Define paths
MODELS_DIR=~/Impact-xG_prediction_model/forecasting_models_v5
DATA_FOLDER="./real_dataset"  # Change this to your actual dataset path
SCALER_FILE="$MODELS_DIR/scaler.save"
SEQ_LENGTH=5
REPORT_FILE="$MODELS_DIR/model_evaluation_report.txt"
RESULTS_CSV="$MODELS_DIR/model_evaluation_results.csv"

# Make sure the directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory not found at $MODELS_DIR"
    exit 1
fi

# Make sure the scaler file exists
if [ ! -f "$SCALER_FILE" ]; then
    echo "Error: Scaler file not found at $SCALER_FILE"
    exit 1
fi

# Initialize the report file
echo "=======================================================" > "$REPORT_FILE"
echo "        TIME SERIES FORECASTING MODELS EVALUATION      " >> "$REPORT_FILE"
echo "        Generated on $(date)                           " >> "$REPORT_FILE"
echo "=======================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Models Directory: $MODELS_DIR" >> "$REPORT_FILE"
echo "Dataset: $DATA_FOLDER" >> "$REPORT_FILE"
echo "Sequence Length: $SEQ_LENGTH" >> "$REPORT_FILE"
echo "Dataset Format: New format (10-second windows with 2-second intervals)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "=======================================================" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Initialize CSV file with headers
echo "Model,Model Type,MSE,MAE,R2,Inference Latency (ms)" > "$RESULTS_CSV"

# Based on the error messages, we need to use "--use_stats" for all models
# since the scaler was trained with those statistical features
USE_STATS="--use_stats"
echo "Using statistical features (--use_stats) for all models based on scaler requirements" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Function to extract metrics from test output
extract_metrics() {
    local log_file=$1
    local model_name=$2
    
    # Check if there was an error
    if grep -q "Error:" "$log_file" || grep -q "Traceback" "$log_file"; then
        local error_msg=$(grep -A 5 "Error:" "$log_file" | tr '\n' ' ' | sed 's/  */ /g')
        if [ -z "$error_msg" ]; then
            error_msg=$(grep -A 5 "Traceback" "$log_file" | tr '\n' ' ' | sed 's/  */ /g')
        fi
        
        # Add to report
        echo "Model: $model_name" >> "$REPORT_FILE"
        echo "Status: FAILED" >> "$REPORT_FILE"
        echo "Error: $error_msg" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "-------------------------------------------------------" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        # Add to CSV with null values
        echo "$model_name,ERROR,null,null,null,null" >> "$RESULTS_CSV"
        return
    fi
    
    local model_type=$(grep "Detected model type:" "$log_file" | cut -d ':' -f 2 | tr -d ' ')
    
    # Extract metrics
    local mse=$(grep "Mean Squared Error (MSE):" "$log_file" | head -1 | awk '{print $5}')
    local mae=$(grep "Mean Absolute Error (MAE):" "$log_file" | head -1 | awk '{print $5}')
    local r2=$(grep "R\^2 Score:" "$log_file" | awk '{print $3}')
    local latency=$(grep "Average Inference Latency on Current System:" "$log_file" | awk '{print $7}')
    
    # Check if values were found
    if [ -z "$mse" ] || [ -z "$mae" ] || [ -z "$r2" ] || [ -z "$latency" ]; then
        # Add to report
        echo "Model: $model_name" >> "$REPORT_FILE"
        echo "Type: $model_type" >> "$REPORT_FILE"
        echo "Status: INCOMPLETE METRICS" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "-------------------------------------------------------" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        # Add to CSV with null values
        echo "$model_name,$model_type,null,null,null,null" >> "$RESULTS_CSV"
        return
    fi
    
    # Add to CSV
    echo "$model_name,$model_type,$mse,$mae,$r2,$latency" >> "$RESULTS_CSV"
    
    # Add to report
    echo "Model: $model_name" >> "$REPORT_FILE"
    echo "Type: $model_type" >> "$REPORT_FILE"
    echo "MSE: $mse" >> "$REPORT_FILE"
    echo "MAE: $mae" >> "$REPORT_FILE"
    echo "R² Score: $r2" >> "$REPORT_FILE"
    echo "Inference Latency: $latency ms" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "-------------------------------------------------------" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
}

# Count models for progress tracking
model_count=$(find "$MODELS_DIR" -name "*.h5" | wc -l)
current_model=0

echo "Found $model_count model(s) to evaluate"
echo ""

# Test each model
for model_file in "$MODELS_DIR"/*.h5; do
    # Skip if not a file (e.g., if no .h5 files found)
    if [ ! -f "$model_file" ]; then
        echo "No .h5 files found in $MODELS_DIR"
        continue
    fi
    
    model_name=$(basename "$model_file")
    current_model=$((current_model + 1))
    
    echo "[$current_model/$model_count] Testing model: $model_name"
    
    # Create a temporary file for this model's output
    temp_log=$(mktemp)
    
    # Run the test with --use_stats since our scaler requires it
    # Use the augmented flag for the new format (now default)
    python3 test_models.py \
        --data_folder "$DATA_FOLDER" \
        --model_file "$model_file" \
        --seq_length "$SEQ_LENGTH" \
        --scaler_file "$SCALER_FILE" \
        --augmented \
        $USE_STATS \
        --verbose 2>&1 | tee "$temp_log"
    
    # Extract metrics from the output
    extract_metrics "$temp_log" "$model_name"
    
    # Remove temporary file
    rm "$temp_log"
    
    echo "Completed testing of $model_name"
    echo ""
done

# Generate a summary section with rankings
echo "Generating summary report..."
python3 generate_model_report.py "$RESULTS_CSV" "$REPORT_FILE"

echo "Testing complete! Evaluation report saved to $REPORT_FILE"
echo "Detailed results saved to $RESULTS_CSV"