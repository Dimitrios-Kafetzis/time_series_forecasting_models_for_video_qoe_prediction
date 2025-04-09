#!/bin/bash
# test_all_models.sh
# Script to run testing on all models in the forecasting_models_v5 directory
# and generate a comprehensive evaluation report
# UPDATED: Now includes validation testing functionality

# Define paths
MODELS_DIR=~/Impact-xG_prediction_model/forecasting_models_v5
DATA_FOLDER="./real_dataset"  # Change this to your actual dataset path
SCALER_FILE="$MODELS_DIR/scaler.save"
SEQ_LENGTH=5
REPORT_FILE="$MODELS_DIR/model_evaluation_report.txt"
RESULTS_CSV="$MODELS_DIR/model_evaluation_results.csv"

# Validation settings (new)
VALIDATION_FOLDER="./validation_dataset"
VALIDATION_OUTPUT="$MODELS_DIR/validation_results"
VALIDATION_ENABLED=false
PREPARE_VALIDATION=false
VALIDATION_SAMPLE_RATIO=0.2
VALIDATION_SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --models-dir)
            MODELS_DIR="$2"
            shift
            shift
            ;;
        --data-folder)
            DATA_FOLDER="$2"
            shift
            shift
            ;;
        --scaler-file)
            SCALER_FILE="$2"
            shift
            shift
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift
            shift
            ;;
        --report-file)
            REPORT_FILE="$2"
            shift
            shift
            ;;
        --use-stats)
            USE_STATS="--use_stats"
            shift
            ;;
        --validate)
            VALIDATION_ENABLED=true
            shift
            ;;
        --validation-folder)
            VALIDATION_FOLDER="$2"
            shift
            shift
            ;;
        --validation-output)
            VALIDATION_OUTPUT="$2"
            shift
            shift
            ;;
        --prepare-validation)
            PREPARE_VALIDATION=true
            shift
            ;;
        --validation-sample)
            VALIDATION_SAMPLE_RATIO="$2"
            shift
            shift
            ;;
        --validation-seed)
            VALIDATION_SEED="$2"
            shift
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --models-dir DIR         Directory containing model files (default: $MODELS_DIR)"
            echo "  --data-folder DIR        Dataset path (default: $DATA_FOLDER)"
            echo "  --scaler-file FILE       Path to scaler file (default: $SCALER_FILE)"
            echo "  --seq-length NUM         Sequence length (default: $SEQ_LENGTH)"
            echo "  --report-file FILE       Path to output report (default: $REPORT_FILE)"
            echo "  --use-stats              Enable statistical features"
            echo "  --validate               Enable validation testing"
            echo "  --validation-folder DIR  Validation dataset folder (default: $VALIDATION_FOLDER)"
            echo "  --validation-output DIR  Validation results output folder (default: $VALIDATION_OUTPUT)"
            echo "  --prepare-validation     Prepare validation dataset before testing"
            echo "  --validation-sample NUM  Validation sample ratio (default: $VALIDATION_SAMPLE_RATIO)"
            echo "  --validation-seed NUM    Random seed for validation sampling (default: $VALIDATION_SEED)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Prepare validation dataset if requested
if [ "$PREPARE_VALIDATION" = true ]; then
    echo "Preparing validation dataset from $DATA_FOLDER..."
    # Make sure the validation folder exists
    mkdir -p "$VALIDATION_FOLDER"
    
    python3 prepare_validation_data.py \
        --input_folder "$DATA_FOLDER" \
        --output_folder "$VALIDATION_FOLDER" \
        --sample_ratio "$VALIDATION_SAMPLE_RATIO" \
        --random_seed "$VALIDATION_SEED"
        
    if [ $? -ne 0 ]; then
        echo "Error preparing validation dataset"
        exit 1
    fi
    
    echo "Validation dataset prepared in $VALIDATION_FOLDER"
fi

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

# Run validation if enabled
if [ "$VALIDATION_ENABLED" = true ]; then
    echo ""
    echo "========================================================="
    echo "Starting validation with controlled experiments..."
    echo "========================================================="
    
    # Create validation output directory
    mkdir -p "$VALIDATION_OUTPUT"
    
    # Run validation across all models
    python3 validate_models.py \
        --validation_folder "$VALIDATION_FOLDER" \
        --model_dir "$MODELS_DIR" \
        --scaler_file "$SCALER_FILE" \
        --output_dir "$VALIDATION_OUTPUT" \
        --seq_length "$SEQ_LENGTH" \
        $USE_STATS
    
    if [ $? -ne 0 ]; then
        echo "Error during validation process"
        exit 1
    fi
    
    echo ""
    echo "Validation results saved to $VALIDATION_OUTPUT"
    
    # Add validation results path to the main report
    echo "" >> "$REPORT_FILE"
    echo "=======================================================" >> "$REPORT_FILE"
    echo "VALIDATION RESULTS" >> "$REPORT_FILE"
    echo "=======================================================" >> "$REPORT_FILE"
    echo "Detailed validation results are available at:" >> "$REPORT_FILE"
    echo "$VALIDATION_OUTPUT" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    # Try to extract top model from validation results
    if [ -f "$VALIDATION_OUTPUT/validation_summary.csv" ]; then
        # Extract best model by R2 score (could be adapted to use different metrics)
        best_model=$(tail -n +2 "$VALIDATION_OUTPUT/validation_summary.csv" | sort -t, -k7 -nr | head -1 | cut -d, -f1)
        if [ ! -z "$best_model" ]; then
            echo "Best performing model based on validation R2 score: $best_model" >> "$REPORT_FILE"
        fi
    fi
fi

echo ""
echo "All evaluation processes complete!"