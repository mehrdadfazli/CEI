#!/bin/bash

# Usage: ./run_amber.sh [config_file]
# Default config file is config.json if not provided

# Check for config file argument
CONFIG_FILE=${1:-config.json}

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required but not installed. Install with 'sudo apt-get install jq' or equivalent."
    exit 1
fi

# Extract fields from config file using jq
MODEL_TYPE=$(jq -r '.model_type' "$CONFIG_FILE")
LOAD_IN_8BIT=$(jq -r '.load_in_8bit' "$CONFIG_FILE")
CACHE_DIR=$(jq -r '.cache_dir' "$CONFIG_FILE")
AMBER_PATH=$(jq -r '.amber_path' "$CONFIG_FILE")
LOG_DIR=$(jq -r '.amber_log_dir' "$CONFIG_FILE")
DO_SAMPLE=$(jq -r '.do_sample' "$CONFIG_FILE")
USE_CEI=$(jq -r '.use_CEI' "$CONFIG_FILE")
NUM_BEAMS=$(jq -r '.num_beams' "$CONFIG_FILE")
MAX_NEW_TOKENS=$(jq -r '.max_new_tokens' "$CONFIG_FILE")
CONTEXT_STRATEGY=$(jq -r '.context_strategy' "$CONFIG_FILE")
TOPK=$(jq -r '.topk' "$CONFIG_FILE")
INJECTION_LAYER=$(jq -r '.injection_layer' "$CONFIG_FILE")
ALPHA=$(jq -r '.alpha' "$CONFIG_FILE")

# Validate required fields
if [ -z "$MODEL_TYPE" ] || [ -z "$AMBER_PATH" ] || [ -z "$LOG_DIR" ] || [ -z "$NUM_BEAMS" ] || \
   [ -z "$MAX_NEW_TOKENS" ] || [ -z "$CONTEXT_STRATEGY" ] || [ -z "$TOPK" ] || \
   [ -z "$INJECTION_LAYER" ] || [ -z "$ALPHA" ]; then
    echo "Error: One or more required fields are missing or null in '$CONFIG_FILE'."
    exit 1
fi

# Build command-line arguments
ARGS=""
ARGS+=" --model_type $MODEL_TYPE"
[ "$LOAD_IN_8BIT" = "true" ] && ARGS+=" --load_in_8bit"
[ -n "$CACHE_DIR" ] && ARGS+=" --cache_dir $CACHE_DIR"
ARGS+=" --amber_path $AMBER_PATH"
ARGS+=" --log_dir $LOG_DIR"
[ "$DO_SAMPLE" = "true" ] && ARGS+=" --do_sample"
[ "$USE_CEI" = "true" ] && ARGS+=" --use_CEI"
ARGS+=" --num_beams $NUM_BEAMS"
ARGS+=" --max_new_tokens $MAX_NEW_TOKENS"
ARGS+=" --context_strategy $CONTEXT_STRATEGY"
ARGS+=" --topk $TOPK"
ARGS+=" --injection_layer $INJECTION_LAYER"
ARGS+=" --alpha $ALPHA"

# Run the Python script
echo "Running run_AMBER_dyn.py with arguments:$ARGS"
python3 run_AMBER_dyn.py $ARGS

if [ $? -ne 0 ]; then
    echo "Error: Python script failed."
    exit 1
fi

echo "AMBER benchmark completed successfully."