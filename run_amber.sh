#!/bin/bash

# Get the configuration file and exp_dir from arguments
CONFIG_FILE=$1
EXP_DIR=$2

# Function to read a single value from JSON
read_config() {
    python -c "import json; config=json.load(open('$CONFIG_FILE')); print(config.get('$1', ''))"
}

# Function to read an array from JSON
read_array() {
    python -c "import json; config=json.load(open('$CONFIG_FILE')); print(' '.join(map(str, config.get('$1', []))))"
}

# Read variables
model_type=$(read_config model_type)
load_in_8bit=$(read_config load_in_8bit)
cache_dir=$(read_config cache_dir)
amber_path=$(read_config amber_path)
do_sample=$(read_config do_sample)
use_CEI=$(read_config use_CEI)
num_beams=$(read_config num_beams)
# log_dir=$(read_config log_dir)
max_new_tokens=$(read_config max_new_tokens)
context_embedding_idx=$(read_config context_embedding_idx)
context_embedding_layer=$(read_config context_embedding_layer)
injection_layer=$(read_config injection_layer)
alpha=$(read_config alpha)

# Check if model_type is read correctly
if [ -z "$model_type" ]; then
    echo "Error: Failed to read 'model_type' from $CONFIG_FILE"
    exit 1
fi

# Set log_dir to exp_dir
log_dir=$EXP_DIR


# Build the command
cmd="python ./run_AMBER.py \
    --model_type $model_type \
    --cache_dir $cache_dir \
    --amber_path $amber_path \
    --log_dir $log_dir \
    --num_beams $num_beams \
    --max_new_tokens $max_new_tokens \
    --context_embedding_idx $context_embedding_idx \
    --context_embedding_layer $context_embedding_layer \
    --injection_layer $injection_layer \
    --alpha $alpha"

# Add boolean flags conditionally
if [ "$load_in_8bit" = "true" ]; then
    cmd="$cmd --load_in_8bit"
fi
if [ "$do_sample" = "true" ]; then
    cmd="$cmd --do_sample"
fi
if [ "$use_CEI" = "true" ] || [ "$use_CEI" = "True" ]; then
    cmd="$cmd --use_CEI"
fi

echo "use_CEI value: $use_CEI"

echo "Running command: $cmd"

# Execute the command
eval "$cmd"