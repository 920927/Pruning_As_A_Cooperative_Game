#!/bin/bash

BASE_MODEL_PATHS=(
    "Llama-2-7b-hf"
)

QUANTIZED_MODEL_DIRS=(
    "quantized_models/GPTQ/Llama-2-7b-hf"
)

NUM_EVAL=${#BASE_MODEL_PATHS[@]}

for ((i = 0; i < $NUM_EVAL; i++)); do
    BASE_MODEL_PATH=${BASE_MODEL_PATHS[$i]}
    QUANTIZED_MODEL_DIR=${QUANTIZED_MODEL_DIRS[$i]}

    echo "BASE_MODEL_PATH: $BASE_MODEL_PATH"
    echo "QUANTIZED_MODEL_DIR: $QUANTIZED_MODEL_DIR"

    if [ ! -d "$QUANTIZED_MODEL_DIR" ]; then
        mkdir -p "$QUANTIZED_MODEL_DIR"
    fi

    python quantize_gptq.py --base_model $BASE_MODEL_PATH --quantized_model_dir $QUANTIZED_MODEL_DIR

done
