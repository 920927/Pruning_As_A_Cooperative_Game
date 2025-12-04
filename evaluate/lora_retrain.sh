#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export BASE_MODEL=Llama-2-7b-hf
export MODEL_NAME=Llama-2-7b-hf
export OUTPUT_PRUNE=pruned_models/llama2_15_14_24_13_25_22_8_23_12_29_21_7
export OUTPUT_TUNE=tuned_models/llama2_15_14_24_13_25_22_8_23_12_29_21_7


python lora_retrain.py \
    --base_model $OUTPUT_PRUNE \
    --data_path yahma/alpaca-cleaned \
    --output_dir $OUTPUT_TUNE \
    --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --micro_batch_size 4 \
    --save_lora_merge --use_bfloat
