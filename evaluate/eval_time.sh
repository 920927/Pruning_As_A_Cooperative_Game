#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

batch_size=1
max_seq_len=128
run_command () {    
    python gen_batch_eval_time.py --base_model $1 \
        --output_dir results_efficiency/$2/batch_gen_out${max_seq_len}_bs${batch_size} \
        --batch_size $batch_size --max_seq_len $max_seq_len $3 
}

# run_command "Llama-2-7b-hf" "Llama-2-7b-hf" "--use_bfloat"



