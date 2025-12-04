#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

run_command () {
    mkdir -p "results/$2/ppl"
    python eval_ppl.py --base_model $1 --tokenizer $4 --output_dir results/$2/ppl $3

#     mkdir -p "results/$2"
#     cd lm-evaluation-harness  
#     lm_eval --model hf \
#         --model_args pretrained=$1 \
#         --tasks lambada_openai \
#         --device cuda --output_csv ../results/$2/zeroshot_acc_lambada.csv
#     cd -  
}


# run_command "mamba-2.8b" "mamba-2.8b" "--use_bfloat" "gpt-neox-20b"
