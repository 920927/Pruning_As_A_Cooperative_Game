import argparse
import csv
import os
import statistics

import torch
from utils import count_params, set_seed, count_quantized_params

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
import gc


def set_model_device_evalmode(
    model, device, fix_decapoda_config=False, use_bfloat=False
):
    if "cuda" in device:
        model.half()
        model = model.to(device)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()

    if use_bfloat:
        model = model.bfloat16()

    gc.collect()
    torch.cuda.empty_cache()

    return model


def get_model(
    base_model=None,
    ckpt=None,
    lora_ckpt=None,
    tokenizer=None,
    model_type="pretrain",
    device="cuda",
    fix_decapoda_config=False,
    use_bfloat=False,
):
    tokenizer = base_model if tokenizer is None else tokenizer
    if model_type == "pretrain":
        config = AutoConfig.from_pretrained(base_model)
        if "gptq" in base_model.lower():
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                base_model,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
                quantize_config=None,
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif (
            "LlamaForCausalLM" in config.__getattribute__("architectures")
            and "llama-3" not in base_model.lower() and 'deepseek' not in base_model.lower()
        ):
            model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    elif model_type in ["pruneLLM", "tune_pruneLLM"]:
        pruned_dict = torch.load(ckpt, map_location="cpu")
        model = pruned_dict["model"]
        tokenizer = pruned_dict["tokenizer"]
        if model_type == "tune_pruneLLM":
            model = PeftModel.from_pretrained(
                model, lora_ckpt, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
    else:
        raise NotImplementedError
    description = "Model Type: {}\n Base: {} \n Pruned: {}\n LORA: {}".format(
        model_type, base_model, ckpt, lora_ckpt
    )

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        tokenizer.pad_token_id = 0
    model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)

    return model, tokenizer, description




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model",
        type=str,
        default="baffo32/decapoda-research-llama-7B-hf",
        help="base model name",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="if None, base model name is used"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="pretrain",
        choices=["pretrain", "pruneLLM", "tune_pruneLLM"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lora_ckpt", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument(
        "--input_prompt", type=str, default="The Leaning Tower of Pisa is known for"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_all_runs", type=int, default=30)
    parser.add_argument("--num_warmup_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_efficiency/llama-1-7b/batch_gen_out128_bs4",
    )
    parser.add_argument(
        "--fix_decapoda_config",
        default=False,
        action="store_true",
        help="fix tokenizer config of baffo32/decapoda-research-llama-7B-hf",
    )
    parser.add_argument("--use_bfloat", default=False, action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer, description = get_model(
        base_model=args.base_model,
        ckpt=args.ckpt,
        lora_ckpt=args.lora_ckpt,
        tokenizer=args.tokenizer,
        model_type=args.model_type,
        device=args.device,
        fix_decapoda_config=args.fix_decapoda_config,
        use_bfloat=args.use_bfloat,
    )

    # Prepare input for batched generation
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        [args.input_prompt] * args.batch_size, return_tensors="pt", padding=True
    )["input_ids"].to(args.device)
    input_len = inputs[0].size(0)

    # Set log path
    os.makedirs(args.output_dir, exist_ok=True)
    time_stat_csv = os.path.join(args.output_dir, "latency_throughput.csv")
    gen_log_path = os.path.join(args.output_dir, "gen_output.txt")
    open(gen_log_path, "w", encoding="utf8").close()  # To remove previous record

    time_list = []
    throughput_list = []

    for i in range(args.num_all_runs):
        if "cuda" in args.device:
            starter, ender = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            starter.record()
            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=inputs,
                    do_sample=True,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_length=(input_len + args.max_seq_len),
                    min_length=(
                        input_len + args.max_seq_len
                    ),  # forced output length (to avoid <EOS> sampling)
                    return_dict_in_generate=True,
                )
            ender.record()
            torch.cuda.synchronize()
            batch_time = starter.elapsed_time(ender) / 1000  # in msec -> sec
        else:
            raise NotImplementedError

        if i < args.num_warmup_runs:  # no record for warmup runs
            continue
        else:
            output = tokenizer.batch_decode(generation_output.sequences)
            for b_idx, s in enumerate(output):
                tmp_size = len(generation_output.sequences[b_idx]) - input_len
                with open(gen_log_path, "a", encoding="utf8") as f:
                    f.write(
                        f"=== output {i} b{b_idx} | leng gen {tmp_size} + input {input_len}\n"
                    )
                    f.write(f"{s}\n")
                    print(
                        f"=== output {i} b{b_idx} | leng gen {tmp_size} + input {input_len}\n"
                    )
                    print(f"{s}\n")
            time_list.append(batch_time)
            throughput_list.append(args.batch_size * args.max_seq_len / batch_time)

    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Current GPU memory occupied: {mem} MiB")
    
    if 'gptq' in args.base_model.lower():
        nparams = count_quantized_params(model)
    else:
        nparams = count_params(model)
    print(f"Params: {nparams}")

    time_mean = statistics.mean(time_list)
    time_std = statistics.pstdev(time_list)

    throughput_mean = statistics.mean(throughput_list)
    throughput_std = statistics.pstdev(throughput_list)
    nbatches = len(throughput_list)

    with open(time_stat_csv, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(
            [
                "time_mean(sec)",
                "time_std",
                "th_mean(tokens/s)",
                "th_std",
                "mem",
                "out_len",
                "in_len",
                "nbatches",
                "batchsz",
                "nparam",
            ]
        )
        logwriter.writerow(
            [
                time_mean,
                time_std,
                throughput_mean,
                throughput_std,
                mem,
                args.max_seq_len,
                input_len,
                nbatches,
                args.batch_size,
                nparams,
            ]
        )
        logwriter.writerow(time_list)
