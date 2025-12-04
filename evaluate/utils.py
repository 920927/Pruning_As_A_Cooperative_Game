import copy
import csv
import gc
import json
import random
import time

import numpy as np
import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)



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
    tokenizer=None,
    device="cuda",
    fix_decapoda_config=False,
    use_bfloat=False,
):
    tokenizer = base_model if tokenizer is None else tokenizer
    config = AutoConfig.from_pretrained(base_model)
    
    if 'rwkv' in base_model.lower():
        model = AutoModelForCausalLM.from_pretrained(
            base_model, trust_remote_code=True, torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    elif 'mamba' in base_model.lower():
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        model = MambaLMHeadModel.from_pretrained(base_model, device=device, dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    elif "gptq" in base_model.lower():
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
        and "llama-3" not in base_model.lower()
    ):
        model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        tokenizer.pad_token_id = 0
    model = set_model_device_evalmode(model, device, fix_decapoda_config, use_bfloat)

    return model, tokenizer

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_quantized_params(model):
    core_params, extra_params = 0, 0

    for name, param in model.named_parameters():
        if any(key in name for key in ["qweight", "scales", "zeros", "g_idx"]):
            extra_params += param.numel()
        else:
            core_params += param.numel()

    # buffer 也可能含有量化辅助信息
    for name, buf in model.named_buffers():
        if any(key in name for key in ["qweight", "scales", "zeros", "g_idx"]):
            extra_params += buf.numel()
        else:
            core_params += buf.numel()

    total_params = core_params + extra_params

    print(f"核心参数数量 (对应原始权重): {core_params:,}")
    print(f"量化额外参数数量 (scales/zeros 等): {extra_params:,}")
    print(f"总参数数量: {total_params:,}")

    return core_params, extra_params, total_params
