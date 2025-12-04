import pandas as pd
import numpy as np
import argparse
import random
from tqdm.notebook import tqdm
from copy import deepcopy

import torch
import torch.nn as nn

from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

# --------------------------------------------------------------------------------------------------------------------------------------------------

def delete_layers(model, layer_indices):
    max_idx = len(model.model.model.layers) - 1
    for idx in layer_indices:
        if idx < 0 or idx > max_idx:
            raise ValueError(f"层索引 {idx} 无效，允许范围 0~{max_idx}")

    # 逆序删除，避免索引错位
    for idx in sorted(layer_indices, reverse=True):
        del model.model.model.layers[idx]

    # 更新 config
    model.config.num_hidden_layers = len(model.model.model.layers)

    return model


def quantize_prune(args):
    model_name = args.model_path
    print(f"🔹 加载模型 {model_name} ...")

    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        use_safetensors=True,
        trust_remote_code=True,
        use_triton=False,
        quantize_config=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(model)

    layers_to_delete = [21, 11, 12, 25, 23, 10]

    print(f"🔹 原始层数: {len(model.model.model.layers)}")
    pruned_model = delete_layers(model, layers_to_delete)
    print(f"✅ 剪枝后层数: {len(pruned_model.model.model.layers)}")
    

    pruned_model.save_pretrained(f"pruned_models/gptq_{'_'.join(map(str, layers_to_delete))}")
    tokenizer.save_pretrained(f"pruned_models/gptq_{'_'.join(map(str, layers_to_delete))}")

    print(f"✅ 模型已保存到 {save_dir}, 剪掉了 {len(layers_to_delete)} 层")


# --------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mamba 层剪枝")
    parser.add_argument("--model_path", type=str, default="GPTQ/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="bookcorpus")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--high_lay", type=int)
    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # 固定随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置设备
    if args.gpu is not None:
        args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    quantize_prune(args)
