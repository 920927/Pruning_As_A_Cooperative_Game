import pandas as pd
import numpy as np
import argparse
import random
from tqdm.notebook import tqdm
from copy import deepcopy

import torch
import torch.nn as nn

from transformers import LlamaForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------------------------------------------------------------------------------

def delete_layers(model, layer_indices):
    """
    根据给定的层索引列表批量删除模型层
    :param model: 原始模型
    :param layer_indices: 要删除的层索引列表（0-based）
    """
    model.to('cpu')
    model_copy = deepcopy(model)

    max_idx = len(model.model.layers) - 1
    for idx in layer_indices:
        if idx < 0 or idx > max_idx:
            raise ValueError(f"层索引 {idx} 无效，允许范围 0~{max_idx}")

    for idx in sorted(layer_indices, reverse=True):
        del model_copy.model.layers[idx]

    for new_idx, layer in enumerate(model_copy.model.layers):
        if hasattr(layer.self_attn, 'layer_idx'):
            layer.self_attn.layer_idx = new_idx

    model_copy.config.num_hidden_layers = len(model_copy.model.layers)

    return model_copy


def slm_prune(args):
    model_name = 'Llama-2-7b-hf'


    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    layers_to_delete = [22, 21, 12, 24, 11, 23, 14, 25, 10, 18, 17, 8]
    
    pruned_model = delete_layers(model, layers_to_delete)
    pruned_model.save_pretrained(f"pruned_models/llama2_{'_'.join(map(str, layers_to_delete))}")
    tokenizer.save_pretrained(f"pruned_models/llama2_{'_'.join(map(str, layers_to_delete))}")

# --------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--dataset", type=str, default="bookcorpus")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--high_lay", type=int)

    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.gpu is not None:
        args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    slm_prune(args)
