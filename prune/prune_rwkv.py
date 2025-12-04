import torch
import torch.nn as nn
import argparse
import random
import numpy as np
from tqdm.notebook import tqdm
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------------------------------------------------------------------------------

def delete_layers_rwkv(model, layer_indices):
    """
    根据给定的层索引列表批量删除 RWKV 模型的层
    :param model: 原始模型
    :param layer_indices: 要删除的层索引列表（0-based）
    """
    model.to('cpu')
    model_copy = deepcopy(model)

    layers = model_copy.rwkv.blocks
    max_idx = len(layers) - 1

    for idx in layer_indices:
        if idx < 0 or idx > max_idx:
            raise ValueError(f"层索引 {idx} 无效，允许范围 0~{max_idx}")

    for idx in sorted(layer_indices, reverse=True):
        del layers[idx]

    for new_idx, block in enumerate(layers):
        if hasattr(block.attention, 'layer_idx'):
            block.attention.layer_idx = new_idx

    model_copy.config.num_hidden_layers = len(layers)

    return model_copy

# --------------------------------------------------------------------------------------------------------------------------------------------------

def rwkv_prune(args):

    model_name = 'rwkv-4-world-7b'

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).to("cuda:0")
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    layers_to_delete = [12, 20, 13, 29, 23, 5, 17, 25, 15]
    
    pruned_model = delete_layers_rwkv(model, layers_to_delete)

    pruned_model.save_pretrained(f"pruned_models/rwkv_{'_'.join(map(str, layers_to_delete))}")
    tokenizer.save_pretrained(f"pruned_models/rwkv_{'_'.join(map(str, layers_to_delete))}")

# --------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_path", type=str, default='path_to_rwkv_model')
    parser.add_argument("--dataset", type=str, default="bookcorpus")
    parser.add_argument("--threshold", type=float, default=0.72)
    parser.add_argument("--high_lay", type=int)

    parser.add_argument("--target_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置设备
    if args.gpu is not None:
        args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 执行 RWKV 模型的剪枝
    rwkv_prune(args)
