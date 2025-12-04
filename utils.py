from datasets import load_dataset
import random
import torch
from copy import deepcopy
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

def get_examples(
    dataset,
    tokenizer,
    n_samples,
    seq_len=128,
    field_name="text",
    add_bos_to_every=False,
    return_raw_dataset=False,
):
    if dataset == "c4":
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
    elif dataset == "bookcorpus":
        dataset = load_dataset("bookcorpus", "plain_text",trust_remote_code=True)

        traindata = dataset["train"]
    else:
        raise NotImplementedError

    if return_raw_dataset:
        return traindata

    tokenized_samples, history = [], []

    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(
                traindata[i][field_name],
                return_tensors="pt",
                add_special_tokens=not add_bos_to_every,
            )
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        j = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tmp_ids = tokenized_sample.input_ids[:, j : j + seq_len]
        if add_bos_to_every:  # add bos token to every segment (especially for gemma)
            tmp_ids = torch.cat(
                (torch.LongTensor([[tokenizer.bos_token_id]]), tmp_ids[:, :-1]), dim=1
            )
        tokenized_samples.append(tmp_ids)

    return torch.cat(tokenized_samples, dim=0)



def get_hidden_states(model, 
                    input_tokens, 
                    batch_size=1, 
                    target_device="cuda",
                    output_device="cpu"):

    hidden_states = []
    model = model.to(target_device)
    
    with torch.no_grad():
        for i in range(0, input_tokens.size(0), batch_size):
            batch = input_tokens[i : i + batch_size].to(target_device)
            
            outputs = model(batch, 
                            labels=batch, 
                            output_hidden_states=True)
            
            last_hidden = outputs.hidden_states[-1].to(output_device)
            hidden_states.append(last_hidden)
                
    return hidden_states


def merge_layers_return_model(model, low_lay, high_lay, weight_factor):
    if low_lay < 0 or high_lay >= len(model.model.layers):
        raise ValueError("层的索引超出了模型的范围")

    model.to('cpu')
    model_copy = deepcopy(model)

    for current_layer_idx in range(low_lay, high_lay + 1): 
#         print('正在处理的层索引：', current_layer_idx)

        for projection in ['gate_proj', 'down_proj', 'up_proj']:
            model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].mlp.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data) * weight_factor
            )

        for projection in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].self_attn.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data) * weight_factor
            )            

    for current_layer_idx in range(high_lay, low_lay, -1):
        del(model_copy.model.layers[current_layer_idx])

    for layer_idx, module in enumerate(model_copy.model.layers):
        module.self_attn.layer_idx = layer_idx

    return model_copy


def cal_sim(args, hidden_states, model2, example_prompts):
    model2.to(args.device)
    sim_ls = []   
    count = 0
    for i in range(0, example_prompts.size(0), args.batch_size):
        example_prompts_tmp = example_prompts[i : i + args.batch_size].to(args.device)
        hidden_states1 = hidden_states[count]
        with torch.no_grad():
            outputs2 = model2(example_prompts_tmp, labels=example_prompts_tmp, output_hidden_states=True)       
            hidden_states2 = outputs2.hidden_states[-1]
            hidden_states2 = hidden_states2.to("cpu")
            
        sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
        count += 1
        
    sim_ls = [i.item() for i in sim_ls]
    print('sim_ls:', np.mean(sim_ls))

    return np.mean(sim_ls)  

def save_merged_model(model_copy, save_path):
    print(f"保存合并后的模型到 {save_path}")
    model_copy.save_pretrained(save_path)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def add_bias_to_linear_layers(model):
    """为所有无偏置的 Linear 层添加可学习的偏置项"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
    return model


def merge_layers_with_bias(model, low_lay, high_lay, norm_factor=0.5):
    if low_lay < 0 or high_lay >= len(model.model.layers):
        raise ValueError("层的索引超出了模型的范围")

    model.to('cpu')
    model_copy = deepcopy(model)

    for current_layer_idx in range(low_lay, high_lay + 1): 
#         print('正在处理的层索引：', current_layer_idx)

        for projection in ['gate_proj', 'down_proj', 'up_proj']:
            model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].mlp.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].mlp.__getattr__(projection).weight.data)
            )

        for projection in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data.add_(
                (model.model.layers[current_layer_idx].self_attn.__getattr__(projection).weight.data - model_copy.model.layers[low_lay].self_attn.__getattr__(projection).weight.data)
            )            

        input_norm = model_copy.model.layers[low_lay].input_layernorm
        current_input_norm = model.model.layers[current_layer_idx].input_layernorm
        input_norm.weight.data.add_(
            (current_input_norm.weight.data - input_norm.weight.data) * norm_factor
        )

        post_attn_norm = model_copy.model.layers[low_lay].post_attention_layernorm
        current_post_attn_norm = model.model.layers[current_layer_idx].post_attention_layernorm
        post_attn_norm.weight.data.add_(
            (current_post_attn_norm.weight.data - post_attn_norm.weight.data) * norm_factor
        )

    for current_layer_idx in range(high_lay, low_lay, -1):
        del(model_copy.model.layers[current_layer_idx])

    for layer_idx, module in enumerate(model_copy.model.layers):
        module.self_attn.layer_idx = layer_idx

    return model_copy


# def calculate_compensation(original_model, merged_model, example_input, device, low_lay, high_lay, batch_size=1):
#     """
#     计算原始模型与合并模型间的偏置补偿值（分批次处理）
    
#     参数:
#         original_model: 原始模型
#         merged_model: 合并后的模型
#         example_input: 输入样本 [num_samples, ...]
#         device: 计算设备 (如 'cuda')
#         low_lay: 起始层索引
#         high_lay: 终止层索引
#         batch_size: 每批样本数
        
#     返回:
#         mlp_compensations: 各层MLP的补偿值 {layer_idx: tensor}
#         attn_compensations: 各层Attention的补偿值 {layer_idx: tensor}
#     """
#     num_samples = example_input.size(0)
#     num_batches = (num_samples + batch_size - 1) // batch_size
    
#     # 初始化补偿值累积字典
#     mlp_compensations = {layer_idx: 0 for layer_idx in range(low_lay, high_lay + 1)}
#     attn_compensations = {layer_idx: 0 for layer_idx in range(low_lay, high_lay + 1)}
    
#     for batch_idx in tqdm(range(num_batches)):
#         start = batch_idx * batch_size
#         end = min(start + batch_size, num_samples)
#         batch_input = example_input[start:end].to(device)
        
#         # 获取原始模型输出
#         with torch.no_grad():
#             original_outputs = original_model(batch_input, output_hidden_states=True)
#             original_hidden = original_outputs.hidden_states
        
#         # 获取合并模型输出
#         with torch.no_grad():
#             merged_outputs = merged_model(batch_input, output_hidden_states=True)
#             merged_hidden = merged_outputs.hidden_states
        
#         # 逐层累积补偿值
#         for layer_idx in range(low_lay, high_lay + 1):
#             # MLP补偿
#             mlp_diff = original_hidden[layer_idx].mean(0) - merged_hidden[layer_idx].mean(0)
#             mlp_compensations[layer_idx] += mlp_diff.to(device)
            
#             # Attention补偿
#             attn_diff = original_hidden[layer_idx].mean(0) - merged_hidden[layer_idx].mean(0)
#             attn_compensations[layer_idx] += attn_diff.to(device)
    
#     # 计算全局平均补偿值
#     for layer_idx in mlp_compensations:
#         mlp_compensations[layer_idx] /= num_batches
#         attn_compensations[layer_idx] /= num_batches
    
#     return mlp_compensations, attn_compensations


def get_original_hidden_states(model, 
                    input_tokens, 
                    batch_size=1, 
                    target_device="cuda",
                    output_device="cpu"):

    last_hidden_states, all_hidden_states = [], []
    model = model.to(target_device)
    
    with torch.no_grad():
        for i in range(0, input_tokens.size(0), batch_size):
            batch = input_tokens[i : i + batch_size].to(target_device)
            
            outputs = model(batch, 
                            labels=batch, 
                            output_hidden_states=True)
            
            last_hidden = outputs.hidden_states[-1].to(output_device)
            last_hidden_states.append(last_hidden)
            all_hidden_states.append(outputs.hidden_states)

    return last_hidden_states, all_hidden_states


def calculate_compensation(original_model, merged_model, example_input, device, low_lay, high_lay, batch_size=4, all_hidden_states=None):
    num_samples = example_input.size(0)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # 初始化补偿值累积字典
    mlp_compensations = {
        layer_idx: {'gate_proj': 0, 'up_proj': 0, 'down_proj': 0}
        for layer_idx in range(low_lay, high_lay + 1)
    }
    attn_compensations = {
        layer_idx: {'q_proj': 0, 'k_proj': 0, 'v_proj': 0, 'o_proj': 0}
        for layer_idx in range(low_lay, high_lay + 1)
    }
    
    merged_model.to(device)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        batch_input = example_input[start:end].to(device)

        # 获取原始模型和合并模型的隐藏状态
        with torch.no_grad():
            original_hidden = all_hidden_states[batch_idx]
            merged_hidden = merged_model(batch_input, output_hidden_states=True).hidden_states
        
        for layer_idx in range(low_lay, high_lay + 1):
            # 确保input_diff是一维张量 [4096]
            input_diff = (original_hidden[layer_idx] - merged_hidden[layer_idx]).mean(dim=[0, 1])
            input_diff = input_diff.squeeze()  # 移除所有长度为1的维度
            
            # 处理MLP各投影层
            mlp_layer = merged_model.model.layers[layer_idx].mlp
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                proj_layer = getattr(mlp_layer, proj)
                if hasattr(proj_layer, 'bias'):
                    with torch.no_grad():
                        weight = proj_layer.weight
                        if proj in ['gate_proj', 'up_proj']:
                            # gate_proj/up_proj: [11008, 4096]
                            comp = torch.matmul(weight, input_diff)  # [11008]
                        else:
                            # down_proj: [4096, 11008]
                            # 我们需要计算的是对输出的补偿，所以应该使用权重矩阵的转置
                            comp = torch.matmul(weight.T, input_diff)  # [11008, 4096] @ [4096] = [11008]
                            # 但是down_proj的偏置是[4096]，所以需要调整计算方式
                            # 正确的做法是计算输入差异对输出的影响
                            comp = torch.matmul(input_diff, weight)  # [4096] @ [4096, 11008] = [11008]
                            # 由于偏置是[4096]，我们需要将11008维的补偿值降维到4096
                            # 这里我们取平均值作为近似
                            comp = comp.mean().expand(4096)  # 将11008维的补偿值平均并扩展到4096维
                    mlp_compensations[layer_idx][proj] += comp.to(device)
            
            # 处理Attention各投影层
            attn_layer = merged_model.model.layers[layer_idx].self_attn
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj_layer = getattr(attn_layer, proj)
                if hasattr(proj_layer, 'bias'):
                    with torch.no_grad():
                        weight = proj_layer.weight  # [4096, 4096]
                        comp = torch.matmul(weight, input_diff)  # [4096]
                    attn_compensations[layer_idx][proj] += comp.to(device)
    
    # 计算全局平均补偿值
    for layer_idx in mlp_compensations:
        for proj in mlp_compensations[layer_idx]:
            mlp_compensations[layer_idx][proj] /= num_batches
    for layer_idx in attn_compensations:
        for proj in attn_compensations[layer_idx]:
            attn_compensations[layer_idx][proj] /= num_batches
    
    return mlp_compensations, attn_compensations



def apply_compensation(merged_model, mlp_compensations, attn_compensations, device):
    """
    应用补偿值到合并模型的偏置项
    
    参数:
        merged_model: 合并后的模型
        mlp_compensations: MLP层补偿值字典 {layer_idx: {'gate_proj': tensor, 'up_proj': tensor, 'down_proj': tensor}}
        attn_compensations: Attention层补偿值字典 {layer_idx: {'q_proj': tensor, 'k_proj': tensor, 'v_proj': tensor, 'o_proj': tensor}}
        device: 目标设备
    """
    target_dtype = next(merged_model.parameters()).dtype
    
    for layer_idx in mlp_compensations:
        # 应用MLP补偿
        for proj, comp in mlp_compensations[layer_idx].items():
            proj_layer = merged_model.model.layers[layer_idx].mlp.__getattr__(proj)
            if hasattr(proj_layer, 'bias'):
                # 确保补偿值的数据类型与模型参数一致
                comp = comp.to(target_dtype).to(proj_layer.bias.device)
                proj_layer.bias.data += comp
        
        # 应用Attention补偿
        for proj, comp in attn_compensations[layer_idx].items():
            proj_layer = merged_model.model.layers[layer_idx].self_attn.__getattr__(proj)
            if hasattr(proj_layer, 'bias'):
                comp = comp.to(target_dtype).to(proj_layer.bias.device)
                proj_layer.bias.data += comp


def merge_layers_with_bias_compensation(
    model, 
    low_lay, 
    high_lay, 
    example_prompts=None,
    all_hidden_states=None,
    norm_factor=0.5, 
    device="cpu",
):
    """
    合并模型层，并计算偏置补偿值
    
    Args:
        model: 原始模型
        low_lay: 起始层索引
        high_lay: 终止层索引
        norm_factor: RMSNorm 调整系数
        device: 计算设备
        example_input: 用于计算补偿的示例输入（可选）
    """
    if low_lay < 0 or high_lay >= len(model.model.layers):
        raise ValueError("层的索引超出了模型的范围")

    model.to('cpu')
    model_copy = deepcopy(model)


    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Linear) and module.bias is None:
            model_dtype = next(model_copy.parameters()).dtype
            
            module.bias = nn.Parameter(
                torch.zeros(module.out_features, dtype=model_dtype, device=device)
            )
            module.weight = module.weight

    for current_layer_idx in range(low_lay, high_lay + 1):
        # 合并 MLP 权重
        for projection in ['gate_proj', 'up_proj', 'down_proj']:
            proj = model_copy.model.layers[low_lay].mlp.__getattr__(projection)
            current_proj = model.model.layers[current_layer_idx].mlp.__getattr__(projection)
            proj.weight.data.add_(
                current_proj.weight.data - proj.weight.data
            )

        # 合并 Self-Attention 权重
        for projection in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj = model_copy.model.layers[low_lay].self_attn.__getattr__(projection)
            current_proj = model.model.layers[current_layer_idx].self_attn.__getattr__(projection)
            proj.weight.data.add_(
                current_proj.weight.data - proj.weight.data
            )

    if example_prompts is not None:
        mlp_comps, attn_comps = calculate_compensation(
            original_model=model,
            merged_model=model_copy,
            example_input=example_prompts,
            device=device,
            low_lay=low_lay,
            high_lay=high_lay,
            batch_size=1,
            all_hidden_states=all_hidden_states,
        )
        apply_compensation(model_copy, mlp_comps, attn_comps, device)


    # 删除多余的层
    for current_layer_idx in range(high_lay, low_lay, -1):
        del model_copy.model.layers[current_layer_idx]

    # 更新剩余层的索引
    for layer_idx, module in enumerate(model_copy.model.layers):
        module.self_attn.layer_idx = layer_idx

    return model_copy

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class DirectCompensationLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 固定的补偿偏置和缩放因子
        self.bias = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.scale = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
    
    def forward(self, x):
        return x * self.scale + self.bias

def add_direct_compensation(model2, hidden_states, example_prompts, device='cuda'):
    """添加直接计算的补偿层"""
    # Move model to device
    model2 = model2.to(device)
    dtype = next(model2.parameters()).dtype  # Get model's dtype
    
    # Ensure all hidden states are on the correct device and dtype
    hidden_states = [h.to(device).to(dtype) for h in hidden_states]
    
    # 一次性收集model2的原始输出
    with torch.no_grad():
        outputs = model2(example_prompts.to(device), output_hidden_states=True)
        original_outputs = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
    
    # 计算平均偏差和缩放因子
    original_mean = original_outputs.mean(dim=(0, 1))  # [hidden_size]
    target_mean = torch.stack([h.mean(dim=(0, 1)) for h in hidden_states]).mean(dim=0)
    
    original_std = original_outputs.std(dim=(0, 1))
    target_std = torch.stack([h.std(dim=(0, 1)) for h in hidden_states]).mean(dim=0)
    
    # 避免除以零
    eps = 1e-6
    original_std = torch.where(original_std < eps, torch.ones_like(original_std)*eps, original_std)
    
    # 计算补偿参数
    scale = target_std / original_std  # Now both are on same device
    bias = target_mean - original_mean * scale
    
    # 创建补偿层并设置参数
    compensation_layer = DirectCompensationLayer(hidden_states[0].size(-1)).to(device).to(dtype)
    with torch.no_grad():
        compensation_layer.scale.copy_(scale)
        compensation_layer.bias.copy_(bias)
    
    # 保存原始forward方法
    original_forward = model2.forward
    
    # 定义补偿后的forward方法
    def compensated_forward(*args, **kwargs):
        outputs = original_forward(*args, **kwargs)
        last_hidden = outputs.hidden_states[-1]
        compensated_hidden = compensation_layer(last_hidden)
        
        # Create new outputs with compensated hidden states
        new_hidden_states = list(outputs.hidden_states[:-1]) + [compensated_hidden]
        
        # Return with correct parameter names
        return outputs.__class__(
            logits=outputs.logits,  # Keep original logits
            past_key_values=outputs.past_key_values,
            hidden_states=new_hidden_states,
            attentions=outputs.attentions
        )
    
    # 替换forward方法
    model2.forward = compensated_forward
    model2.compensation_layer = compensation_layer
    
    return model2, outputs.hidden_states[-1]  # 返回原始输出用于相似度计算

def cal_sim_with_compensation(args, hidden_states, model2, example_prompts):
    """先逐个计算原始相似度，再计算全局补偿后相似度"""
    model2.eval()
    model2.to(args.device)
    
    # 确保所有hidden_states在相同设备上
    hidden_states = [h.to(args.device) for h in hidden_states]
    
    original_outputs = []
    original_sims = []
    
    # 第一阶段：逐个处理并收集原始输出和相似度
    for i in range(len(example_prompts)):
        # 处理单个样本
        with torch.no_grad():
            prompt = example_prompts[i:i+1].to(args.device)
            outputs = model2(prompt, output_hidden_states=True)
            original_output = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
            original_outputs.append(original_output)
            
            # 计算原始相似度
            sim = F.cosine_similarity(
                hidden_states[i].flatten().unsqueeze(0),
                original_output.flatten().unsqueeze(0)
            )
            original_sims.append(sim.item())
    
    # 计算原始平均相似度
    avg_original_sim = sum(original_sims) / len(original_sims)
    print(f'原始平均相似度: {avg_original_sim:.4f}')
    
    # 第二阶段：计算全局补偿参数
    original_stack = torch.cat(original_outputs, dim=0)  # [N, seq_len, hidden_size]
    target_stack = torch.stack([h.squeeze(0) for h in hidden_states])  # [N, seq_len, hidden_size]
    
    original_mean = original_stack.mean(dim=(0, 1))
    target_mean = target_stack.mean(dim=(0, 1))
    original_std = original_stack.std(dim=(0, 1))
    target_std = target_stack.std(dim=(0, 1))
    
    eps = 1e-6
    original_std = torch.where(original_std < eps, torch.ones_like(original_std)*eps, original_std)
    
    scale = target_std / original_std
    bias = target_mean - original_mean * scale
    
    # 创建补偿层
    compensation_layer = DirectCompensationLayer(hidden_states[0].size(-1)).to(args.device)
    with torch.no_grad():
        compensation_layer.scale.copy_(scale)
        compensation_layer.bias.copy_(bias)
    
    # 第三阶段：应用补偿并重新计算相似度
    compensated_sims = []
    for i in range(len(example_prompts)):
        with torch.no_grad():
            # 应用补偿
            compensated_output = compensation_layer(original_outputs[i])
            
            # 计算补偿后相似度
            sim = F.cosine_similarity(
                hidden_states[i].flatten().unsqueeze(0),
                compensated_output.flatten().unsqueeze(0)
            )
            compensated_sims.append(sim.item())
    
    avg_compensated_sim = sum(compensated_sims) / len(compensated_sims)
    print(f'补偿后平均相似度: {avg_compensated_sim:.4f}')
    print(f'相似度提升: {avg_compensated_sim - avg_original_sim:.4f}')
    
    model2.to('cpu')
    
    # return {
    #     'original_sims': original_sims,  # 每个样本的原始相似度
    #     'compensated_sims': compensated_sims,  # 每个样本补偿后相似度
    #     'avg_original': avg_original_sim,
    #     'avg_compensated': avg_compensated_sim,
    #     'improvement': avg_compensated_sim - avg_original_sim
    # }

    return avg_compensated_sim
