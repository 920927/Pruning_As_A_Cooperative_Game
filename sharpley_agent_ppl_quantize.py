import time
import csv
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from contextlib import contextmanager
from utils import get_examples
import matplotlib.pyplot as plt
import math

from auto_gptq import AutoGPTQForCausalLM

####################################################
# ######## Set random seed ##########
# ###################################################
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

####################################################
# ######## Basic function ##########
# ###################################################

def cal_ppl(model, input_ids, tokenizer=None):
    """
    计算给定输入的 perplexity
    适配 MambaLMHeadModel：手动计算 loss
    """
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # 兼容 tokenizer 是否有 pad_token_id
        if tokenizer is not None and tokenizer.pad_token_id is not None:
            ignore_index = tokenizer.pad_token_id
        else:
            ignore_index = -100

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        ppl = torch.exp(loss)
    return ppl.item()


def gen_ks(num_layers, max_remove, include_full=False):
    min_keep = num_layers - max_remove
    ks = list(range(min_keep, num_layers))
    if include_full:
        ks.append(num_layers)
    return tuple(ks)

def stratified_by_hamming_weight(num_layers, mc, per_k=None, rng=None):
#     ks = (30, 27, 24, 21, 18)
    ks = (27, 24, 21, 18)
    rng = np.random.default_rng(rng)
    if per_k is None:
        q, r = divmod(mc, len(ks))
        per_k = [q + (1 if i < r else 0) for i in range(len(ks))]
    masks = []
    for k, cnt in zip(ks, per_k):
        for _ in range(cnt):
            ones_idx = rng.choice(num_layers, size=k, replace=False)
            m = np.zeros(num_layers, dtype=np.int8)
            m[ones_idx] = 1
            masks.append(m)
    return np.stack(masks, axis=0)

####################################################
# ######## Efficient inference with masks ##########
# ###################################################

def run_model_with_layer_mask(model, input_ids, layer_mask, tokenizer):
    zero_layers = [i for i, bit in enumerate(layer_mask) if bit == 0]
    with zero_out_layer_params(model, zero_layers):
        ppl = cal_ppl(model, input_ids, tokenizer)
    return ppl

@contextmanager
def zero_out_layer_params(model, layer_indices):
    if not layer_indices:
        yield
        return
    saved = []
    try:
        for idx in layer_indices:
            layer = model.model.model.layers[idx]
            for name, p in layer.named_parameters(recurse=True):
                saved.append((p, p.data.clone()))
                p.data.zero_()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        yield
    finally:
        for p, data in saved:
            p.data.copy_(data)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

####################################################
# #############   Agent Net   ##############
# ###################################################

class NeuralNet(nn.Module):
    def __init__(self, in_size, bs, lr, seed):
        super(NeuralNet, self).__init__()
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(in_size, int(in_size * 2)),
            nn.CELU(),
            nn.Linear(int(in_size * 2), 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.loss_fn = nn.MSELoss()
        self.bs = bs

    def change_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def forward(self, x):
        return self.model(x)

    def train_step(self, pred, y):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, pred, y):
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred, y)
        return loss.item()

####################################################
# ######## Functions for each step ##########
# ###################################################

def learning_feat(target_model, example_prompts, masks_list, batch_size, baseline_ppl, tokenizer, device):
    fc_res = []
    for mask in tqdm(masks_list):
        ppl_total, count = 0.0, 0
        for j in range(0, example_prompts.size(0), batch_size):
            inp = example_prompts[j: j + batch_size].to(device)
            ppl_masked = run_model_with_layer_mask(target_model, inp, mask, tokenizer)
            ppl_total += ppl_masked
            count += 1
        avg_ppl = ppl_total / count
        score = baseline_ppl / avg_ppl
        if not (math.isnan(score) or math.isinf(score)):
            fc_res.append(score)
    return fc_res

def step1_save_data(model, example_prompts, tokenizer, simu_num=8000, batch_size=45, save_prefix="PIE_data"):
    num_feat = len(model.model.model.layers)
    print(f"num of model layers: {num_feat}")

    baseline_total, count = 0.0, 0
    for j in range(0, example_prompts.size(0), batch_size):
        inp = example_prompts[j: j + batch_size].to(device)
        baseline_total += cal_ppl(model, inp, tokenizer)
        count += 1
    baseline_ppl = baseline_total / count
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    masks_list = stratified_by_hamming_weight(num_feat, simu_num)
    probs = torch.tensor(
        learning_feat(model, example_prompts, masks_list, batch_size, baseline_ppl, tokenizer, device),
        dtype=torch.float32
    )
    masks_list_torch = torch.tensor(masks_list.tolist(), dtype=torch.float)
    torch.save({"masks": masks_list_torch, "probs": probs},
               f"{save_prefix}_train_data.pt")
    print(f"Step1完成: 数据已保存到 {save_prefix}_train_data.pt")

def step2_train_model(num_feat, lr=0.008, epochs=300, bs=300, save_prefix="PIE_data"):
    saved = torch.load(f"{save_prefix}_train_data.pt")
    masks, probs = saved["masks"], saved["probs"]

    data = [[x, y] for x, y in zip(masks, probs)]
    net = NeuralNet(num_feat, bs, lr, seed).cuda()
    net.change_lr(lr)
    scheduler = torch.optim.lr_scheduler.StepLR(net.optimizer, step_size=100, gamma=0.1)
    loader = DataLoader(dataset=data[num_feat:], batch_size=bs, shuffle=True)

    loss_history = []
    for epoch in range(epochs):
        total_loss, total_count = 0, 0
        for x, y in loader:
            pred = net(x.cuda())
            tmploss = net.train_step(pred.squeeze(1), y.cuda())
            total_loss += tmploss * x.shape[0]
            total_count += x.shape[0]
        avg_loss = total_loss / total_count
        loss_history.append(total_loss)
        print(f"epoch {epoch} loss: {avg_loss:.6f}")
        scheduler.step()
        
    torch.save(net.state_dict(), f"{save_prefix}_agent_model.pth")
    print(f"Step2完成: 模型已保存到 {save_prefix}_agent_model.pth")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), loss_history, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss_curve.png", dpi=300)
    plt.close()
    print(f"Loss 曲线已保存到 {save_prefix}_loss_curve.png")

def step3_estimate_contrib(num_feat, mc=80000, rank=None, save_prefix="PIE_data"):
    net = NeuralNet(num_feat, bs=300, lr=0.008, seed=seed).cuda()
    net.load_state_dict(torch.load(f"{save_prefix}_agent_model.pth"))
    net.eval()

    base_masks = stratified_by_hamming_weight(num_feat, mc)
    print("base_masks shape:", base_masks.shape)

    results = []
    for m in tqdm(range(num_feat)):
        sim_acc = 0.0
        for s in range(mc):
            base_mask = base_masks[s].copy()
            mask1 = base_mask.copy(); mask1[m] = 1
            mask0 = base_mask.copy(); mask0[m] = 0
            sim1 = net(torch.tensor(mask1, dtype=torch.float32).unsqueeze(0).cuda()).item()
            sim0 = net(torch.tensor(mask0, dtype=torch.float32).unsqueeze(0).cuda()).item()
            sim_acc += (sim1 - sim0)
        avg_sim = sim_acc / mc
        results.append(avg_sim)
        print(f'layer {m} average contribution: {avg_sim:.6f}')

    filename = f'{save_prefix}_layer_contrib.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "contribution"])
        for idx, value in enumerate(results):
            writer.writerow([idx, value])
    print(f"Step3完成: 结果已保存到 {filename}")

    if rank is not None and rank > 0:
        top_indices = sorted(range(len(results)), key=lambda i: results[i], reverse=False)[:rank]
        print(f"排名前 {rank} 的层索引: {top_indices}")

    return results

####################################################
# ######## Main Entry ##########
# ###################################################
if __name__ == "__main__":
    step = 3  # 修改为 1/2/3

    gpu = 0
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model_name = "quantized_models/GPTQ/llama2_21_23_11"
    dataset_name = 'bookcorpus'

    if step == 1:
        print("加载模型与tokenizer...")
        model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
            quantize_config=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        example_prompts = get_examples(dataset=dataset_name, tokenizer=tokenizer, n_samples=10, seq_len=128).to(device)
        step1_save_data(model, example_prompts, tokenizer)

    elif step == 2:
        num_feat = 29
        step2_train_model(num_feat)

    elif step == 3:
        num_feat = 29
        step3_estimate_contrib(num_feat, rank=24)
