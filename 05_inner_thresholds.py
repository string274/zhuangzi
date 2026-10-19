import os
import re
import json
import torch
import numpy as np
import torch.nn as nn
from model import TinyPOSLanguageModel

# ==================== 路径配置 ====================
base_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者"
vocab_path = os.path.join(base_dir, "vocab.json")
model_path = os.path.join(base_dir, "best_tiny_model.pt")
mean_vector_path = os.path.join(base_dir, "train_hidden_mean.npy")  # 由 内篇余弦值.py 生成
inner_dir = os.path.join(base_dir, "内_ids")  # 原始文本目录，用于提取POS序列

# ==================== 加载词表和模型 ====================
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinyPOSLanguageModel(vocab_size=vocab_size, num_layers=3, dropout=0.4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss(ignore_index=0)

# 加载内篇黄金质心（128维向量）
train_hidden_mean = np.load(mean_vector_path)

# ==================== 工具函数 ====================
def read_ids_from_file(file_path):
    raw_ids = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(']')
            content = parts[1] if len(parts) > 1 else parts[0]
            ids = [int(x) for x in content.strip().split() if x]
            raw_ids.extend(ids)
    return raw_ids

def make_sliding_blocks(pos_list, block_size=256, stride=128):
    """固定滑窗切分（与外杂篇.py完全一致）"""
    blocks = []
    n = len(pos_list)
    if n < block_size:
        if n > 0:
            blocks.append(pos_list)
        return blocks
    for i in range(0, n - block_size + 1, stride):
        blocks.append(pos_list[i : i + block_size])
    return blocks

def calc_block_metrics(block_ids):
    """
    计算单个固定滑窗块的 PPL 和 余弦距离
    block_ids 已经是数字 ID 列表
    """
    # 补齐或截断至256
    if len(block_ids) < 256:
        ids = block_ids + [0] * (256 - len(block_ids))
    else:
        ids = block_ids[:256]
    
    x = torch.tensor([ids[:-1]], dtype=torch.long).to(device)
    y = torch.tensor([ids[1:]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logits, hidden_states = model(x, return_intermediate=True)
        
        # 1. 计算 PPL
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        num_tokens = (y != 0).sum().item()
        ppl = torch.exp(loss).item() if num_tokens > 0 else float('inf')
        
        # 2. 提取有效句末的128维向量
        valid_len = (x[0] != 0).sum().item()
        if valid_len > 0:
            hidden_vector = hidden_states[0, valid_len - 1].cpu().numpy()
        else:
            hidden_vector = np.zeros(128)
        
        # 3. 计算与黄金质心的余弦距离
        dot_prod = np.dot(hidden_vector, train_hidden_mean)
        norm_a = np.linalg.norm(hidden_vector)
        norm_b = np.linalg.norm(train_hidden_mean)
        if norm_a > 0 and norm_b > 0:
            cosine_dist = 1.0 - (dot_prod / (norm_a * norm_b))
        else:
            cosine_dist = 1.0
            
    return ppl, cosine_dist
# ==================== 主程序：遍历内篇7篇 ====================
all_ppls = []
all_dists = []

print("🎯 开始用固定滑窗（256/128）通读内篇7篇...")
print("-" * 60)

for i in range(1, 8):
    file_path = os.path.join(inner_dir, f"ids_{i}.txt")
    if not os.path.exists(file_path):
        continue
    
    raw_ids = read_ids_from_file(file_path)
    blocks = make_sliding_blocks(raw_ids)
    
    print(f"  📄 {i}.txt : {len(blocks)} 个固定滑窗切片")
    
    for block in blocks:
        ppl, dist = calc_block_metrics(block)
        all_ppls.append(ppl)
        all_dists.append(dist)

# ==================== 计算阈值 ====================
N = len(all_ppls)
mu_ppl = np.mean(all_ppls)
sigma_ppl = np.std(all_ppls, ddof=1)
mu_dist = np.mean(all_dists)
sigma_dist = np.std(all_dists, ddof=1)

# 计算每个切片的 D_style
all_d_style = []
for ppl, dist in zip(all_ppls, all_dists):
    z_ppl = (ppl - mu_ppl) / sigma_ppl
    z_dist = (dist - mu_dist) / sigma_dist
    d_style = np.sqrt(z_ppl**2 + z_dist**2)
    all_d_style.append(d_style)

# 95% 经验分位数阈值
d_crit = np.percentile(all_d_style, 95)

# ==================== 输出结果 ====================
print("-" * 60)
print(f"✅ 内篇固定滑窗统计完成！总切片数: {N}")
print(f"  PPL 均值 μ = {mu_ppl:.2f}, 标准差 σ = {sigma_ppl:.2f}")
print(f"  Dist 均值 μ = {mu_dist:.3f}, 标准差 σ = {sigma_dist:.3f}")
print(f"  D_style 95% 经验分位数阈值 = {d_crit:.3f}")
print(f"  超过此阈值的内篇切片占比 = 5%（由定义保证）")
print("=" * 60)
stats = {
    "mu_ppl": mu_ppl,
    "sigma_ppl": sigma_ppl,
    "mu_dist": mu_dist,
    "sigma_dist": sigma_dist,
    "d_crit": d_crit,
    "n_slices": N
}
np.save(os.path.join(base_dir, "inner_stats.npy"), stats)
print(f"💾 统计参数 (μ, σ, D_crit) 已保存至 inner_stats.npy")