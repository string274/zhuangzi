import os
import torch
import json
import numpy as np
from model import TinyPOSLanguageModel

def export_pure_train_centroid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🌟 启动《庄子·内篇》黄金指纹质心固化管线")
    print(f"当前计算后端: {device}\n")
    
    # 🌟 统一绝对路径锁死
    base_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者"
    vocab_path = os.path.join(base_dir, "vocab.json")
    model_weight_path = os.path.join(base_dir, "best_tiny_model.pt")
    
    # 1. 读取词表以对齐矩阵维度
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # 2. 复活最终形态的预训练判官
    model = TinyPOSLanguageModel(vocab_size=vocab_size, num_layers=3, dropout=0.4).to(device)
    if os.path.exists(model_weight_path):
        print(f"🔌 成功注入最终最优预训练权重: {model_weight_path}")
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
    else:
        print(f"❌ 严重错误：未在路径下找到 best_tiny_model.pt！请确保预训练权重已保存。")
        return
        
    model.eval() # 开启评估模式，冻结参数与 Dropout
    
    # 3. 收集内篇 1-7 篇的全部数字 Token 流
    train_files = [f"内_ids\ids_{i}.txt" for i in range(1, 8)]
    all_inner_vectors = []
    
    print("\n🎬 开始通读内篇 1-7 篇以淬炼句法骨架...")
    
    for file_name in train_files:
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ 未找到内篇文件 {file_name}，跳过。")
            continue
            
        # 读取纯数字流
        raw_ids = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(']')
                content = parts[1] if len(parts) > 1 else parts[0]
                ids = [int(x) for x in content.strip().split() if x]
                raw_ids.extend(ids)
                
        # 仿照你的数据块切分逻辑（固定 256 长度，不重叠或重叠皆可，这里采用 256 块切分方案）
        window_size = 256
        total_tokens = len(raw_ids)
        
        file_vector_count = 0
        for i in range(0, total_tokens, window_size):
            chunk = raw_ids[i:i+window_size]
            if len(chunk) < window_size:
                chunk = chunk + [0] * (window_size - len(chunk)) # 右侧补零
                
            x_tensor = torch.tensor([chunk[:-1]], dtype=torch.long).to(device) # Batch_size = 1
            
            with torch.no_grad():
                # 核心拦截：提取中间第 2 层注意力层的隐藏状态
                _, hidden_states = model(x_tensor, return_intermediate=True)
                
                # 找到当前块的真实有效句长（排除末尾补零）
                valid_len = (x_tensor[0] != 0).sum().item()
                if valid_len > 0:
                    # 拦截该有效序列最后一位的 128维特征向量
                    h_vec = hidden_states[0, valid_len - 1].cpu().numpy()
                    all_inner_vectors.append(h_vec)
                    file_vector_count += 1
                    
        print(f"  ✅ {file_name} 通读完毕，成功提取出 {file_vector_count} 个高维句法指纹。")

    # 4. 🏆 实施统计学凝聚：计算 128 维特征空间的“绝对质心 (Centroid)”
    all_inner_vectors = np.array(all_inner_vectors)
    train_hidden_mean = np.mean(all_inner_vectors, axis=0)
    
    # 5. 💾 固化到硬盘
    output_npy_path = os.path.join(base_dir, "train_hidden_mean.npy")
    np.save(output_npy_path, train_hidden_mean)
    
    print("\n" + "="*60)
    print(f"🏆 历史性的高维指纹固化完成！")
    print(f"内篇总计 {len(all_inner_vectors)} 个文本块的特征矩阵已成功凝聚。")
    print(f"💾 黄金质心已安全保存至二进制文件: \n  --> {output_npy_path}")
    print("="*60)
    print("🎯 现在你的‘刻度尺桩子’已经焊死在硬盘里了。")

if __name__ == "__main__":
    export_pure_train_centroid()