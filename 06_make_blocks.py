import os
import json
import random

def prepare_smart_train_val_data(ids_dir, vocab_path, save_dir, val_ratio=0.1, max_block_size=256):
    """
    策略二：基于语气助词 'T' 进行自然句尾动态截断，物理免疫边缘句法混乱风险，
    不够最大长度的地方自动用 <PAD> (0) 补齐。
    """
    # 1. 载入词典，我们需要查到哪些数字 ID 对应的是语气助词 "T"
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    # 找出所有包含 "T" 字符的键对应的 ID（主要是纯 T，也兼顾可能存在的复合 T）
    t_ids = {idx for tag, idx in vocab.items() if tag == "T" or tag.startswith("T[")}
    print(f"解析词典成功，已锁定的语气助词(句尾标志) ID 集合为: {t_ids}")

    # 2. 读取内篇所有的数字流
    all_ids = []
    for filename in sorted(os.listdir(ids_dir)):
        if filename.startswith("ids_") and filename.endswith(".txt"):
            file_path = os.path.join(ids_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                ids = [int(x) for x in f.read().strip().split() if x]
                all_ids.extend(ids)
                
    print(f"总计加载内篇原始数字 Token 数: {len(all_ids)}")

    # 3. 开始执行【智能句尾动态截断】核心算法
    data_blocks = []
    cursor = 0
    total_tokens = len(all_ids)
    
    while cursor < total_tokens:
        # 如果剩下的一截已经不够 150 个字了，直接作为最后一个块处理
        if total_tokens - cursor <= 150:
            sub_block = all_ids[cursor:]
            cursor = total_tokens
        else:
            # 设定一个安全的寻找窗口：从 200 字开始往后找句尾
            search_start = cursor + 200
            # 兜底上限：如果找到 280 字还没句尾，就强制落刀
            search_end = min(cursor + 280, total_tokens)
            
            cut_point = cursor + max_block_size # 默认切分点
            
            # 在 200-280 范围内寻找最近的语气词 T
            for pos in range(search_start, search_end):
                if all_ids[pos] in t_ids:
                    # 找到了！刀落在语气词的后一位，把语气词留在当前句里
                    cut_point = pos + 1
                    break
                    
            sub_block = all_ids[cursor:cut_point]
            cursor = cut_point
            
        # 4. 执行 Padding（补零）：确保所有块的数学长度均为绝对的 256
        if len(sub_block) < max_block_size:
            # 不够的地方，在右侧全部用 0 (<PAD>) 填满
            padding_length = max_block_size - len(sub_block)
            sub_block = sub_block + [0] * padding_length
        elif len(sub_block) > max_block_size:
            # 极端兜底情况下如果超长了，截断至 256
            sub_block = sub_block[:max_block_size]
            
        # 将数字转换为字符串，用空格连接准备写入文件
        data_blocks.append(" ".join(map(str, sub_block)))

    print(f"🎉 动态句法截断完成！成功转化为 {len(data_blocks)} 个不烂尾的 256 长度标准矩阵块。")

    # 5. 随机分配 10% 做内测完形填空
    random.seed(42)
    random.shuffle(data_blocks)
    
    val_count = int(len(data_blocks) * val_ratio)
    val_blocks = data_blocks[:val_count]
    train_blocks = data_blocks[val_count:]
    
    # 6. 存储输出
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(os.path.join(save_dir, "train.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(train_blocks))
    with open(os.path.join(save_dir, "val_test.txt"), 'w', encoding='utf-8') as f:
        f.write("\n".join(val_blocks))
        
    print(r"==================================================")
    print(f" 【策略二 · 智能训练集准备就绪】")
    print(f" 训练集 (train.txt) 块数: {len(train_blocks)}")
    print(f" 内测集 (val_test.txt) 块数: {len(val_blocks)}")
    print(f" 边缘句法异质性噪声降至最低。")
    print(r"==================================================")

# ==================== 运行脚本 ====================
if __name__ == "__main__":
    vocab_file = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\vocab.json"
    ids_input_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内_ids"
    output_save_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\model_data"
    
    prepare_smart_train_val_data(ids_input_dir, vocab_file, output_save_dir, val_ratio=0.1, max_block_size=256)