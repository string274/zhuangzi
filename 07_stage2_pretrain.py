import os
import torch
import json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TinyPOSLanguageModel

class POSDataset(Dataset):
    def __init__(self, file_path):
        self.chunks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                ids = [int(x) for x in line.strip().split() if x]
                if ids:
                    self.chunks.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        token_tensor = self.chunks[idx]
        x = token_tensor[:-1]
        y = token_tensor[1:]
        return x, y

def evaluate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x) # 这里默认不抽取中间特征，只计算Loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            num_tokens = (y != 0).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    if total_tokens == 0: return float('inf')
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def train_and_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前激活的计算后端: {device}")
    
    vocab_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\vocab.json"
    train_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\model_data\train.txt"
    val_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\model_data\val_test.txt"
    model_save_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\best_tiny_model.pt"
    nei_ids_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内_ids"
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    train_dataset = POSDataset(train_path)
    val_dataset = POSDataset(val_path)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 🌟 调整为防御过拟合的黄金3层配置
    model = TinyPOSLanguageModel(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        hidden_dim=256,
        num_layers=3, # 升级为 3 层！
        dropout=0.4
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_ppl = float('inf')
    patience = 5  
    patience_counter = 0
    
    print("\n>>> 开始第一阶段内测：模型死磕训练与过拟合动态监控...")
    for epoch in range(1, 100):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        val_ppl = evaluate_perplexity(model, val_loader, criterion, device)
        print(f" Epoch {epoch:02d} | 训练平均Loss: {epoch_loss/len(train_loader):.4f} | 10%内测集 PPL: {val_ppl:.2f}")
        
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f">>> 提示：10%内测集PPL连续 {patience} 轮未获改善。触发早停，锁定最优权重。")
                break
                
    print(f"【第一阶段内测通过】最佳泛化状态下的内测集 PPL 成绩为: {best_val_ppl:.2f}")
    
    print("\n>>> 开始第二阶段内测：通读内篇独立篇章，确立真作 PPL 基准线...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    baseline_ppls = {}
    for filename in sorted(os.listdir(nei_ids_dir)):
        if filename.startswith("ids_") and filename.endswith(".txt"):
            file_path = os.path.join(nei_ids_dir, filename)
            single_dataset = POSDataset(file_path)
            single_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)
            single_ppl = evaluate_perplexity(model, single_loader, criterion, device)
            baseline_ppls[filename] = single_ppl
            print(f" 篇章基准线测评 -> {filename}: 固有 PPL = {single_ppl:.2f}")
            
    all_vals = list(baseline_ppls.values())
    avg_baseline = sum(all_vals) / len(all_vals)
    print(f"\n==================================================")
    print(f" 🎉 内篇闭环测试圆满结束！")
    print(f" 庄子真作固有句法困惑度（均值基准线）: {avg_baseline:.2f}")
    print(f"==================================================")

if __name__ == "__main__":
    train_and_baseline()