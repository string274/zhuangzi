import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ==========================================
# 0. 映射表与文件路径配置
# ==========================================
inner_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内"
outer_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\外杂"
output_txt_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\traditional_results.txt"

# 外杂篇 ID 到中文篇名的映射表
CHAPTER_NAMES = {
    8: "骈拇", 9: "马蹄", 10: "胠箧", 11: "在宥", 12: "天地", 13: "天道", 14: "天运",
    15: "刻意", 16: "缮性", 17: "秋水", 18: "至乐", 19: "达生", 20: "山木", 21: "田子方",
    22: "知北游", 23: "庚桑楚", 24: "徐无鬼", 25: "则阳", 26: "外物", 27: "寓言",
    28: "让王", 29: "盗跖", 30: "说剑", 31: "渔父", 32: "列御寇", 33: "天下"
}

# ==========================================
# 1. 定义 20 个核心功能词表
# ==========================================
FUNCTION_WORDS_20 = [
    # 人称代词 (6)
    "吾", "我", "予", "余", "尔", "汝",
    # 助动词 (2)
    "犹", "必",
    # 介词 (3)
    "以", "与", "于",
    # 连词 (2)
    "而", "则",
    # 副词 (3)
    "皆", "诚", "岂",
    # 句末语气词 (4)
    "也", "矣", "哉", "邪"
]

# ==========================================
# 2. 语料解析与滑动切片函数
# ==========================================
def parse_and_extract_words(raw_text):
    """从中研院语料中解析出纯汉字词流，剥离 POS 标签"""
    pattern = re.compile(r'([\u4e00-\u9fa5]+)\(([^)]+)\)')
    matches = pattern.findall(raw_text)
    return [m[0] for m in matches]

def make_sliding_blocks(words_list, block_size=256, stride=128):
    """单篇严格隔离切片，绝对不会跨篇混串"""
    blocks = []
    n = len(words_list)
    if n < block_size:
        if n > 0:
            blocks.append(" ".join(words_list))
        return blocks
    for i in range(0, n - block_size + 1, stride):
        block_text = " ".join(words_list[i : i + block_size])
        blocks.append(block_text)
    return blocks

# ==========================================
# 3. 数据加载 
# ==========================================
raw_inner_texts = []
for i in range(1, 8):
    file_path = os.path.join(inner_dir, f"{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_inner_texts.append(f.read())

raw_outer_texts = {} # 修正为字典结构
for i in range(8, 34):
    file_path = os.path.join(outer_dir, f"{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_outer_texts[i] = f.read()

# 内篇不滑块，直接以 7 篇为 7 个整体大块

# 内篇严格按单篇独立滑动切片
# 找到原代码第 3 节的 inner_blocks 循环，替换为：
inner_chapters_blocks = []
inner_blocks = []
for i in range(1, 8):
  file_path = os.path.join(inner_dir, f"{i}.txt")
  if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      words = parse_and_extract_words(f.read())
      if words:
        blocks = make_sliding_blocks(words, block_size=256, stride=128)
        inner_chapters_blocks.append(blocks)  # 新增：按篇保留
        inner_blocks.extend(blocks)

# 构建外篇切片字典
outer_blocks_dict = {}
for chap_id, text in raw_outer_texts.items():
    words = parse_and_extract_words(text)
    blocks = make_sliding_blocks(words, block_size=256, stride=128)
    if blocks:
        outer_blocks_dict[chap_id] = blocks

# ==========================================
# 4. 20 功能词相对词频提取与 Z-score 标准化
# ==========================================
vectorizer = CountVectorizer(
    vocabulary=FUNCTION_WORDS_20,
    token_pattern=r'(?u)\b\w+\b'
)


# 计算内篇切片的 20 个功能词相对词频（每个切片除以 256.0）
X_inner_counts = vectorizer.fit_transform(inner_blocks).toarray()
X_inner_rel_freq = X_inner_counts / 256.0

# 标准化 (Z-score Scaling)
scaler = StandardScaler()
X_inner_scaled = scaler.fit_transform(X_inner_rel_freq)

def run_loco_cross_validation(inner_chapters_blocks, vectorizer, nu_val=0.1):
  loco_details, total_tp, total_fp = [], 0, 0
  for test_idx in range(len(inner_chapters_blocks)):
    train_blocks = [
        b
        for idx, ch in enumerate(inner_chapters_blocks)
        if idx != test_idx
        for b in ch
    ]
    test_blocks = inner_chapters_blocks[test_idx]

    X_tr = vectorizer.transform(train_blocks).toarray() / 256.0
    X_te = vectorizer.transform(test_blocks).toarray() / 256.0

    scaler_loco = StandardScaler()
    X_tr_s = scaler_loco.fit_transform(X_tr)
    X_te_s = scaler_loco.transform(X_te)

    svm_loco = OneClassSVM(kernel="rbf", gamma="scale", nu=nu_val).fit(X_tr_s)
    preds = svm_loco.predict(X_te_s)

    tp_i, fp_i = int(np.sum(preds == 1)), int(np.sum(preds == -1))
    total_tp += tp_i
    total_fp += fp_i
    loco_details.append({
        "留出测试篇章": f"内篇_ids_{test_idx+1}.txt",
        "总切片数": len(test_blocks),
        "TP": tp_i,
        "FP": fp_i,
        "真作保留率": f"{(tp_i / len(test_blocks)) * 100:.2f}%",
    })
  return pd.DataFrame(loco_details), total_tp, total_fp


# 执行函数
df_loco_summary, total_tp, total_fp = run_loco_cross_validation(
    inner_chapters_blocks, vectorizer, nu_val=0.1
)

# 计算内篇风格中心 (20 维向量)
inner_centroid = np.mean(X_inner_rel_freq, axis=0)

# ==========================================
# 5. One-Class SVM 模型训练
# ==========================================
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
oc_svm.fit(X_inner_scaled)

# ==========================================
# 6. 外杂篇测试与结果整理
# ==========================================
results = []
exp1_slice_dict = {}
for chap_id, blocks in outer_blocks_dict.items():
    chap_name = CHAPTER_NAMES.get(chap_id, f"篇目_{chap_id}")
    
    # 计算外篇相对词频与 Z-score
    X_outer_counts = vectorizer.transform(blocks).toarray()
    X_outer_rel_freq = X_outer_counts / 256.0
    X_outer_scaled = scaler.transform(X_outer_rel_freq)
    
    # 1. SVM 决策边界距离 (越小越偏离)
    svm_scores = oc_svm.decision_function(X_outer_scaled)
    
    # 2. 与内篇中心的余弦距离
    cosine_dists = [
        1.0 - (np.dot(vec, inner_centroid) / (np.linalg.norm(vec) * np.linalg.norm(inner_centroid) + 1e-8))
        for vec in X_outer_rel_freq
    ]
    
    # 3. 离群切片比例
    outlier_ratio = np.mean(oc_svm.predict(X_outer_scaled) == -1)
    
    results.append({
        "Chapter_ID": chap_id,
        "Chapter_Name": chap_name,
        "Total_Blocks": len(blocks),  
        "FW_SVM_Decision_Score": round(np.mean(svm_scores), 4),
        "FW_Cosine_Distance": round(np.mean(cosine_dists), 4),
        "Outlier_Block_Ratio": f"{outlier_ratio * 100:.2f}%"
    })
    exp1_slice_dict[f"ids_{chap_id}.txt"] = [round(float(s), 4) for s in svm_scores]
df_fw_svm_results = pd.DataFrame(results)

# ==========================================
# 7. 导出写入 TXT 文档
# ==========================================

total_in = total_tp + total_fp
loco_acc = (total_tp / total_in) * 100 if total_in > 0 else 0.0

output_content = "=" * 70 + "\n"
output_content += " 【基线实验一】20 个功能词相对词频 + One-Class SVM 考据结果\n"
output_content += "=" * 70 + "\n\n"

output_content += ">>> 模块一：内篇 7 篇严密留一交叉检验 (LOCO-CV) 真实正样本评估\n"
output_content += "-" * 70 + "\n"
output_content += df_loco_summary.to_string(index=False) + "\n"
output_content += "-" * 70 + "\n"
output_content += f"【内篇交叉检验全局汇总】: 总切片 = {total_in} | TP (命中) = {total_tp} | FP (误伤) = {total_fp} | 真实保留率 = {loco_acc:.2f}%\n\n"

output_content += ">>> 模块二：全本《庄子·外杂篇》普查扫描结果\n"
output_content += "-" * 70 + "\n"
output_content += df_fw_svm_results.to_string(index=False) + "\n\n"
output_content += "-" * 70 + "\n"
output_content += "指标说明：\n"
output_content += "1. TP / FP: 内篇切片在未参与训练的样本外（Out-of-Sample）留一检验下的命中与误伤计数。\n"
output_content += "2. FW_SVM_Decision_Score: 超平面决策距离，数值越负代表偏离内篇越远。\n"
output_content += "3. FW_Cosine_Distance: 与内篇 20 维功能词风格质心的余弦距离。\n"
output_content += "4. Outlier_Block_Ratio: 该篇被 SVM 判定为离群/伪作文本块的百分比。\n"
output_content += "-" * 70 + "\n"

# 写入本地文件
with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(output_content)

print("实验完成！结果已成功写入文件：")
print(output_txt_path)
print(f"\n【内篇 LOCO 汇总】TP = {total_tp} | FP = {total_fp} | 真实保留率 = {loco_acc:.2f}%")
print("\n部分外杂篇结果预览：")
# ==========================================
# 【追加】计算实验1（功能词）的内篇专属基准（μ + 2σ）
# ==========================================
# 1. 计算全量内篇切片在训练好的SVM上的决策分数
inner_decision_scores = oc_svm.decision_function(X_inner_scaled)
# 2. 计算全量内篇切片到风格质心的余弦距离
inner_cos_dists = [
    1.0 - (np.dot(vec, inner_centroid) / (np.linalg.norm(vec) * np.linalg.norm(inner_centroid) + 1e-8))
    for vec in X_inner_rel_freq
]

mu_score, sigma_score = np.mean(inner_decision_scores), np.std(inner_decision_scores)
mu_dist, sigma_dist = np.mean(inner_cos_dists), np.std(inner_cos_dists)

threshold_score = mu_score - 2 * sigma_score  # 注意：SVM分数是越大越像内篇，所以离群阈值在低分端
threshold_dist = mu_dist + 2 * sigma_dist


print(f"\n【实验1 专属基准线】")
print(f"SVM决策分数阈值 (μ-2σ): {threshold_score:.4f} (低于此值判定为显著离群)")
print(f"余弦距离阈值 (μ+2σ): {threshold_dist:.4f} (高于此值判定为显著离群)")
print(f"实验1 余弦 μ={mu_dist:.4f}, σ={sigma_dist:.4f}")
print(f"实验1 SVM  μ={mu_score:.4f}, σ={sigma_score:.4f}")
print(df_fw_svm_results.head(10).to_string(index=False))


# ==========================================
# 【就地画图】实验1 组合图（柱状+折线）
# ==========================================
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ===== 强制让Matplotlib显示中文（Windows专用） =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']  # 优先用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号（-）显示成方块的问题

# 直接用内存里的 df_fw_svm_results 变量
df = df_fw_svm_results

# 你刚算出来的专属阈值
threshold_cos = threshold_dist  # 直接用原始浮点数，比如 0.266547... 
threshold_svm = threshold_score

fig, ax1 = plt.subplots(figsize=(16, 6))

# --- 左轴：余弦距离（柱子） ---
x = np.arange(len(df))
bars = ax1.bar(x, df['FW_Cosine_Distance'], color='steelblue', alpha=0.7, label='余弦距离 (左轴)', zorder=2)


ax1.set_xlabel('外杂篇篇目', fontsize=12)
ax1.set_ylabel('余弦距离', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(0, max(df['FW_Cosine_Distance'].max() * 1.2, threshold_cos * 1.5))
ax1.axhline(y=threshold_cos, color='steelblue', linestyle='--', linewidth=2, 
            label=f'余弦阈值 (μ+2σ = {threshold_cos:.3f})', zorder=4)

# --- 右轴：SVM决策分数（折线） ---
ax2 = ax1.twinx()
ax2.plot(x, df['FW_SVM_Decision_Score'], color='darkorange', marker='o', linestyle='-', 
         linewidth=2, markersize=8, label='SVM决策距离 (右轴)', zorder=5)
ax2.axhline(y=threshold_svm, color='darkorange', linestyle='--', linewidth=2,
            label=f'SVM阈值 (μ-2σ = {threshold_svm:.3f})', zorder=4)
ax2.set_ylabel('SVM 决策距离', color='darkorange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange')

# 设置Y轴范围，让0线显眼
ymin = min(df['FW_SVM_Decision_Score'].min() * 1.3, threshold_svm * 1.5)
ymax = max(df['FW_SVM_Decision_Score'].max() * 1.3, 0.1)
ax2.set_ylim(ymin, ymax)

# --- 装饰 ---
ax1.set_xticks(x)
ax1.set_xticklabels(df['Chapter_Name'], rotation=45, ha='right', fontsize=10)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)

plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.tight_layout()

# 保存到当前目录
plt.savefig('exp1_combined_chart.png', dpi=300, bbox_inches='tight')
print("\n✅ 实验1 组合图已保存为：exp1_combined_chart.png")

import json
with open(r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\exp1_slice_scores.json", "w", encoding="utf-8") as f:
    json.dump(exp1_slice_dict, f, ensure_ascii=False, indent=2)
plt.show()