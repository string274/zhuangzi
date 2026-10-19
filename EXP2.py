import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# ==========================================
# 0. 路径与篇名配置
# ==========================================
inner_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内"
outer_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\外杂"
output_txt_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\traditional2_results.txt"

CHAPTER_NAMES = {
    8: "骈拇", 9: "马蹄", 10: "胠箧", 11: "在宥", 12: "天地", 13: "天道", 14: "天运",
    15: "刻意", 16: "缮性", 17: "秋水", 18: "至乐", 19: "达生", 20: "山木", 21: "田子方",
    22: "知北游", 23: "庚桑楚", 24: "徐无鬼", 25: "则阳", 26: "外物", 27: "寓言",
    28: "让王", 29: "盗跖", 30: "说剑", 31: "渔父", 32: "列御寇", 33: "天下"
}

# ==========================================
# 1. 语料解析（只提取 POS 词性标签，剥离汉字）
# ==========================================
def extract_pos_tags(raw_text):
    """从中研院语料中解析出纯 POS 词性序列流"""
    pattern = re.compile(r'[\u4e00-\u9fa5]+\(([^)]+)\)')
    matches = pattern.findall(raw_text)
    return matches  # 返回词性字符串列表，如 ['NB4', 'VG', 'NA1', ...]

def make_sliding_blocks_strict(pos_list, block_size=256, stride=128):
    """单篇严格隔离切片，保证绝不跨篇混串"""
    blocks = []
    n = len(pos_list)
    if n < block_size:
        if n > 0:
            blocks.append(" ".join(pos_list))
        return blocks
    for i in range(0, n - block_size + 1, stride):
        block_text = " ".join(pos_list[i : i + block_size])
        blocks.append(block_text)
    return blocks

# ==========================================
# 2. 数据加载（内篇整篇，外篇单篇隔离切片）
# ==========================================
raw_inner_texts = []
for i in range(1, 8):
    file_path = os.path.join(inner_dir, f"{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_inner_texts.append(f.read())

raw_outer_texts = {}
for i in range(8, 34):
    file_path = os.path.join(outer_dir, f"{i}.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_outer_texts[i] = f.read()

# 内篇严格按单篇独立滑动切片
inner_chapters_blocks = []
inner_blocks = []
for i in range(1, 8):
  file_path = os.path.join(inner_dir, f"{i}.txt")
  if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
      pos_tags = extract_pos_tags(f.read())
      if pos_tags:
        blocks = make_sliding_blocks_strict(
            pos_tags, block_size=256, stride=128
        )
        inner_chapters_blocks.append(blocks)  # 新增：按篇保留
        inner_blocks.extend(blocks)

# 构建外篇切片字典 (严格单篇隔离)
outer_blocks_dict = {}
for chap_id, text in raw_outer_texts.items():
    pos_tags = extract_pos_tags(text)
    if not pos_tags:
        continue
    blocks = make_sliding_blocks_strict(pos_tags, block_size=256, stride=128)
    if blocks:
        outer_blocks_dict[chap_id] = blocks

# ==========================================
# 3. 提取 POS Bi-gram (2-gram) 特征矩阵
# ==========================================
# 设置 ngram_range=(2, 2) 提取双词性搭接特征；min_df=2 过滤全书偶然出现 1 次的噪点
vectorizer = CountVectorizer(
    ngram_range=(2, 2),
    token_pattern=r'(?u)\b\S+\b',
    min_df=2
)

# 拟合内篇频次矩阵并转换为相对词频（按各篇真实词数归一化）
# 拟合内篇切片频次矩阵并计算相对频次（每个切片除以 256.0）
X_inner_counts = vectorizer.fit_transform(inner_blocks).toarray()
X_inner_rel_freq = X_inner_counts / 256.0

# 计算内篇未经 Z-score 的原始 POS Bi-gram 风格质心
inner_centroid = np.mean(X_inner_rel_freq, axis=0)

# 标准化 (Z-score Scaling)
scaler = StandardScaler()
X_inner_scaled = scaler.fit_transform(X_inner_rel_freq)

def run_loco_cross_validation_pos2g(
    inner_chapters_blocks, vectorizer, nu_val=0.01
):
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
df_loco_pos2g, total_tp_2g, total_fp_2g = run_loco_cross_validation_pos2g(
    inner_chapters_blocks, vectorizer, nu_val=0.01
)

# ==========================================
# 4. One-Class SVM 模型训练
# ==========================================
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
oc_svm.fit(X_inner_scaled)

# ==========================================
# 5. 外杂篇测试与结果整理
# ==========================================
results = []
exp2_slice_dict = {}
for chap_id, blocks in outer_blocks_dict.items():
    chap_name = CHAPTER_NAMES.get(chap_id, f"篇目_{chap_id}")
    
    # 提取外篇 POS Bi-gram 相对词频（分母 256.0）与 Z-score
    X_outer_counts = vectorizer.transform(blocks).toarray()
    X_outer_rel_freq = X_outer_counts / 256.0
    X_outer_scaled = scaler.transform(X_outer_rel_freq)
    
    # 1. SVM 决策距离均值
    svm_scores = oc_svm.decision_function(X_outer_scaled)
    
    # 2. 与内篇 POS Bi-gram 质心的余弦距离
    cosine_dists = [
        1.0 - (np.dot(vec, inner_centroid) / (np.linalg.norm(vec) * np.linalg.norm(inner_centroid) + 1e-8))
        for vec in X_outer_rel_freq
    ]
    
    # 3. 离群切片比例
    outlier_ratio = np.mean(oc_svm.predict(X_outer_scaled) == -1)
    
    results.append({
        "Chapter_ID": chap_id,
        "Chapter_Name": chap_name,
        "POS2G_SVM_Decision_Score": round(np.mean(svm_scores), 4),
        "POS2G_Cosine_Distance": round(np.mean(cosine_dists), 4),
        "Outlier_Block_Ratio": f"{outlier_ratio * 100:.2f}%"
    })
    exp2_slice_dict[f"ids_{chap_id}.txt"] = [round(float(s), 4) for s in svm_scores]
df_pos2g_results = pd.DataFrame(results)

# ==========================================
# 6. 导出结果至 TXT 文件
# ==========================================
# ==========================================
# 6. 导出结果至 TXT 文件
# ==========================================
total_in_2g = total_tp_2g + total_fp_2g
loco_acc_2g = (total_tp_2g / total_in_2g) * 100 if total_in_2g > 0 else 0.0

output_content = "=" * 70 + "\n"
output_content += " 【基线实验二】POS Bi-gram (2-gram) + One-Class SVM 考据结果\n"
output_content += f" 提取到的 POS Bi-gram 有效特征维数: {X_inner_counts.shape[1]} 维\n"
output_content += "=" * 70 + "\n\n"

output_content += ">>> 模块一：内篇 7 篇严密留一交叉检验 (LOCO-CV) 真实正样本评估\n"
output_content += "-" * 70 + "\n"
output_content += df_loco_pos2g.to_string(index=False) + "\n"
output_content += "-" * 70 + "\n"
output_content += f"【内篇交叉检验全局汇总】: 总切片 = {total_in_2g} | TP (命中) = {total_tp_2g} | FP (误伤) = {total_fp_2g} | 真实保留率 = {loco_acc_2g:.2f}%\n\n"

output_content += ">>> 模块二：全本《庄子·外杂篇》普查扫描结果\n"
output_content += "-" * 70 + "\n"
output_content += df_pos2g_results.to_string(index=False) + "\n\n"
output_content += "-" * 70 + "\n"
output_content += "指标说明：\n"
output_content += "1. TP / FP: 内篇切片在未参与训练的样本外（Out-of-Sample）留一检验下的命中与误伤计数。\n"
output_content += "2. POS2G_SVM_Decision_Score: 基于双词性搭接特征的 SVM 决策超平面距离。\n"
output_content += "3. Outlier_Block_Ratio: 被 SVM 判定为离群文本块的百分比。\n"
output_content += "-" * 70 + "\n"

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(output_content)
# ==========================================
# 【追加】计算实验2（POS 2-gram）的内篇专属基准（μ + 2σ）
# ==========================================
inner_decision_scores = oc_svm.decision_function(X_inner_scaled)
inner_cos_dists = [
    1.0 - (np.dot(vec, inner_centroid) / (np.linalg.norm(vec) * np.linalg.norm(inner_centroid) + 1e-8))
    for vec in X_inner_rel_freq
]

mu_score, sigma_score = np.mean(inner_decision_scores), np.std(inner_decision_scores)
mu_dist, sigma_dist = np.mean(inner_cos_dists), np.std(inner_cos_dists)

threshold_score = mu_score - 2 * sigma_score
threshold_dist = mu_dist + 2 * sigma_dist

print(f"\n【实验2 专属基准线】")
print(f"SVM决策分数阈值 (μ-2σ): {threshold_score:.4f}")
print(f"余弦距离阈值 (μ+2σ): {threshold_dist:.4f}")
print(f"实验2 余弦 μ={mu_dist:.4f}, σ={sigma_dist:.4f}")
print(f"实验2 SVM  μ={mu_score:.4f}, σ={sigma_score:.4f}")
print("基线实验二完成！结果已成功导出至：")
print(output_txt_path)
print(f"\n捕获的 POS Bi-gram 特征维数: {X_inner_counts.shape[1]}")
print(f"【内篇 LOCO 汇总】TP = {total_tp_2g} | FP = {total_fp_2g} | 样本外保留率 = {loco_acc_2g:.2f}%")


# ==========================================
# 【就地画图】实验2 组合图（柱状+折线）
# ==========================================
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ===== 强制显示中文（Windows专用） =====
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ===== 直接从内存里拿数据 =====
df = df_pos2g_results  # 实验2的数据表

# ===== 实验2的专属阈值（你刚才算出来的） =====
threshold_cos = threshold_dist  # 余弦距离阈值
threshold_svm =threshold_score   # SVM决策分数阈值

# ===== 开始画图 =====
fig, ax1 = plt.subplots(figsize=(16, 6))

# --- 左轴：余弦距离（柱子） ---
x = np.arange(len(df))
bars = ax1.bar(x, df['POS2G_Cosine_Distance'], color='steelblue', alpha=0.7, label='余弦距离 (左轴)', zorder=2)


ax1.set_xlabel('外杂篇篇目', fontsize=12)
ax1.set_ylabel('余弦距离 ', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(0, max(df['POS2G_Cosine_Distance'].max() * 1.2, threshold_cos * 1.5))
ax1.axhline(y=threshold_cos, color='steelblue', linestyle='--', linewidth=2, 
            label=f'余弦阈值 (μ+2σ = {threshold_cos:.4f})', zorder=4)

# --- 右轴：SVM决策分数（折线） ---
ax2 = ax1.twinx()
ax2.plot(x, df['POS2G_SVM_Decision_Score'], color='darkorange', marker='o', linestyle='-', 
         linewidth=2, markersize=8, label='SVM决策距离 (右轴)', zorder=5)
ax2.axhline(y=threshold_svm, color='darkorange', linestyle='--', linewidth=2,
            label=f'SVM阈值 (μ-2σ = {threshold_svm:.4f})', zorder=4)
ax2.set_ylabel('SVM 决策距离', color='darkorange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange')

# 让SVM的0线居中偏上
ymin = min(df['POS2G_SVM_Decision_Score'].min() * 1.3, threshold_svm * 1.5)
ymax = max(df['POS2G_SVM_Decision_Score'].max() * 1.3, 0.1)
ax2.set_ylim(ymin, ymax)

# --- 装饰：X轴篇名 ---
ax1.set_xticks(x)
ax1.set_xticklabels(df['Chapter_Name'], rotation=45, ha='right', fontsize=10)

# --- 合并图例（左轴+右轴） ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)

plt.grid(axis='x', linestyle=':', alpha=0.3)
plt.tight_layout()

# --- 保存高清图到当前文件夹 ---
plt.savefig('exp2_combined_chart.png', dpi=300, bbox_inches='tight')
print("\n✅ 实验2 组合图已保存为：exp2_combined_chart.png")

# --- 弹出显示 ---

import json
with open(r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\exp2_slice_scores.json", "w", encoding="utf-8") as f:
    json.dump(exp2_slice_dict, f, ensure_ascii=False, indent=2)
plt.show()