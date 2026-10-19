import os
import json

def build_vocabulary(cleaned_dirs, vocab_save_path):
    """
    第一步：扫描所有指定的清洗后文本文件夹，自动收集所有不重复的细粒度词性组合，构建完美的字典。
    """
    # 放入最核心的两个大模型控制符号
    # <PAD> 用于后续滑动窗口或批处理时，给不够长的句子补零
    # <UNK> 用于防止未来遇到未知的脏标签时模型崩溃
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    
    unique_tags = set()
    
    # 支持同时扫描多个清洗后的文件夹（内篇和外杂篇一起扫描，确保词典最全）
    for cleaned_dir in cleaned_dirs:
        if not os.path.exists(cleaned_dir):
            continue
        for filename in sorted(os.listdir(cleaned_dir)):
            if filename.startswith("cleaned_") and filename.endswith(".txt"):
                file_path = os.path.join(cleaned_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 按照空格切分出所有的 Token (如 NB1[+prop], VG 等)
                        tokens = content.split()
                        unique_tags.update(tokens)
                        
    # 将所有实际出现过的组合标签（包括大类、活用拼接后的整体）按字母排序后编上数字号
    for idx, tag in enumerate(sorted(list(unique_tags)), start=2):
        vocab[tag] = idx
        
    # 将词典保存为 JSON 文件，方便写论文和模型以后调用
    with open(vocab_save_path, 'w', encoding='utf-8') as f_out:
        json.dump(vocab, f_out, ensure_ascii=False, indent=4)
        
    print("==================================================")
    print(" 【第一阶段】全量专属词典构建成功！")
    print(f" 词典总大小 (Vocab Size): {len(vocab)} (已包含 2 个控制符号)")
    print(f" 词典已保存至: {vocab_save_path}")
    print("==================================================")
    return vocab

def convert_directory_to_ids(cleaned_dir, ids_output_dir, vocab, folder_name=""):
    """
    第二步：利用构建好的词典，批量把某个文件夹下的文本标签序列（"VG NA1"）翻译成数字序列（"3 4"）。
    """
    if not os.path.exists(ids_output_dir):
        os.makedirs(ids_output_dir)
        
    print(f"\n正在数字化转换【{folder_name}】...")
    
    for filename in sorted(os.listdir(cleaned_dir)):
        if filename.startswith("cleaned_") and filename.endswith(".txt"):
            input_path = os.path.join(cleaned_dir, filename)
            # 输出文件名把 cleaned_ 替换成 ids_
            output_path = os.path.join(ids_output_dir, filename.replace("cleaned_", "ids_"))
            
            with open(input_path, 'r', encoding='utf-8') as f:
                tokens = f.read().strip().split()
                
            # 查表翻译：如果在词典里就变成数字，不在就变成 <UNK> 的数字 1
            id_sequence = [str(vocab.get(token, 1)) for token in tokens]
            
            # 将数字序列用空格连起来保存
            with open(output_path, 'w', encoding='utf-8') as f_out:
                f_out.write(" ".join(id_sequence))
                
            print(f" 转化成功: {filename} -> {os.path.basename(output_path)} (包含 {len(id_sequence)} 个数字)")


# ==================== 执行词典构建与数字化 ====================
if __name__ == "__main__":
    
    # 1. 填入你上一步生成的两个清洗后语料文件夹路径
    nei_cleaned_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内_cleaned"
    wai_cleaned_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\外杂_cleaned"
    
    # 2. 配置文件保存的目标路径
    vocab_file_path = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\vocab.json"
    nei_ids_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内_ids"
    wai_ids_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\外杂_ids"
    
    # 3. 汇总文件夹进行词典提取（合并内篇和外杂篇，防止漏掉任何一个低频活用标签）
    all_cleaned_dirs = [nei_cleaned_dir, wai_cleaned_dir]
    my_vocab = build_vocabulary(all_cleaned_dirs, vocab_file_path)
    
    # 4. 分别对【内篇】和【外杂篇】执行数字化编码
    if os.path.exists(nei_cleaned_dir):
        convert_directory_to_ids(nei_cleaned_dir, nei_ids_dir, my_vocab, folder_name="内篇")
        
    if os.path.exists(wai_cleaned_dir):
        convert_directory_to_ids(wai_cleaned_dir, wai_ids_dir, my_vocab, folder_name="外杂篇")

    print(" 【全部数据数字化大功告成！】")