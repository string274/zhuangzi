import os
import re

def clean_ancient_text(input_file_path, output_file_path):
    """
    极精细清洗：确保只剔除带 Error 的单个 Token，同行前后所有 Token 绝对不漏。
    """
    # 1. 匹配出每一个独立的词性单元（包含它后面可能带有的 Error 注释）
    # 示例匹配： 於(P) 或 是(NH) 或 は(VG)/* Error */
    token_unit_pattern = re.compile(r'[^()\s]+\([^)]+\)(?:\[[^\]]+\])?(?:\s*/\* Error \*/)?')
    
    # 2. 从合法的词性单元中提取核心标签的正则
    pos_extract_pattern = re.compile(r'\(([^)]+)\)(?:\[([^\]]+)\])?')
    
    cleaned_tokens = []
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 过滤纯注释行
            if line.strip().startswith("/*") and line.strip().endswith("*/") and "Error" not in line:
                continue
            
            # 找出这一行中所有的词性单元块
            units = token_unit_pattern.findall(line)
            
            for unit in units:
                # 如果这个单元里包含 Error，只跳过当前这一个单元！
                if "/* Error */" in unit:
                    continue
                
                # 从正常的单元中提取词性
                match = pos_extract_pattern.search(unit)
                if match:
                    main_pos = match.group(1)
                    attr = match.group(2)
                    
                    # 拼接细粒度标签
                    if attr:
                        token = f"{main_pos}[{attr}]"
                    else:
                        token = main_pos
                    
                    cleaned_tokens.append(token)
                    
    # 用空格拼接
    output_content = " ".join(cleaned_tokens)
    
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write(output_content)
    print(f"【精准清洗完成】: {os.path.basename(input_file_path)} -> 词性数: {len(cleaned_tokens)}")


# ==================== 正式批量清洗运行 ====================
if __name__ == "__main__":
    
    # 1. 内部临时定义：批量循环处理文件夹的函数上
    def batch_process_directory(input_dir, output_dir):
        # 如果输出文件夹不存在，Python 会自动帮你建立
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        for filename in sorted(os.listdir(input_dir)):
            # 兼容 .txt 结尾或者没有后缀的纯数字文件名
            if filename.endswith(".txt") or filename.isdigit() or "." in filename: 
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"cleaned_{filename}")
                
                # 只有当它是一个真正的文件时才调用清洗函数
                if os.path.isfile(input_path):
                    clean_ancient_text(input_path, output_path)

    # 2. 配置你的【内篇】路径并执行清洗
    nei_input_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内"
    nei_output_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\内_cleaned"
    
    print("【第一阶段】开始精准批量清洗——内篇语料...")
    batch_process_directory(nei_input_dir, nei_output_dir)
    print(">>> 内篇清洗完毕！\n")
    
    # 3. 配置你的【外杂篇】路径并执行清洗
    wai_input_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\外杂"
    wai_output_dir = r"C:\Users\String Quartet\Desktop\专业课\myself\庄子作者\外杂_cleaned"
    
    print("【第二阶段】开始精准批量清洗——外杂篇语料...")
    batch_process_directory(wai_input_dir, wai_output_dir)
    print(">>> 外杂篇清洗完毕！\n")
    
    print("==========================================")
    print(" 【全部语料预处理成功！】")
    print(" 干净的词性流已分别存入 内_cleaned 和 外杂_cleaned")
    print("==========================================")
    