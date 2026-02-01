
import json
import re

def clean_answer(answer_str):
    """
    针对数学数据集答案的清洗函数
    """
    if not isinstance(answer_str, str):
        answer_str = str(answer_str)
    
    # =======================================================
    # 1. 清理 LaTeX 环境中的垃圾字符
    # =======================================================
    # 移除 LaTeX 的紧缩空格符 (\!) 和小空格 (\,)
    answer_str = re.sub(r'\\[!,;]', '', answer_str)
    
    # 移除美元符号 \$ 或 $
    answer_str = answer_str.replace('\\$', '').replace('$', '')

    # =======================================================
    # 2. 修复单位和文字 (\text, \mbox)
    # =======================================================
    
    # 常见的需要完全删除的单位词
    units_to_remove = [
        'cm', 'm', 'inches', 'feet', 'degrees', 'cents', 'dollars', 
        'rupee', 'kuna', 'pula', 'sq\\.?\\s*units', 'units'
    ]
    
    def text_replacer(match):
        full_match = match.group(0)       # 获取完整的 "\text{...}"
        content = match.group(1).strip()  # 获取括号内的内容 "..."
        
        # 检查 content 是否以单位结尾，或者本身就是单位
        for unit in units_to_remove:
            # 匹配如 " cm" 或 "5 cents" 中的 "cents"
            if re.search(r'\b' + unit + r'\b', content, re.IGNORECASE):
                # 尝试保留前面的数字，删除单位
                # 例如 \text{5 cents} -> 5
                num_part = re.search(r'[\d\.]+', content)
                return num_part.group(0) if num_part else ''
        
        # 【修改点】：如果不是单位（例如是人名 "Navin"），直接返回完整的原始匹配字符串
        # 这样就保留了 \text{} 包装
        return full_match

    # 匹配 \text{...} 或 \mbox{...}
    answer_str = re.sub(r'\\(?:text|mbox)\{([^\}]+)\}', text_replacer, answer_str)

    # 二次清理：有时单位不在 \text{} 里，而是直接跟在数字后面
    for unit in units_to_remove:
        answer_str = re.sub(r'\s+' + unit + r'\b', '', answer_str, flags=re.IGNORECASE)

    # =======================================================
    # 3. 修复度数符号
    # =======================================================
    answer_str = re.sub(r'\^?\{?\\circ\}?', '', answer_str)

    # =======================================================
    # 4. 修复数字中的逗号 (千分位)
    # =======================================================
    # 核心逻辑：只删除"两侧都是数字"且"右边正好是3位数字"的逗号
    while True:
        new_str = re.sub(r'(\d),\s*(\d{3})\b', r'\1\2', answer_str)
        if new_str == answer_str:
            break
        answer_str = new_str

    # =======================================================
    # 5. 修复空格
    # =======================================================
    # replace(" ", "") 会去除所有空格。
    # 对于保留下来的 \text{ Navin }，这里会变成 \text{Navin}，符合 LaTeX 标准格式
    answer_str = answer_str.replace(' ', '')
    
    # =======================================================
    # 6. 最终整理
    # =======================================================
    # 移除可能残留的 \boxed{} 包装
    match_boxed = re.search(r'\\boxed\{(.*)\}', answer_str)
    if match_boxed:
        answer_str = match_boxed.group(1)
        
    return answer_str.strip()

# ==========================================
# 执行文件处理
# ==========================================
input_file = './test.jsonl'        # 你的原始文件名
output_file = './data.jsonl'       # 输出文件名

# 统计修正计数
fixed_count = 0

print("开始清洗数据...")
with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line_num, line in enumerate(infile):
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
            if 'answer' in data:
                original = data['answer']
                cleaned = clean_answer(original)
                
                if original != cleaned:
                    fixed_count += 1
                    # 打印含有 text 的修复情况以供检查
                    if '\\text' in original:
                         print(f"Line {line_num+1} Fixed: [{original}]  ->  [{cleaned}]")
                
                data['answer'] = cleaned
            
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            
        except json.JSONDecodeError:
            print(f"Error parsing line {line_num}")

print(f"\n清洗完成！共修复了 {fixed_count} 条数据。结果已保存至 {output_file}")