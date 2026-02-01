import json
import os

def extract_answers_to_txt(jsonl_file, txt_file):
    if not os.path.exists(jsonl_file):
        print(f"❌ 错误：找不到文件 {jsonl_file}")
        return

    print(f"正在处理 {jsonl_file} ...")
    
    count = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f_in, \
         open(txt_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # 1. 优先查找 'final_answer' (图片显示的字段)
                # 2. 如果没有，查找 'answer' (如果是你之前转换过的版本)
                # 3. 如果都没有，查找 'solution' (作为兜底，虽然通常很长)
                ans_content = None
                
                if 'final_answer' in data:
                    ans_content = data['final_answer']
                elif 'answer' in data:
                    ans_content = data['answer']
                elif 'solution' in data:
                    # 注意：solution 通常很长，如果只要最终数值，通常不取这个
                    ans_content = data['solution']
                
                # 处理数据格式
                # 图片中显示 final_answer 是列表格式 ["2"]，我们需要提取里面的字符串
                if isinstance(ans_content, list):
                    if len(ans_content) > 0:
                        ans_content = ans_content[0] # 提取列表第一个元素
                    else:
                        ans_content = ""
                
                # 确保转为字符串并写入，去除首尾空白
                final_str = str(ans_content).strip()
                f_out.write(final_str + "\n")
                count += 1
                
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无法解析的行")
                continue

    print(f"✅ 提取完成！")
    print(f"📄 共提取 {count} 行答案")
    print(f"💾 结果已保存至: {txt_file}")

if __name__ == "__main__":
    input_filename = "College_math.jsonl"
    output_filename = "College_math.txt"
    
    extract_answers_to_txt(input_filename, output_filename)