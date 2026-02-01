import pandas as pd
import os

def convert_parquet_to_jsonl(input_file, output_file):
    print(f"正在读取 {input_file} ...")
    
    try:
        # 1. 读取 parquet 文件
        df = pd.read_parquet(input_file, engine='pyarrow')
        
        print(f"读取成功，共 {len(df)} 行数据。正在写入 {output_file} ...")
        
        # 2. 写入 jsonl 文件
        # orient='records': 将每一行转换为一个字典
        # lines=True: 开启 jsonl 模式（换行符分隔）
        # force_ascii=False: 防止中文或特殊数学符号变成乱码 (\uXXXX)
        df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
        print("转换完成！")
        
    except Exception as e:
        print(f"转换出错: {e}")

if __name__ == "__main__":
    # 设置输入和输出文件名
    input_path = "train-9k.parquet"
    output_path = "train-9k.jsonl"
    
    if os.path.exists(input_path):
        convert_parquet_to_jsonl(input_path, output_path)
    else:
        print(f"错误: 找不到文件 {input_path}")