import json

def get_unique_types(file_path):
    unique_types = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 跳过空行
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if 'type' in data:
                        unique_types.add(data['type'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line[:50]}...")
                    continue
                    
        return list(unique_types)

    except FileNotFoundError:
        return "File not found."

# 使用示例
file_name = 'train-00000-of-00001 (1).jsonl'
result = get_unique_types(file_name)

# 打印结果
for t in result:
    print(t)