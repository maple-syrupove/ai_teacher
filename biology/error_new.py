import json
from pathlib import Path

def extract_main_1_errors(input_dir: str, output_path: str):
    input_path = Path(input_dir)
    result = []

    # 遍历所有符合命名格式的文件
    for file in input_path.glob("question_*_error.json"):
        try:
            index_str = file.stem.split("_")[1]  # 提取 index
            question_index = int(index_str)

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                for item in data:
                    if item.get("main") == 1:
                        item["question_index"] = question_index
                        result.append(item)

        except Exception as e:
            print(f"❌ 处理文件 {file.name} 出错: {e}")

    # 输出结果
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(result, f_out, ensure_ascii=False, indent=4)

    print(f"✅ 提取完成，共提取 {len(result)} 条记录，已保存到 {output_path}")

# 执行脚本
if __name__ == "__main__":
    extract_main_1_errors("error_biology_all", "error_biology.json")
