import json
from pathlib import Path
from typing import List, Dict

def extract_main_1_errors(input_dir: str, output_path: str):
    """
    从指定目录的 "question_*_error.json" 文件中，
    仅提取 "main" 字段为 1 的主要错误。

    Args:
        input_dir (str): 包含JSON文件的输入目录路径。
        output_path (str): 输出JSON文件的路径。
    """
    input_path = Path(input_dir)
    result: List[Dict] = []

    print(f"--- 开始提取主要错误 (main=1) ---")
    # 遍历所有符合命名格式的文件
    for file in input_path.glob("question_*_error.json"):
        try:
            # 从文件名 "question_123_error.json" 中提取索引 "123"
            index_str = file.stem.split("_")[1]
            question_index = int(index_str)

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 遍历文件中的每个错误项
                for item in data:
                    # 只选择 "main" 字段为 1 的项
                    if item.get("main") == 1:
                        item["question_index"] = question_index
                        result.append(item)

        except Exception as e:
            print(f"❌ 处理文件 {file.name} 出错: {e}")

    # 将提取结果写入输出文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(result, f_out, ensure_ascii=False, indent=4)

    print(f"✅ 主要错误提取完成，共提取 {len(result)} 条记录，已保存到 {output_path}")

def extract_all_errors(input_dir: str, output_path: str):
    """
    从指定目录的 "question_*_error.json" 文件中，
    提取所有错误，不考虑 "main" 字段的值。

    Args:
        input_dir (str): 包含JSON文件的输入目录路径。
        output_path (str): 输出JSON文件的路径。
    """
    input_path = Path(input_dir)
    result: List[Dict] = []

    print(f"\n--- 开始提取所有错误 ---")
    # 遍历所有符合命名格式的文件
    for file in input_path.glob("question_*_error.json"):
        try:
            # 从文件名 "question_123_error.json" 中提取索引 "123"
            index_str = file.stem.split("_")[1]
            question_index = int(index_str)

            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

                # 遍历文件中的每个错误项
                for item in data:
                    # **不再有 if 条件，直接处理所有项**
                    item["question_index"] = question_index
                    result.append(item)

        except Exception as e:
            print(f"❌ 处理文件 {file.name} 出错: {e}")

    # 将提取结果写入输出文件
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(result, f_out, ensure_ascii=False, indent=4)

    print(f"✅ 全部错误提取完成，共提取 {len(result)} 条记录，已保存到 {output_path}")

def extract_errors_by_index(input_dir: str, output_path: str, target_index: int):
    """
    从指定目录的 "question_*_error.json" 文件中，
    提取所有错误项中 index == target_index 的项（不考虑 main 字段）。

    Args:
        input_dir (str): 包含JSON文件的输入目录路径。
        output_path (str): 输出JSON文件的路径。
        target_index (int): 只提取该 index 的错误项。
    """
    input_path = Path(input_dir)
    result: List[Dict] = []

    print(f"\n--- 开始提取所有错误项中 index={target_index} 的错误 ---")
    for file in input_path.glob("question_*_error.json"):
        try:
            # 从文件名 "question_123_error.json" 中提取索引 "123"
            file_index_str = file.stem.split("_")[1]
            file_question_index = int(file_index_str)
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    if item.get("index") == target_index:
                        item["question_index"] = file_question_index
                        result.append(item)
        except Exception as e:
            print(f"❌ 处理文件 {file.name} 出错: {e}")
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(result, f_out, ensure_ascii=False, indent=4)
    print(f"✅ index={target_index} 的错误项提取完成，共提取 {len(result)} 条记录，已保存到 {output_path}")


# ===== 执行脚本 =====
if __name__ == "__main__":
    #INPUT_DIRECTORY = "error_math_new"
    
    # 1. 调用旧函数，只提取 main=1 的错误
    #OUTPUT_MAIN_ERRORS = "errors.json"
    #extract_main_1_errors(INPUT_DIRECTORY, OUTPUT_MAIN_ERRORS)
    extract_errors_by_index("error_math_100", "errors5.json", 5)
    # 2. 调用新函数，提取所有错误
    #OUTPUT_ALL_ERRORS = "all_errors.json"
    #extract_all_errors(INPUT_DIRECTORY, OUTPUT_ALL_ERRORS)