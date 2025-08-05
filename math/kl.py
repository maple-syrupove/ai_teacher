import os
import json
import re

def parse_and_unroll_dialogue(history_text, target_format_func):
    """
    解析history_prompt文本，并将其展开为多个训练样本。
    """
    # --- 这里是唯一的修改点 ---
    # 将正则表达式从寻找中文的“学生：”改为英文的“Student：”
    parts = re.split(r'(Student：|Teacher：)', history_text.strip())
    # --- 修改结束 ---

    # 剔除分割后可能产生的开头的空字符串或问题描述
    if not parts[0].strip():
        parts = parts[1:]
    
    question = parts[0]
    # 如果第一部分不是分隔符，说明它是问题描述，将其与第一个角色发言合并
    if 'Student：' not in parts[0] and 'Teacher：' not in parts[0]:
        # 将问题描述和第一轮发言内容合并
        if len(parts) > 2:
            parts[2] = parts[0].strip() + " " + parts[2].strip()
        parts = parts[1:] # 剔除问题描述部分
    
    dialogue_turns = []
    for i in range(0, len(parts), 2):
        # 确保 parts[i+1] 存在，防止索引越界
        if i + 1 < len(parts):
            role = "student" if "Student" in parts[i] else "teacher"
            content = parts[i+1].strip()
            
            # 忽略最后的 "zzc" 回答
            if role == "student" and content.lower() == "zzc":
                continue
            dialogue_turns.append({"role": role, "content": content})

    unrolled_samples = []
    for i in range(len(dialogue_turns)):
        if dialogue_turns[i]["role"] == "teacher":
            current_dialogue_slice = dialogue_turns[:i+1]
            formatted_sample = target_format_func(current_dialogue_slice)
            if formatted_sample:
                unrolled_samples.append(formatted_sample)

    return unrolled_samples

def format_to_alpaca_history(dialogue_slice):
    """
    将对话切片格式化为Alpaca history样式。
    （此函数无需修改）
    """
    if not dialogue_slice or dialogue_slice[-1]["role"] != "teacher":
        return None
    if len(dialogue_slice) < 2 or dialogue_slice[-2]["role"] != "student":
        return None

    output = dialogue_slice[-1]["content"]
    instruction = dialogue_slice[-2]["content"]
    history_pairs = []
    history_source = dialogue_slice[:-2]
    if len(history_source) % 2 == 0:
        for i in range(0, len(history_source), 2):
            if history_source[i]["role"] == "student" and history_source[i+1]["role"] == "teacher":
                history_pairs.append([
                    history_source[i]["content"], 
                    history_source[i+1]["content"]
                ])
            else:
                return None
    else: # 历史记录不是成对出现
        return None
    
    return {
        "instruction": instruction,
        "output": output,
        "history": history_pairs
    }

def process_single_folder(input_folder):
    """
    处理【单个】文件夹内的所有JSON文件，并返回一个包含所有样本的列表。
    （此函数无需修改）
    """
    all_processed_samples = []
    print(f"📁 开始处理文件夹: {input_folder}")

    if not os.path.isdir(input_folder):
        print(f"   - 错误：文件夹 '{input_folder}' 不存在。")
        return all_processed_samples

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                round_used = data.get("rounds_used") # 修正键名
                if round_used is not None:
                    try:
                        if int(round_used) > 10:
                            continue
                    except (ValueError, TypeError):
                        continue
                
                question = data.get("question", "")
                history_prompt = data.get("history_prompt", "")
                full_history = question + history_prompt
                
                if full_history:
                    unrolled_data = parse_and_unroll_dialogue(full_history, format_to_alpaca_history)
                    all_processed_samples.extend(unrolled_data)

            except Exception as e:
                print(f"   - 处理文件 {filename} 时发生错误: {e}")

    print(f"   - ✅ 完成，找到 {len(all_processed_samples)} 条有效样本。")
    return all_processed_samples


if __name__ == '__main__':
    # --- 配置区域 ---
    BASE_PATH = 'result-v7_100/deepseek-reasoner_results' 
    FINAL_OUTPUT_PATH = os.path.join(BASE_PATH, 'all_students_data_filtered_combined.json')
    NUM_STUDENTS = 100

    # --- 主处理逻辑 ---
    all_data_combined = []

    print(f"🚀 开始批量处理 {NUM_STUDENTS} 个学生文件夹，从: {BASE_PATH}")
    print("-" * 50)

    for i in range(NUM_STUDENTS):
        current_student_folder = os.path.join(BASE_PATH, f'student{i}')
        
        if not os.path.isdir(current_student_folder):
            print(f"⚠️ 警告: 找不到目录 {current_student_folder}，已跳过。")
            continue

        data_from_folder = process_single_folder(current_student_folder)
        
        if data_from_folder:
            all_data_combined.extend(data_from_folder)

    print("-" * 50)
    print(f"💾 所有文件夹处理完毕，正在将数据写入最终文件...")

    with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_data_combined, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 全部处理完成！")
    print(f"   - 数据已全部保存到: {FINAL_OUTPUT_PATH}")
    print(f"   - 总共生成了 {len(all_data_combined)} 条训练样本。")