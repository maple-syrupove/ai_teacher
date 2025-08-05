import os
import json
import re
import numpy as np

# 文件夹路径
dialogue_dir = 'result-v5.1\gpt-4o_results'
output_file = 'result-v5.1\gpt-4o_results\\a_out.txt'

num_students = 1  # 学生数量
num_errors = 10  # 问题数量

# 初始化数据结构
student_rounds = [[0 for j in range(num_errors)] for i in range(num_students)]
teacher_token_avgs = [[0 for j in range(num_errors)] for i in range(num_students)]

# 遍历所有文件
for fname in os.listdir(dialogue_dir):
    match = re.match(r'dialogue_(\d+)_(\d+)_result\.json', fname)
    if not match:
        # 兼容旧命名方式 dialogue_{dialogue_id}_result.json
        match2 = re.match(r'dialogue_(\d+)_result\.json', fname)
        if not match2:
            continue
        dialogue_id = int(match2.group(1))
        student_id = dialogue_id // 1000
        error_id = dialogue_id % 1000
    else:
        student_id = int(match.group(1))
        error_id = int(match.group(2))
    if student_id >= num_students or error_id >= num_errors:
        continue
    # 读取文件
    with open(os.path.join(dialogue_dir, fname), 'r', encoding='utf-8') as f:
        data = json.load(f)
    rounds_used = data.get('rounds_used', None)
    teacher_token_avg = data.get('teacher_token_avg', None)
    talk = data.get('history_prompt', None)
    if rounds_used is not None:
        student_rounds[student_id][error_id] = rounds_used
        # if rounds_used < 4:
        #     print(student_id * 1000 + error_id)
    if teacher_token_avg is not None:
        teacher_token_avgs[student_id][error_id] = teacher_token_avg
    if talk is None or 'zzc' not in talk:
        print(student_id * 1000 + error_id)

# 打开输出文件
with open(output_file, 'w', encoding='utf-8') as out_f:
    # 统计并输出
    for student_id in range(num_students):
        rounds = student_rounds[student_id]
        tokens = teacher_token_avgs[student_id]
        if len(rounds) == 0:
            output_line = f"学生{student_id}: 无数据"
        else:
            avg = np.mean(rounds)
            var = np.var(rounds)
            avg_tokens = np.mean(tokens)
            # 格式1：索引:值
            formatted_rounds = [f"{i + student_id * 1000:4d}:|{v:2d}|" for i, v in enumerate(rounds)]
            output_line = f"学生{student_id}: 平均对话轮数 = {avg:.2f}, 方差 = {var:4.2f}, 轮数列表 = {formatted_rounds}"
            output_tokens = f"学生{student_id}: 单论对话(一次问答)平均tokens = {avg_tokens:.2f}, 单个json平均tokens = {avg * avg_tokens:.2f}"

        # 输出到控制台和文件
        print(output_line)
        print(output_tokens)
        out_f.write(output_tokens + '\n')
        out_f.write(output_line + '\n')

print(f"结果已保存到 {output_file}")