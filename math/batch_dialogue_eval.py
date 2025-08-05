import os
import json
from pathlib import Path
from math import sqrt
from statistics import mean, variance
from main import run_one_dialogue, load_questions_map
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
STUDENT_FILE = "student_output.json"
ERROR_DIR = "error_math_new"
QUESTION_FILE = "new_questions.json"
OUT_BASE = "batch_dialogue_eval"
N_PERSONA = 5
N_ERROR_FILE = 50
N_ERROR_PER_FILE = 6
N_THREADS = 30
LOG_FILE = "batch_dialogue_eval.log"

# 日志配置
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# 读取5个人格
with open(STUDENT_FILE, "r", encoding="utf-8") as f:
    personas = json.load(f)
    personas = personas[:N_PERSONA]

# 读取题目映射
question_map = load_questions_map(QUESTION_FILE)

# 收集所有error文件
error_files = sorted([f for f in Path(ERROR_DIR).glob("*.json")])[:N_ERROR_FILE]

# 统计数据结构
results = {}  # {persona_name: {error_index: [rounds_used, ...]}}

def dialogue_task(persona_idx, persona, file_idx, error_file, error_idx, error, question_map):
    persona_name = f"persona_{persona_idx+1}"
    # 获取题目index
    question_index = error.get("question_index")
    question = question_map.get(question_index)
    dialogue_id = f"{file_idx+1}_{error_idx+1}_{persona_idx+1}"
    out_dir = Path(OUT_BASE) / persona_name / f"error_{error_idx+1}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dialogue_{file_idx+1}.json"
    try:
        # 捕获run_one_dialogue的日志输出
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        rounds = run_one_dialogue(question, persona, error, dialogue_id, question_map)
        sys.stdout = old_stdout
        log_output = mystdout.getvalue()
        logging.info(f"[Persona {persona_name}][Error {error_idx+1}][File {file_idx+1}] 模型输出:\n{log_output}")
        # 读取刚刚保存的对话json，获取rounds_used
        with open(out_path, "r", encoding="utf-8") as f:
            dialogue_data = json.load(f)
        rounds_used = dialogue_data.get("rounds_used", 0)
        return (persona_name, error_idx, rounds_used)
    except Exception as e:
        logging.error(f"Error: {e} at persona {persona_name}, file {error_file}, error_idx {error_idx}")
        return (persona_name, error_idx, 0)

# 任务列表
tasks = []
for persona_idx, persona in enumerate(personas):
    persona_name = f"persona_{persona_idx+1}"
    results[persona_name] = {i: [] for i in range(N_ERROR_PER_FILE)}
    for file_idx, error_file in enumerate(error_files):
        with open(error_file, "r", encoding="utf-8") as f:
            error_list = json.load(f)
        for error_idx, error in enumerate(error_list):
            if error_idx >= N_ERROR_PER_FILE:
                break
            tasks.append((persona_idx, persona, file_idx, error_file, error_idx, error, question_map))

# 多线程执行
total_tasks = len(tasks)
with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    future_to_task = {executor.submit(dialogue_task, *task): task for task in tasks}
    for i, future in enumerate(as_completed(future_to_task)):
        persona_name, error_idx, rounds_used = future.result()
        results[persona_name][error_idx].append(rounds_used)
        if (i+1) % 10 == 0 or (i+1) == total_tasks:
            print(f"已完成 {i+1}/{total_tasks} 轮对话")

# 统计均值、方差
stat = {}
for persona_name, error_dict in results.items():
    stat[persona_name] = {}
    for error_idx, rounds_list in error_dict.items():
        if rounds_list:
            m = mean(rounds_list)
            v = variance(rounds_list) if len(rounds_list) > 1 else 0
        else:
            m, v = 0, 0
        stat[persona_name][f"error_{error_idx+1}"] = {"mean": m, "var": v}

# 组间方差
all_means = []
for persona_name in stat:
    for error_key in stat[persona_name]:
        all_means.append(stat[persona_name][error_key]["mean"])
if len(all_means) > 1:
    between_var = variance(all_means)
else:
    between_var = 0

# 输出统计结果
with open(Path(OUT_BASE) / "stat_summary.json", "w", encoding="utf-8") as f:
    json.dump({"stat": stat, "between_var": between_var}, f, ensure_ascii=False, indent=2)

print("统计完成，结果已保存到", Path(OUT_BASE) / "stat_summary.json") 