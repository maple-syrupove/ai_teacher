import requests
import json
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import tiktoken
from openai import OpenAI
import time
from typing import Dict, List, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
from llm_inf import LLMHandler_inf
import glob
import re
import threading

# ==================== vLLM 相关导入 ====================
from vllm import LLM, SamplingParams
from vllm.distributed import destroy_model_parallel
# =======================================================

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 全局变量，用于存储加载后的本地模型实例
local_llm_instance = None
# 用于确保模型只被加载一次的线程锁
llm_lock = threading.Lock()

logging.basicConfig(filename='main_gpt.log', level=logging.INFO, encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s')


def get_local_llm(local_model_path):
    """
    初始化并返回本地vLLM引擎的单例。
    通过线程锁确保模型只在首次调用时被加载一次，并优化了初始化参数。
    """
    global local_llm_instance
    if local_llm_instance is None:
        with llm_lock:
            if local_llm_instance is None:
                model_path = Path(__file__).parent / local_model_path
                logging.info(f"首次调用，正在从路径加载本地模型: {model_path}")
                print(f"首次调用，正在从路径加载本地模型: {model_path}")

                # ==================== 优化后的vLLM初始化 ====================
                local_llm_instance = LLM(
                    model=str(model_path),
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.95,  # 提高显存使用率以增加KV缓存
                    dtype="bfloat16"  # 若GPU支持（如A100/H100），可获性能提升。否则可改为 "half" 或 "auto"
                )
                # ============================================================
                
                logging.info("本地模型加载成功。")
                print("本地模型加载成功。")
    return local_llm_instance


def num_tokens_from_messages(messages, model="gpt-4.1"):
    """计算消息总 token 数（适配 OpenAI chat 格式）"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2
    return num_tokens


def run_all_dialogues_batched(student_output, error_output, all_question_map, model_name, save_path, local_model_path):
    """
    (核心优化函数)
    通过按轮次、批处理的模式运行所有对话，以最大化性能。
    """
    # 从环境变量中安全地获取 API 密钥
    api_key = os.getenv("OPENAI_API_KEY")
    student_model_name = "gpt-4.1"
    teacher_model_name = model_name
    referee_model_name = "gpt-4o"
    max_rounds = 20
    max_workers_openai = 10 # 用于OpenAI API请求的并发数

    # ==================== 1. 初始化所有对话的状态 ====================
    teacher_system_prompt = '''
Please act as a professional and dedicated teacher. You will engage in a dialogue with a student who is asking you questions. Your sole objective is to use Socratic questioning to precisely identify the student's error in solving the given problem.
Your task is only to identify the mistake, not to guide the student toward the correct answer.
Your constraints are as follows:
**Do not guide, hint, or ask the student to reply with "zzc." Do not directly ask for the student's complete incorrect problem-solving steps.
**Only when you are 100% certain of the student's mistake can you point it out with a clear and concise statement. After diagnosing and stating the error, wait for the student's final response. If the student does not reply with "zzc," try a different line of questioning or continue the dialogue, do not repeatedly confirm the same error or ask the same question.
**The prompt history contains past dialogues. Strictly avoid mimicking the student's tone or restating their answers for analysis. You must always maintain the demeanor and tone of a professional teacher.
**Your reply to student must not exceed 80 tokens. Please keep your questions and answers concise.
'''
    dialogue_states = []
    for student_id, profile in enumerate(student_output):
        for error_id, error in enumerate(error_output):
            question_index = error.get("question_index")
            question = all_question_map.get(question_index, {"question": "（未找到原始题目）"})
            
            student_system_prompt = f'''
You are a student who made a mistake in solving a problem, with the error detailed in 【Specific Error JSON】. You will engage in direct dialogue with the teacher, responding to their questions about your mistake.
【Specific Error JSON】:
{json.dumps(error, ensure_ascii=False, indent=4)}
Your constraints are as follows:
**The initial conversation can only begin with your wrong answer, without any additional explanation or reason.
**Do not reveal the entire incorrect solving process at once. Only provide the information necessary to answer the teacher's current question.
**If the teacher points out your mistake in a statement and this accusation exactly matches the reason in 【Specific Error JSON】, you must respond with the exact string "zzc". If the teacher states an error that does not match your actual mistake, you should deny it and wait for the teacher's next question. If the teacher asks you a question (i.e., a sentence ending with a question mark), you must answer it directly. Exception: If asked what your mistake is (e.g., "What was your error?"), reply "I don't know" and do not reveal the reason. Never respond with "zzc" when answering a question, regardless of whether you think it reveals your mistake.
**Your reply to teacher must not exceed 80 tokens. Please keep your questions and answers concise.
'''

            dialogue_states.append({
                "dialogue_id": student_id * 1000 + error_id,
                "is_active": True,
                "history_prompt": "",
                "round_used": 0,
                "question_obj": question,
                "profile_obj": profile,
                "error_obj": error,
                "student_system_prompt": student_system_prompt,
                "teacher_system_prompt": teacher_system_prompt,
                "teacher_token_counts": []
            })
    
    # ==================== 2. 获取模型和处理器实例 ====================
    llm = get_local_llm(local_model_path) if teacher_model_name == "qwen3-8b" else None
    student_handler = LLMHandler_inf(api_key, student_model_name)
    teacher_handler_api = LLMHandler_inf(api_key, teacher_model_name) if llm is None else None
    referee_handler = LLMHandler_inf(api_key, referee_model_name)

    # ==================== 3. 按轮次进行对话模拟 ====================
    for round_num in range(max_rounds):
        active_dialogues = [d for d in dialogue_states if d["is_active"]]
        if not active_dialogues:
            print("所有对话已完成，提前结束循环。")
            break
        
        print(f"\n===== 开始第 {round_num + 1} 轮对话 (剩余 {len(active_dialogues)} 个) =====")
        current_round = round_num + 1
        
        for d in active_dialogues:
            d["round_used"] = current_round

        # --- 学生轮 (并行I/O) ---
        with ThreadPoolExecutor(max_workers=max_workers_openai) as executor:
            future_to_dialogue = {}
            for d in active_dialogues:
                current_prompt = f"math question：{d['question_obj']['question']}\nmax rounds：{max_rounds}\nBelow is the dialogue history between the student and the teacher:\n{d['history_prompt']}"
                student_messages = [{"role": "system", "content": d['student_system_prompt']}, {"role": "user", "content": current_prompt}]
                future = executor.submit(student_handler.get_completion, student_messages, temperature=1.0)
                future_to_dialogue[future] = d
            
            for future in tqdm(as_completed(future_to_dialogue), total=len(active_dialogues), desc=f"第{current_round}轮: 学生回复进度"):
                d = future_to_dialogue[future]
                try:
                    student_response = future.result()
                    if student_response:
                        d["history_prompt"] += f"\n学生：{student_response}"
                        if "zzc" in student_response:
                            d["is_active"] = False
                    else:
                        logging.warning(f"学生API调用返回空 for dialogue {d['dialogue_id']}")
                        d["is_active"] = False # API调用失败，终止该对话
                except Exception as e:
                    logging.error(f"学生API调用失败 for dialogue {d['dialogue_id']}: {e}")
                    d["is_active"] = False

        # --- 教师轮 (批处理计算或并行I/O) ---
        dialogues_for_teacher = [d for d in active_dialogues if d["is_active"]]
        if not dialogues_for_teacher:
            continue

        if llm: # 使用本地vLLM进行批处理
            prompts_for_vllm = []
            tokenizer = llm.get_tokenizer()
            for d in dialogues_for_teacher:
                current_prompt = f"math question：{d['question_obj']['question']}\nmax rounds：{max_rounds}\nBelow is the dialogue history between the student and the teacher:\n{d['history_prompt']}"
                teacher_messages = [{"role": "system", "content": d['teacher_system_prompt']}, {"role": "user", "content": current_prompt}]
                formatted_prompt = tokenizer.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True)
                prompts_for_vllm.append(formatted_prompt)

            if prompts_for_vllm:
                sampling_params = SamplingParams(temperature=1.0, max_tokens=128)
                print(f"向vLLM提交 {len(prompts_for_vllm)} 个请求的批处理任务...")
                teacher_outputs = llm.generate(prompts_for_vllm, sampling_params)
                print("vLLM批处理完成。")
                
                for i, d in enumerate(dialogues_for_teacher):
                    response_text = teacher_outputs[i].outputs[0].text.strip()
                    d["history_prompt"] += f"\n教师：{response_text}"
                    d["teacher_token_counts"].append(len(tokenizer.encode(response_text)))
        
        elif teacher_handler_api: # 使用远程API
            with ThreadPoolExecutor(max_workers=max_workers_openai) as executor:
                future_to_dialogue = {}
                for d in dialogues_for_teacher:
                    current_prompt = f"math question：{d['question_obj']['question']}\nmax rounds：{max_rounds}\nBelow is the dialogue history between the student and the teacher:\n{d['history_prompt']}"
                    teacher_messages = [{"role": "system", "content": d['teacher_system_prompt']}, {"role": "user", "content": current_prompt}]
                    future = executor.submit(teacher_handler_api.get_completion, teacher_messages, temperature=1.0)
                    future_to_dialogue[future] = d
                
                # 获取tiktoken编码器用于计算token数
                try:
                    encoding = tiktoken.encoding_for_model(teacher_model_name)
                except KeyError:
                    encoding = tiktoken.get_encoding("cl100k_base")

                for future in tqdm(as_completed(future_to_dialogue), total=len(dialogues_for_teacher), desc=f"第{current_round}轮: 教师(API)回复进度"):
                    d = future_to_dialogue[future]
                    try:
                        teacher_response = future.result()
                        if teacher_response:
                            d["history_prompt"] += f"\n教师：{teacher_response}"
                            d["teacher_token_counts"].append(len(encoding.encode(teacher_response)))
                        else:
                            logging.warning(f"教师API调用返回空 for dialogue {d['dialogue_id']}")
                            # 教师回复失败不终止对话，继续下一轮
                    except Exception as e:
                        logging.error(f"教师API调用失败 for dialogue {d['dialogue_id']}: {e}")

    # ==================== 4. 结束后统一处理（裁判 & 保存）====================
    print("\n所有对话模拟完成，开始进行裁判评估和保存...")
    def get_referee_judgment_and_save(d):
        # 定义裁判prompt
        referee_system_prompt = f'''
You are a professional evaluator of Socratic-style dialogues. Your task is to assess the teacher’s performance based on the dialogue history and the student's actual error, provided below.
【Student's True Error JSON】:
{json.dumps(d['error_obj'], ensure_ascii=False, indent=4)}
Please evaluate the following three dimensions:
1.  **Accuracy**: Did the teacher correctly identify the student's error as detailed in the JSON above?
2.  **Socratic Method**: Did the teacher guide the student through effective questioning, without giving direct answers or hints?
3.  **Adherence to Rules**: Did the teacher fully comply with all constraints : {d['teacher_system_prompt']} 
Use the following rating scale:
- 2: Excellent (The goal was fully achieved)
- 1: Mediocre (Partially achieved, or with some issues)
- 0: Non-compliant (The goal was not met)
Your output MUST strictly follow the format below, providing a numerical score for EACH of the three dimensions.
Accuracy: [Did the student's final response contain "zzc" .If NO, Accuracy = 0 else Score 0, 1, or 2]
Socratic Method: [Score 0, 1, or 2]
Adherence to Rules: [Score 0, 1, or 2]
'''
        # 获取裁判结果
        referee_messages = [{"role": "system", "content": referee_system_prompt}, {"role": "user", "content": f"Dialogue History:\n{d['history_prompt']}"}]
        judgment = referee_handler.get_completion(referee_messages, temperature=0.5) or "Referee call failed or was not executed."
        
        # 准备最终输出数据
        output_data = {
            "dialogue_id": d["dialogue_id"],
            "question": d["question_obj"]['question'],
            "correct_answer": all_question_map.get(d["error_obj"].get("question_index"), {}).get("answer"),
            "question_index": d["error_obj"].get("question_index"),
            "student": d["profile_obj"],
            "error": d["error_obj"],
            "student_system_prompt": d["student_system_prompt"],
            "teacher_system_prompt": d["teacher_system_prompt"],
            "history_prompt": d["history_prompt"],
            "referee_judgment": judgment,
            "rounds_used": d["round_used"],
            "rounds_max": max_rounds,
            "end_time": datetime.now().isoformat(),
            "teacher_token_total": sum(d["teacher_token_counts"]),
            "teacher_token_avg": sum(d["teacher_token_counts"]) / len(d["teacher_token_counts"]) if d["teacher_token_counts"] else 0,
            "cost": d["teacher_token_counts"]
        }
        
        # 保存JSON文件
        dialogue_dir = Path(__file__).parent / save_path
        dialogue_dir.mkdir(exist_ok=True)
        out_path = dialogue_dir / f"dialogue_{d['dialogue_id']}_result.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        return f"Dialogue {d['dialogue_id']} processed and saved."

    with ThreadPoolExecutor(max_workers=max_workers_openai) as executor:
        futures = [executor.submit(get_referee_judgment_and_save, d) for d in dialogue_states]
        for future in tqdm(as_completed(futures), total=len(dialogue_states), desc="最终评估与保存进度"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"最终处理环节失败: {e}")

    print("所有结果已保存。")


def load_questions_map(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    return {q["index"]: q for q in all_questions}


def calculate_average_scores(results_directory, avg_rounds):
    accuracy_scores = []
    socratic_scores = []
    rules_scores = []
    
    if not os.path.exists(results_directory):
        print(f"错误：目录 '{results_directory}' 不存在。")
        return

    file_count = 0
    for filename in os.listdir(results_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(results_directory, filename)
            file_count += 1
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    judgment_text = data.get("referee_judgment", "")

                    if not judgment_text:
                        print(f"警告：文件 {filename} 中缺少 'referee_judgment' 内容。")
                        continue

                    acc_match = re.search(r"Accuracy\s*[:：]?\s*(\d)", judgment_text, re.IGNORECASE)
                    soc_match = re.search(r"Socratic Method\s*[:：]?\s*(\d)", judgment_text, re.IGNORECASE)
                    rul_match = re.search(r"Adherence to Rules\s*[:：]?\s*(\d)", judgment_text, re.IGNORECASE)

                    if acc_match: accuracy_scores.append(int(acc_match.group(1)))
                    if soc_match: socratic_scores.append(int(soc_match.group(1)))
                    if rul_match: rules_scores.append(int(rul_match.group(1)))

                except json.JSONDecodeError:
                    print(f"警告：文件 {filename} 不是有效的JSON格式。")
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {e}")

    print(f"\n--- 目录 '{results_directory}' 的评估报告 ---")
    print(f"共分析了 {file_count} 个对话文件。")

    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    avg_socratic = sum(socratic_scores) / len(socratic_scores) if socratic_scores else 0
    avg_rules = sum(rules_scores) / len(rules_scores) if rules_scores else 0

    print(f"平均准确度 (Accuracy): {avg_accuracy:.2f}")
    print(f"平均苏格拉底方法 (Socratic Method): {avg_socratic:.2f}")
    print(f"平均规则遵守度 (Adherence to Rules): {avg_rules:.2f}")
    
    if avg_rounds is not None:
        final_score = (3 - (avg_rules + avg_socratic) / 2) * avg_rounds
        print(f"最终评分轮数是： {final_score:.2f}")
    
    print("------------------------------------------")


if __name__ == "__main__":
    import check 

    with open("student_output8.json", "r", encoding="utf-8") as f:
        student_output = json.load(f)
    with open("a_errors_10.json", "r", encoding="utf-8") as f:
        error_output = json.load(f)
    question_map = load_questions_map("a_new_questions_10.json")

    local_model_path = "LLaMA-Factory/qwen3-8b"
    
    modellist = [
        # "qwen3-8b",
        "gpt-4.1",
        # "gpt-4o",
    ]
    
    dirlist = [
        str(Path("test") / model_name) for model_name in modellist
    ]
    
    try:
        for model_name, dir_name in zip(modellist, dirlist):
            print(f"===== 开始处理模型 {model_name} =====")
            Path(dir_name).mkdir(parents=True, exist_ok=True)
            print(f"结果将保存到: {dir_name}")
            
            # 预加载模型，如果需要的话
            if model_name == "qwen3-8b":
                get_local_llm(local_model_path)
            
            # ==================== 调用优化后的批处理函数 ====================
            run_all_dialogues_batched(student_output, error_output, question_map, model_name, dir_name, local_model_path)
            # =============================================================
            
            avg_rounds = check.start(dir_name, dir_name, num_students=1, num_errors=10) 
            calculate_average_scores(dir_name, avg_rounds)
            print(f"===== 模型 {model_name} 处理完毕 =====")
            
    finally:
        # ==================== 正确、稳定地销毁vLLM资源 ====================
        if local_llm_instance is not None:
            print("\n正在销毁vLLM引擎，释放资源...")
            destroy_model_parallel()
            local_llm_instance = None
            print("vLLM引擎已销毁。")
        # ================================================================