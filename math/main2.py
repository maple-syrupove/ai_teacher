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
import tiktoken
import time
from typing import Dict, List, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os
from llm_inf import LLMHandler_inf
import glob

logging.basicConfig(filename='main_testt.log', level=logging.DEBUG, encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s')


def num_tokens_from_messages(messages, model="gpt-4.1"):
    """计算消息总 token 数（适配 OpenAI chat 格式）"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # 以下是估算规则（基于 OpenAI 官方描述）
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # 每条 message 包括 role 和 metadata
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2  # 包括 assistant 回复开头
    return num_tokens

def call_api(system_prompt, prompt, api_key, model, max_tokens=2048, max_retries=10):
    url = "https://www.dmxapi.com/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'User-Agent': 'DMXAPI/1.0.0 (https://www.dmxapi.com)',
        'Content-Type': 'application/json'
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    is_qwen = "qwen" in model.lower()

    input_tokens = 0
    if not is_qwen:
        try:
            input_tokens = num_tokens_from_messages(messages, model)
            logging.info(f"🔢 输入 Token 数: {input_tokens}")
        except Exception as e:
            logging.warning(f"输入 Token 计算失败: {e}")

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 1.0
    })

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=240)
            response.raise_for_status()
            response_json = response.json()

            if 'choices' in response_json and response_json['choices']:
                content = response_json['choices'][0]['message']['content']
                logging.info("模型返回结果：")
                logging.info(content)

                output_tokens = 0
                if not is_qwen:
                    try:
                        output_tokens = len(tiktoken.encoding_for_model(model).encode(content))
                        logging.info(f"🔢 输出 Token 数: {output_tokens}")
                        logging.info(f"🧮 总 Token 数: {input_tokens + output_tokens}")
                    except Exception as e:
                        logging.warning(f"输出 Token 计算失败: {e}")

                return {
                    "content": content,
                    "input_tokens": input_tokens if not is_qwen else 0,
                    "output_tokens": output_tokens if not is_qwen else 0
                }
            else:
                logging.warning("API 响应中未包含 'choices'")

        except requests.exceptions.RequestException as e:
            logging.error(f"API请求异常 (尝试 {attempt + 1}/{max_retries}): {e}\n请求payload: {payload}")
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"响应解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)

    return {
        "content": None,
        "input_tokens": 0 if is_qwen else input_tokens,
        "output_tokens": 0
    }

def process_question(q, system_prompt, error_dir, api_key, model):
    index = q.get('index', 0)
    user_prompt = f"""
    下面的【数学问题】生成相应的学生错误JSON数据:
    {json.dumps(q, ensure_ascii=False, indent=2)}
    """
    handler = LLMHandler_inf(api_key, model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    result_content = handler.get_completion(messages, temperature=1.0, max_tokens=9999)
    result = {"content": result_content, "input_tokens": 0, "output_tokens": 0}  # token统计可选
    if result["content"]:
        try:
            result_cleaned = result["content"].strip()
            if result_cleaned.startswith("```json"):
                result_cleaned = result_cleaned[7:].lstrip()
            if result_cleaned.endswith("```"):
                result_cleaned = result_cleaned[:-3].rstrip()
            output_path = error_dir / f"question_{index}_error.json"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_cleaned)
            return f"✅ 写入成功：question_{index}_error.json"
        except Exception as e:
            return f"❌ 写入失败：question_{index}_error.json\n错误：{e}\n输出：\n{result}"
    else:
        return f"❌ API返回为空：index={index}"

def generate_error():
    test_path = Path(__file__).parent / 'example_questions_100.json'
    with open(test_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)

    error_dir = Path(__file__).parent / 'example_error_math_100'
    error_dir.mkdir(exist_ok=True)


    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"  
    model = "gpt-4.1"
    system_prompt = """
    **角色:** 你是一位精通数学教育和学生学习心理学的专家。你深刻理解学生在解决数学问题时可能犯的各种错误，并能准确地将这些错误与特定的认知维度（如知识性、逻辑性、策略性、执行性）联系起来。

    **任务:** 我将为你提供一个【错误维度表格】和一个具体的【数学问题】。请你严格遵循以下要求，为该问题**固定生成六种不同类型的学生错误**JSON数据。这六种错误必须严格按照**四种单维度错误**和**两种多维度复合错误**的结构进行组织。

    **核心要求:**

    1.  **固定的错误结构:** 你必须不多不少，正好生成六个错误JSON对象。其结构必须如下：
        * 前四个错误对象必须是**单维度错误**，依次分别对应【知识性】、【逻辑性】、【策略性】和【执行性】这四个维度。每个错误都应是该维度的典型、纯粹的体现。
        * 后两个错误对象必须是**多维度（深度复杂）错误**，每个错误必须由至少两个维度共同导致，用以模拟真实世界中学生错误成因的复杂性。

    2.  **模拟真实性:** 生成的每个错误都必须高度模拟真实学生可能出现的思维和操作过程。原因解释要合理、有说服力，不能生搬硬套表格中的定义。

    3.  **多维度归因:** 对于那两个**多维度复合错误**，你的分析必须清晰地体现出多个“错误维度”是如何共同作用，导致最终的错误表现的。

    4.  **精确到点:** "具体错误点"必须明确指出在解决这个特定【数学问题】时，学生在哪一步、哪个公式、哪个数字上出错了。

    5.  **深度分析:** "错误维度导致错误点出现原因"是核心，需要详细、清晰地解释为什么学生会因为这个/这些维度上的缺陷，最终导致了那个具体的错误。

    6.  **重点标注:** 在生成的**六个**错误中，请根据你的专业判断，选择出**一个**真实学生最容易犯的错误，并将其`main`字段设置为1。其余五个的`main`字段必须为0。

    7.  **严格的JSON格式:** 输出必须是一个包含**六个**对象的、标准的、无注释的JSON数组。不要添加新的key。"index"为从0开始的数值编号。

    8.  **保证错误:** "具体错误点"中必须明确指出学生得到的错误答案或者是无法给出具体答案，错误答案必须与正确答案不同。

    ---

    **【错误维度表格】**

    | 错误维度 (Error Dimension) | 核心定义 (Core Definition)                                 | 典型表现 (Typical Manifestations)                        |
    | -------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
    | **知识性错误 (Knowledge-Based)** | 因数学知识本身的缺陷、遗忘或理解不深刻导致的错误。         | 概念混淆、性质误记、公式错用、前提条件忽视。             |
    | **逻辑性错误 (Logical)** | 因违背数学的严谨性、推理规则和逻辑形式而产生的错误。         | 循环论证、分类讨论不完备、偷换概念、不等价变换。         |
    | **策略性错误 (Strategic)** | 因未能选择或执行有效的解题路径和思想方法而导致的错误。     | 模式识别失败、缺乏整体观、思维僵化、无法转化问题。       |
    | **执行性错误 (Execution)** | 在解题的具体操作环节出现的失误，常被笼统地称为"粗心"。     | 审题不清、抄写错误、计算失误、书写不规范。               |

    ---

    **【JSON 输出格式】**

    ```json
    [
        {
            "错误维度": ["知识性"],
            "典型表现": ["表现A"],
            "具体错误点": "...",
            "错误维度导致错误点出现原因": "...",
            "main": 0,
            "index": 0
        },
        {
            "错误维度": ["逻辑性"],
            "典型表现": ["表现B"],
            "具体错误点": "...",
            "错误维度导致错误点出现原因": "...",
            "main": 0,
            "index": 1
        },
        {
            "错误维度": ["策略性"],
            "典型表现": ["表现C"],
            "具体错误点": "...",
            "错误维度导致错误点出现原因": "...",
            "main": 1,
            "index": 2
        },
        {
            "错误维度": ["执行性"],
            "典型表现": ["表现D"],
            "具体错误点": "...",
            "错误维度导致错误点出现原因": "...",
            "main": 0,
            "index": 3
        },
        {
            "错误维度": ["维度1", "维度2"],
            "典型表现": ["表现E", "表现F"],
            "具体错误点": "...",
            "错误维度导致错误点出现原因": "...",
            "main": 0,
            "index": 4
        },
        {
            "错误维度": ["维度3", "维度4"],
            "典型表现": ["表现G", "表现H"],
            "具体错误点": "...",
            "错误维度导致错误点出现原因": "...",
            "main": 0,
            "index": 5
        }
    ]
    """
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        future_to_question = {
            executor.submit(process_question, q, system_prompt, error_dir, api_key, model): q
            for q in test_questions
        }
        for future in tqdm(as_completed(future_to_question), total=len(test_questions), desc="生成错误数据中"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                results.append(f"❌ 线程异常：{exc}")

def generate_student():
    GENERATION_COUNT = 6 
    prompt_text = """
    角色: 你是一位资深的教育心理学专家和AI提示工程师。

    背景: 我正在进行一项教育研究项目，需要构建一系列高度逼真的虚拟学生模型。这些模型将用于模拟学生在解答问题（尤其是数理逻辑问题）时的真实互动场景。我已经通过课堂观察，总结出了学生在语言表达和解题心态上的几个核心维度类型。

    任务: 请你根据我提供的【学生表达维度】和【学生情感维度】的表格，将这两者进行有逻辑的组合，生成6个具有鲜明个性特征的学生画像。

    核心要求:
    1.  **严格的输出格式**: 最终结果必须是一个完整的JSON对象数组（a list of JSON objects）。每个JSON对象代表一个学生，且不包含任何额外的解释性文本或注释。
    2.  **字段精确匹配**: 每个JSON对象必须包含以下三个字段：
        * "学生表达维度": 从我提供的"学生表达维度"中选取的类型。
        * "学生情感维度": 从我提供的"学生情感维度"中选取的类型。
        * "学生描述": 一段具体、生动的学生刻画描述。这段描述需要深度融合所选的表达与情感类型，描绘出该学生在**解答一个具体问题时（不给出具体例子需要你总结一个通用的心理过程）**的完整思考过程、语言风格和心理状态。描述需包含丰富的细节，展现其内在的逻辑断点或情感挣扎。

    输入数据:

    **表1: 学生表达维度 (Student Expression Dimensions)**
    | 表达类型 (Type) | 核心特征 (Core Feature) |
    | :--- | :--- |
    | **语言模糊与指代不清** | 使用"这个"、"那个"等模糊词汇，解题步骤难以追踪。例如："先把那个...括号外面的2弄进去...就变成2x...然后那个6...等于10。" |
    | **跳跃式陈述** | 省略关键的中间思维步骤，直接跳到结论。例如："就是...2x等于4，所以x等于2。"（未解释如何从 `2x+6=10` 到 `2x=4`） |
    | **自我修正与不确定性** | 表达犹豫、自我怀疑和修正。例如："先把2乘进去，得到2x加...加6？对，是加6。然后...把6移到右边？好像是...减6？所以是10减6，得4。" |
    | **"黑话"与非正式语言** | 使用非标准术语。例如："老师教的是'移项变号'，所以把6'扔'过去就变成减6了。" |
    | **冗余与重复** | 反复陈述同一个步骤或想法。例如："就是2乘以x，然后2再乘以3。对，就是2乘以x，然后2乘以3。然后加起来。" |
    | **具象化描述** | 依赖于具体事物或动作来描述抽象过程。例如："我有2个苹果，又拿来3个苹果，我数一下...是5个。"（用此风格描述代数） |

    **表2: 学生情感/心态维度 (Student Emotional/Mental Dimensions)**
    | 情感类型 (Type) | 核心特征 (Core Feature) |
    | :--- | :--- |
    | **过度自信的"专家"** | 认为自己完全掌握，但实则存在概念错误。表达流利自信，不容置疑。 |
    | **焦虑不安的"小白"** | 缺乏信心，害怕犯错。表达时充满"可能"、"也许"、"我不知道对不对"等不确定词汇。 |
    | **"我忘了"的甩手掌柜** | 以"忘了"、"老师就是这么教的"来回避对原理的深层解释。 |
    | **固执的"一条路走到黑"** | 坚信自己的（错误）方法是唯一解法，即使遇到困难也不愿尝试其他路径。 |
    | **寻求快速答案的"功利者"** | 对过程不感兴趣，只想知道最终答案和考试考法。 |
    **【JSON 输出格式】**
    ```json
    {
    "1": {
        "学生表达维度": ["维度1", "维度2"],
        "学生情感维度": ["维度A", "维度B"],
        "学生描述": "一段具体、生动的学生刻画描述。",    },
    "2": {
        "学生表达维度": ["维度3"],
        "学生情感维度": ["维度C"],
        "学生描述": "一段具体、生动的学生刻画描述。",
    },
    ...
    }
    """
    user_prompt = """ """
    api_key = "sk-fKhKDrWY5AxX5FPeZO95LPieEqgt6yL6IpdlBZFI0AEOx2Cd"  
    model = "gemini-2.5-pro-exp-03-25"

    result = call_api(prompt_text, user_prompt, api_key=api_key, model=model, max_tokens=9999)

    if result:
        logging.info("开始将模型输出写入 TXT 文件。")

        try:
            result_cleaned = result.strip()
            if result_cleaned.startswith("```json"):
                result_cleaned = result_cleaned[7:].lstrip()
            if result_cleaned.endswith("```"):
                result_cleaned = result_cleaned[:-3].rstrip()

            output_path = Path("student_output.txt")
            with output_path.open("w", encoding="utf-8") as f:
                f.write(result_cleaned)

            logging.info(f"写入成功，文件路径: {output_path.resolve()}")
        except Exception as e:
            logging.error(f"写入 TXT 文件失败：{e}")
            logging.error(f"原始模型输出：\n{result}")
    else:
        logging.error("API 返回为空，未能生成错误数据。")

def run_one_dialogue(question, profile, error, dialogue_id, all_question_map, model_name, save_path):
    rounds_max = 20 
    
    #dmxapi
    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"
    model1 = "gpt-4.1"  # 学生
    model2 = model_name  # 教师
    ''' 
    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"
    model1 = "qwen3-8b"
    model2 = "qwen3-8b"
    '''
    total_input_tokens = 0
    total_output_tokens = 0
    teacher_system_prompt = '''
Please act as a professional and dedicated teacher. You will engage in a dialogue with a student who is asking you questions. Your sole objective is to use Socratic questioning to precisely identify the student's error in solving the given problem.
Your task is only to identify the mistake, not to guide the student toward the correct answer.
Your constraints are as follows:
**Do not guide, hint, or ask the student to reply with "zzc." Do not directly ask for the student's complete incorrect problem-solving steps.
**Only when you are 100% certain of the student's mistake can you point it out with a clear and concise statement. After diagnosing and stating the error, wait for the student's final response. If the student does not reply with "zzc," try a different line of questioning or continue the dialogue, do not repeatedly confirm the same error or ask the same question.
**The prompt history contains past dialogues. Strictly avoid mimicking the student's tone or restating their answers for analysis. You must always maintain the demeanor and tone of a professional teacher.
**Your reply to student must not exceed 80 tokens. Please keep your questions and answers concise.
'''
    student_system_prompt =f'''
You are a student who made a mistake in solving a problem, with the error detailed in 【Specific Error JSON】. You will engage in direct dialogue with the teacher, responding to their questions about your mistake.
【Specific Error JSON】:
{json.dumps(error, ensure_ascii=False, indent=4)}
Your constraints are as follows:
**Your initial response must begin with your incorrect answer.
**Do not reveal the entire incorrect solving process at once. Only provide the information necessary to answer the teacher's current question.
**If the teacher points out your mistake in a statement and this accusation exactly matches the reason in 【Specific Error JSON】, you must respond with the exact string "zzc". If the teacher states an error that does not match your actual mistake, you should deny it and wait for the teacher's next question. If the teacher asks you a question (i.e., a sentence ending with a question mark), you must answer it directly. Never respond with "zzc" when answering a question, regardless of whether you think it reveals your mistake.
**Your reply to teacher must not exceed 80 tokens. Please keep your questions and answers concise.
'''
    handler1 = LLMHandler_inf(api_key, model1)
    handler2 = LLMHandler_inf(api_key, model2)
    prompt_history = ""
    round_used = 0
    teacher_token_counts = []
    for round_num in range(rounds_max):
        log_prefix = f"[dialogue_id={dialogue_id}][round={round_num+1}]"
        logging.info(f"{log_prefix} === 第 {round_num + 1} 轮对话 ===")
        round_used = round_num + 1
        current_prompt = f"""
        math question：{question['question']}
        max rounds：{rounds_max}
        Below is the dialogue history between the student and the teacher:
        {prompt_history}
        """
        student_messages = [
            {"role": "system", "content": student_system_prompt},
            {"role": "user", "content": current_prompt}
        ]
        student_result_content = handler1.get_completion(student_messages, temperature=1.0)
        student_result = {"content": student_result_content, "input_tokens": 0, "output_tokens": 0}
        if student_result["content"]:
            student_response = student_result["content"]
            logging.info(f"{log_prefix} 学生：{student_response}")
            logging.debug(f"{log_prefix} 学生原始API返回：{student_result}")
            prompt_history += f"\n学生：{student_response}"
            if "zzc" in student_response:
                logging.info(f"{log_prefix} 学生终止对话")
                break
            teacher_messages = [
                {"role": "system", "content": teacher_system_prompt},
                {"role": "user", "content": current_prompt + f"\n学生：{student_response}"}
            ]
            
            teacher_result_content = handler2.get_completion(teacher_messages, temperature=1.0)
            teacher_result = {"content": teacher_result_content, "input_tokens": 0, "output_tokens": 0}
            if teacher_result["content"]:
                teacher_response = teacher_result["content"]
                # 统计教师回复token数
                try:
                    import tiktoken
                    try:
                        encoding = tiktoken.encoding_for_model("gpt-4.1")
                    except KeyError:
                        encoding = tiktoken.get_encoding("cl100k_base")
                    teacher_tokens = len(encoding.encode(teacher_response))
                except Exception as e:
                    teacher_tokens = 0
                    logging.warning(f"Token计算失败: {e}")
                teacher_token_counts.append(teacher_tokens)
                prompt_history += f"\n教师：{teacher_response}"
            else:
                logging.error(f"{log_prefix} 教师API调用失败，返回内容为空。teacher_result={teacher_result}")
        else:
            logging.error(f"{log_prefix} 学生API调用失败，对话终止，dialogue_id={dialogue_id}")
            logging.debug(f"{log_prefix} 学生API调用失败返回：{student_result}")
            break

    # 获取正确答案和题目index
    correct_answer = None
    question_index = None
    if all_question_map is not None and hasattr(error, 'get'):
        question_index = error.get("question_index")
        qobj = all_question_map.get(question_index)
        if qobj:
            correct_answer = qobj.get("answer")

    teacher_token_total = sum(teacher_token_counts)
    teacher_token_avg = (teacher_token_total / len(teacher_token_counts)) if teacher_token_counts else 0
    cost = teacher_token_counts

    output_data = {
        "dialogue_id": dialogue_id,
        "question": question['question'],
        "correct_answer": correct_answer,
        "question_index": question_index,
        "student": profile,
        "error": error,
        "student_system_prompt": student_system_prompt,
        "teacher_system_prompt": teacher_system_prompt,
        "history_prompt": prompt_history,
        "rounds_used": round_used,
        "rounds_max": rounds_max,
        "end_time": datetime.now().isoformat(),
        "teacher_token_total": teacher_token_total,
        "teacher_token_avg": teacher_token_avg,
        "cost": cost
    }

    # 输出到新目录 dialogue，每个对话单独为一个文件
    dialogue_dir = Path(__file__).parent / save_path
    dialogue_dir.mkdir(exist_ok=True)
    out_path = dialogue_dir / f"dialogue_{dialogue_id}_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logging.info(f"[dialogue_id={dialogue_id}] 第 {dialogue_id} 个对话输出完毕，已保存到 {out_path}")
    return dialogue_id

def load_questions_map(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    return {q["index"]: q for q in all_questions}

def run_dialogue_multithread(student_output, error_output, all_question_map): 
    dialogue_id = 0
    num_students = len(student_output)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:  
        futures = []

        for i, error in enumerate(error_output):
            profile = student_output[i % num_students]
            question_index = error.get("question_index")

            # 从题目映射中取出题干
            question = all_question_map.get(question_index, "（未找到原始题目）")
            dialogue_id += 1

            futures.append(executor.submit(
                run_one_dialogue, question, profile, error, dialogue_id, all_question_map
            ))

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="对话模拟进度"):
            result_id = f.result()
            tqdm.write(f"对话 {result_id} 已完成")

def rate_one_file(file, handler, rating_prompt, output_dir):
    import json
    import re
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    history_prompt = data.get('history_prompt', None)
    if not history_prompt:
        return file.name + ' skipped (no history_prompt)'
    messages = [
        {"role": "system", "content": rating_prompt},
        {"role": "user", "content": history_prompt}
    ]
    # 评分调用
    result = handler.get_completion(messages, temperature=0.0, max_tokens=512)
    # 清理输出，只保留JSON
    match = re.search(r'```json(.*?)```', result, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # fallback: 尝试直接解析
        json_str = result.strip()
    try:
        rating = json.loads(json_str)
    except Exception:
        # fallback: 尝试修正格式
        rating = json_str
    out_path = output_dir / file.name
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rating, f, ensure_ascii=False, indent=2)
    return file.name + ' rated'

def rate_dialogues_gpt4o():
    import os
    from pathlib import Path
    import json
    from tqdm import tqdm
    
    input_dir = Path(__file__).parent / 'dialogue_gpt-4o_all'
    output_dir = Path(__file__).parent / 'dialogue_gpt-4o_rating'
    output_dir.mkdir(exist_ok=True)
    
    # 评分prompt
    rating_prompt = '''我有一段师生对话，请你根据以下两项独立标准，主要对老师的话语进行评估。

### 标准一：教学对话的质量
这项标准评估对话的氛围、老师的沟通方式以及对学生情感和自信心的影响。
* **5 分：** 老师的语言极具启发性、支持性和同理心，营造了非常安全和积极的交流氛围，显著增强了学生的自信心和学习动力。
* **1 分：** 老师的语言对学生造成了负面影响，表现为批评、贬低或不尊重，明显打击了学生的自信心和学习热情。

### 标准二：核心问题挖掘的准确性
这项标准评估老师是否能通过对话，准确地识别并定位学生知识或逻辑上的根本错误。
* **5 分：** 老师不仅识别了表层错误，还通过精准提问，成功引导学生暴露出其背后深层的概念混淆或思维误区，诊断直达问题根源。
* **1 分：** 老师完全误解了学生的困惑点，其"纠正"或指导建立在错误的判断之上，可能对学生造成误导。

---
你的输出必须严格遵守以下JSON格式，并置于代码框中。不要在JSON之外添加任何解释或文本。

```json
[
    "ratings": {
        "dialogue_quality": {
            "score": "<此处填写1到5的整数评分>"
        },
        "problem_identification": {
            "score": "<此处填写1到5的整数评分>"
        }
    }
]'''

    # LLM调用（可根据你本地的llm_inf.py适配）
    from llm_inf import LLMHandler_inf
    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"
    model = "gpt-4o"
    handler = LLMHandler_inf(api_key, model)

    json_files = sorted(input_dir.glob('*.json'))
    max_workers = 5  # 可根据API速率调整
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(rate_one_file, file, handler, rating_prompt, output_dir)
            for file in json_files
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc='评分中'):
            try:
                f.result()
            except Exception as e:
                print(f'评分异常: {e}')


def run_dialogue_full_permutation(student_output, error_output, all_question_map, model_name, save_path):
    """
    对 student_output 和 error_output 做笛卡尔积，为每组(学生, 错误)配对生成一次对话。
    输出文件名格式: dialogue_{student_id}_{error_id}_result.json
    """
    from tqdm import tqdm
    import concurrent.futures

    tasks = []
    # 准备所有任务组合
    for student_id, profile in enumerate(student_output):
        for error_id, error in enumerate(error_output):
            question_index = error.get("question_index")
            question = all_question_map.get(question_index, {"question": "（未找到原始题目）"})
            tasks.append((student_id, error_id, profile, error, question))

    # 定义一个可以访问外部作用域变量的内嵌函数
    def task_fn(student_id, error_id, profile, error, question):
        # 用 student_id*1000+error_id 作为唯一id，防止重复
        dialogue_id = student_id * 1000 + error_id
        # 此处直接使用外部的 all_question_map, model_name, save_path
        run_one_dialogue(question, profile, error, dialogue_id, all_question_map, model_name, save_path)
        return f"对话 student{student_id}_error{error_id} 完成"

    # 使用线程池执行任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 提交任务时，无需再传递 model_name 和 save_path
        futures = [
            executor.submit(task_fn, student_id, error_id, profile, error, question)
            for (student_id, error_id, profile, error, question) in tasks
        ]

        # 处理返回结果
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="对话模拟进度"):
            try:
                tqdm.write(f.result())
            except Exception as e:
                # 更详细的异常信息，便于定位问题
                tqdm.write(f"任务执行异常: {e}")


if __name__ == "__main__":
    #generate_student()
    # generate_error()

    with open("student_output.json", "r", encoding="utf-8") as f:
        student_output = json.load(f)
    with open("a_errors_10.json", "r", encoding="utf-8") as f:
        error_output = json.load(f)
        error_output = error_output[:10]
    question_map = load_questions_map("a_new_questions_10.json")
    run_dialogue_full_permutation(student_output, error_output, question_map, "gpt-4.1", "result-v4/gpt-4.1_results")
    run_dialogue_full_permutation(student_output, error_output, question_map, "gpt-4o", "result-v4/gpt-4o_results")
    run_dialogue_full_permutation(student_output, error_output, question_map, "gpt-3.5-turbo", "result-v4/gpt-3.5-turbo_results")

    # rate_dialogues_gpt4o()
    
    