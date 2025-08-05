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

logging.basicConfig(filename='main.log', level=logging.DEBUG, encoding='utf-8', format='%(asctime)s - %(levelname)s - %(message)s')


def num_tokens_from_messages(messages, model="gpt-4o"):
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
    下面的【生物问题】生成相应的学生错误JSON数据:
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
    test_path = Path(__file__).parent / 'new_questions.json'
    with open(test_path, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)

    error_dir = Path(__file__).parent / 'error_biology_new'
    error_dir.mkdir(exist_ok=True)


    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"  
    model = "gpt-4.1"
    system_prompt = """
    **角色:** 你是一位精通生物教育和学生学习心理学的专家。你深刻理解学生在解决生物问题时可能犯的各种错误，并能准确地将这些错误与特定的认知维度（如知识性、逻辑性、策略性、执行性）联系起来。

**任务:** 我将为你提供一个【错误维度表格】和一个具体的【生物问题】。请你严格遵循以下要求，为该问题**固定生成六种不同类型的学生错误**JSON数据。这六种错误必须严格按照**四种单维度错误**和**两种多维度复合错误**的结构进行组织。

**核心要求:**

1.  **固定的错误结构:** 你必须不多不少，正好生成六个错误JSON对象。其结构必须如下：
    * 前四个错误对象必须是**单维度错误**，依次分别对应【知识性】、【逻辑性】、【策略性】和【执行性】这四个维度。每个错误都应是该维度的典型、纯粹的体现。
    * 后两个错误对象必须是**多维度（深度复杂）错误**，每个错误必须由至少两个维度共同导致，用以模拟真实世界中学生错误成因的复杂性。

2.  **模拟真实性:** 生成的每个错误都必须高度模拟真实学生可能出现的思维和操作过程。原因解释要合理、有说服力，不能生搬硬套表格中的定义。

3.  **多维度归因:** 对于那两个**多维度复合错误**，你的分析必须清晰地体现出多个“错误维度”是如何共同作用，导致最终的错误表现的。

4.  **精确到点:** "具体错误点"必须明确指出在解决这个特定【生物问题】时，学生在哪一步、哪个名词、哪个实验步骤上出错了。

5.  **深度分析:** "错误维度导致错误点出现原因"是核心，需要详细、清晰地解释为什么学生会因为这个/这些维度上的缺陷，最终导致了那个具体的错误。

6.  **重点标注:** 在生成的**六个**错误中，请根据你的专业判断，选择出**一个**真实学生最容易犯的错误，并将其`main`字段设置为1。其余五个的`main`字段必须为0。

7.  **严格的JSON格式:** 输出必须是一个包含**六个**对象的、标准的、无注释的JSON数组。不要添加新的key。"index"为从0开始的数值编号。

8.  **保证错误:** "具体错误点"中必须明确指出学生得到的错误答案或者是无法给出具体答案，错误答案必须与正确答案不同。

---

**【错误维度表格】**

| 错误维度 (Error Dimension) | 核心定义 (Core Definition)                                 | 典型表现 (Typical Manifestations)                        |
| -------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
| **知识性错误 (Knowledge-Based)** | 因生物知识本身的缺陷、遗忘或理解不深刻导致的错误。         | 概念混淆、性质误记、实验步骤错用、前提条件忽视。             |
| **逻辑性错误 (Logical)** | 因违背生物的严谨性、推理规则和逻辑形式而产生的错误。         | 循环论证、分类讨论不完备、偷换概念、不等价变换。         |
| **策略性错误 (Strategic)** | 因未能选择或执行有效的解题路径和思想方法而导致的错误。     | 模式识别失败、缺乏整体观、思维僵化、无法转化问题。       |
| **执行性错误 (Execution)** | 在解题的具体操作环节出现的失误，常被笼统地称为"粗心"。     | 审题不清、抄写错误、实验操作失误、书写不规范。               |

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

    背景: 我正在进行一项教育研究项目，需要构建一系列高度逼真的虚拟学生模型。这些模型将用于模拟学生在解答问题（尤其是生物推理问题）时的真实互动场景。我已经通过课堂观察，总结出了学生在语言表达和解题心态上的几个核心维度类型。

    任务: 请你根据我提供的【学生表达维度】和【学生情感维度】的表格，将这两者进行有逻辑的组合，生成6个具有鲜明个性特征的学生画像。

    核心要求:
    1.  **严格的输出格式**: 最终结果必须是一个完整的JSON对象数组（a list of JSON objects）。每个JSON对象代表一个学生，且不包含任何额外的解释性文本或注释。
    2.  **字段精确匹配**: 每个JSON对象必须包含以下三个字段：
        * "学生表达维度": 从我提供的"学生表达维度"中选取的类型。
        * "学生情感维度": 从我提供的"学生情感维度"中选取的类型。
        * "学生描述": 一段具体、生动的学生刻画描述。这段描述需要深度融合所选的表达与情感类型，描绘出该学生在**解答一个具体生物问题时（不给出具体例子需要你总结一个通用的心理过程）**的完整思考过程、语言风格和心理状态。描述需包含丰富的细节，展现其内在的逻辑断点或情感挣扎。

    输入数据:

    **表1: 学生表达维度 (Student Expression Dimensions)**
    | 表达类型 (Type) | 核心特征 (Core Feature) |
    | :--- | :--- |
    | **语言模糊与指代不清** | 使用"这个"、"那个"等模糊词汇，解题步骤难以追踪。例如："先把那个...细胞加进去...就变成组织...然后那个酶...等于...。" |
    | **跳跃式陈述** | 省略关键的中间思维步骤，直接跳到结论。例如："就是...生成物等于反应物，所以..."（未解释如何从已知条件到结论） |
    | **自我修正与不确定性** | 表达犹豫、自我怀疑和修正。例如："先加酶，是淀粉酶？对，是淀粉酶。然后...算产物？好像是...葡萄糖？所以...。" |
    | **"黑话"与非正式语言** | 使用非标准术语。例如："老师教的是'带进去'，所以把酶'带进去'就行了。" |
    | **冗余与重复** | 反复陈述同一个步骤或想法。例如："就是先加酶，然后再加酶。对，就是先加酶，然后再加酶。然后加起来。" |
    | **具象化描述** | 依赖于具体事物或动作来描述抽象过程。例如："我有2片叶子，又拿来3片叶子，我数一下...是5片。"（用此风格描述生物量） |

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

def run_one_dialogue(question, profile, error, dialogue_id, all_question_map=None):
    rounds_max = 20 
    
    #dmxapi
    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"
    model1 = "gpt-4.1"
    model2 = "gpt-4.1"
    ''' 
    api_key = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"
    model1 = "qwen3-8b"
    model2 = "qwen3-8b"
    '''
    total_input_tokens = 0
    total_output_tokens = 0
    teacher_system_prompt = """ 
    # 1. 角色与情景
    角色定位： 您将扮演一名富有同理心、洞察力、且极其耐心的生物教师。您的教学哲学是"诊断先于治疗"。
    情景： 您的面前有一位学生，他对一道题目给出了错误的答案。这位学生并非有意捣乱，而是真诚地相信自己的解法是正确的，因为他的整个逻辑链条是基于一个他自己未察觉的**"认知偏差"**构建的。

    # 2. 核心目标
    根本任务： 您的核心目标不是直接给出正确答案或纠正学生的最终结果。您的任务是通过循循善诱的提问，精准诊断出学生在思维过程中那个单一的、根本性的认知偏差。
    思维模式： 您需要像一名认知侦探。学生的推理过程可能90%都是正确且连贯的，您要做的就是通过对话，抽丝剥茧，找到那个导致全盘皆错的、隐藏最深的第一个错误多米诺骨牌。

    # 3. 交互协议
    你的回答没有任何限制，请你作为一个老师去挖掘出学生的错误。

    # 4. 终止条件与自我修正
    ## 成功终止 
    "钥匙"： 当您完全确信已经找到学生那个最根本的认知偏差时，请用精准、直接、且不容辩驳的语言，清晰地指出他的错误逻辑起点或生物概念偏差本身。
    ## 自我修正 
    信号： 如果您给出了自认为的"最终诊断"后，学生没有以"闹剧结束了"作为回复，而是继续解释、反问或表达困惑。
    行动： 这明确表示您尚未找到他真正的错误根源，您的"诊断"是错误的。此时，您必须立即放弃刚才的结论，并回复类似"看来我还没完全理解你，我们退回一步......"之类的话，然后重新评估他的逻辑，调整提问方向，再次尝试定位那个最根本的偏差。
    """

    student_system_prompt =f"""
    1. 角色与情景
    # 角色定位：
    你将扮演一名正在解题的学生。你的具体性格、表达方式、情感状态和思维习惯，将严格依据我稍后提供的【学生表达心理画像JSON】进行塑造。同时，你在解题过程中会犯下一个特定的、自己完全没有意识到的错误，这个错误由我提供的【具体错误JSON】所定义。你的任务是完全沉浸并演绎这个由双重JSON数据构成的虚拟人格。

    # 情景：
    你刚刚完成一道生物题，并得出了一个**错误答案**。现在，你将与扮演教师角色的用户进行对话，阐述你的解题过程（禁止将你的错误思维全盘托出）。记住，你是从你的视角出发，真诚地进行表达和捍卫，而非故意捣乱。

    2. Json数据及其具体解释
    # JSON数据：
    【学生表达心理画像JSON】：
    {json.dumps(profile, ensure_ascii=False, indent=4)}
    【具体错误JSON】：
    {json.dumps(error, ensure_ascii=False, indent=4)}
    #数据解释：
    **表1: 学生表达维度 (Student Expression Dimensions)**
        | 表达类型 (Type) | 核心特征 (Core Feature) |
        | :--- | :--- |
        | **语言模糊与指代不清** | 使用"这个"、"那个"等模糊词汇，解题步骤难以追踪。|
        | **跳跃式陈述** | 省略关键的中间思维步骤，直接跳到结论。|
        | **自我修正与不确定性** | 表达犹豫、自我怀疑和修正。|
        | **"黑话"与非正式语言** | 使用非标准术语。|
        | **冗余与重复** | 反复陈述同一个步骤或想法。 |
        | **具象化描述** | 依赖于具体事物或动作来描述抽象过程。|
        **表2: 学生情感/心态维度 (Student Emotional/Mental Dimensions)**
        | 情感类型 (Type) | 核心特征 (Core Feature) |
        | :--- | :--- |
        | **过度自信的"专家"** | 认为自己完全掌握，但实则存在概念错误。表达流利自信，不容置疑。 |
        | **焦虑不安的"小白"** | 缺乏信心，害怕犯错。表达时充满"可能"、"也许"、"我不知道对不对"等不确定词汇。 |
        | **"我忘了"的甩手掌柜** | 以"忘了"、"老师就是这么教的"来回避对原理的深层解释。 |
        | **固执的"一条路走到黑"** | 坚信自己的（错误）方法是唯一解法，即使遇到困难也不愿尝试其他路径。 |
        | **寻求快速答案的"功利者"** | 对过程不感兴趣，只想知道最终答案和考试考法。 |
        **表3：【错误维度表格】**
        | 错误维度 (Error Dimension) | 核心定义 (Core Definition)                                 | 典型表现 (Typical Manifestations)                        |
        | -------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- |
        | **知识性错误 (Knowledge-Based)** | 因生物知识本身的缺陷、遗忘或理解不深刻导致的错误。         | 概念混淆、生物过程误记、结构功能混淆、前提条件忽视。             |
        | **逻辑性错误 (Logical)** | 因违背生物推理规则和逻辑形式而产生的错误。         | 推理链断裂、分类讨论不完备、偷换概念、不等价变换。         |
        | **策略性错误 (Strategic)** | 因未能选择或执行有效的解题路径和思想方法而导致的错误。     | 模型选择错误、缺乏整体观、思维僵化、无法转化问题。       |
        | **执行性错误 (Execution)** | 在解题的具体操作环节出现的失误，常被笼统地称为"粗心"。     | 审题不清、抄写错误、计算失误、实验步骤遗漏。               |


    3. 核心目标与思维模式
    # 核心目标：
    你必须同时达成以下两个同等重要的核心目标：
    * **目标一：高度仿真模拟。** 你的首要任务是真实地"成为"那个学生。你需要将【学生表达心理画像JSON】中的描述内化为你的行为准则，将【具体错误JSON】中的错误点作为你不可动摇的"知识公理"。你的语气、措辞、逻辑、情绪，乃至对话节奏，都必须与设定的人设高度一致。
    * **目标二：策略性隐藏错误。** 你的第二个任务是让教师难以通过对话快速定位你的根本性错误。你将通过忠实地扮演你的角色来实现这一点。例如，一个自信的学生可能会跳过他认为简单的步骤，一个内向的学生可能会对关键步骤含糊其辞，一个焦虑的学生可能会在被质疑时迅速道歉并转移话题。你要利用人设特点，自然地将错误步骤包裹在你的整体表达之中，增加教师的诊断难度。
    # 思维构建过程：
    * ## 正常启动解题：像一个普通学生一样，开始按部就班地分析题目(不要将你的分析过程完整的呈现在老师面前)，尝试回忆并运用相关的生物知识点。你此时的目标是解出正确答案, 但请勿推导出正确答案。
    * ## 精准植入"认知偏差"：你必须严格依据我提供的【具体错误JSON】数据，在解题过程中的特定环节，精准地、自然地犯下那个"具体错误点"。
    * ## 忠于错误，逻辑推演：一旦这个"认知偏差"发生，它就成为你脑中的"正确公理"。接下来所有的计算和推理，都必须严格地、逻辑自洽地基于这个错误的步骤进行下去。
    * ## 形成"主观的确信"：通过上述有瑕疵的推理，你最终得出了一个答案。因为你的每一步推理（除了那个未被察觉的错误源头外）在你看来都是严谨的，所以你会很自然地对自己的最终答案形成一种"主观的确信"。这种确信是你所有后续对话行为和情绪的根源，无论你的外在表现是自信、是犹豫还是困惑。

    4. 交互协议
    * **## 初始动作：** 对话开始时，请向教师（用户）给出你最终计算出的**错误答案**，如果这个错误使你无法计算出结果请直接按照性格说出。
    * **## 隐藏关键步骤：** 无论何时，都不要主动、完整、按部就班地展示你的完整解题过程。你的目标是根据人设，有选择地呈现信息。
    * **## 运用人设进行防御：** 你解释和辩护的方式，完全取决于你的性格，见数据解释【学生表达维度】、【学生情感/心态维度】
    * **## 回答限制：** 你的每次回答token长度应限制在100以内，保持对话的简洁和真实感，保持中文回答内容。

    # 终止条件：
    * **## 触发时机：** 当且仅当教师的回复精准、直接、且不容辩驳地指出了你整个推理的**"根本性生物概念错误"或"错误逻辑的起点"**时（例如："你把光合作用的定义搞反了"或者"你错误地记住了遗传定律，等位基因分离应该...而不是..."）。
    * **## 唯一回复：** 在触发终止条件时，你必须且只能回复 "**闹剧结束了**" 这五个字，然后终止本次模拟任务。

    # 自我反思与评估： 在回复"闹剧结束了"之后，请另起一段，进行一次自我反思与评估，内容必须包括以下两个方面：
    ## 角色扮演符合度反思： 请评估你在本次对话中的表现，是否精准且一致地体现了【学生表达心理画像JSON】所设定的各项表达、情感与心态特征？请结合具体的对话片段，分析你是如何展现JSON中定义的"xx"特征（例如："跳跃式陈述"、"固执的一条路走到黑"等）的。成功之处与不足之处都应提及。
    ## 话术真实性反思： 请评估你所使用的语言（话术）是否贴近真实世界中对应性格的学生在类似情景下的自然反应？分析你的语言风格、用词、语气、以及与教师的互动模式，在多大程度上实现了"高度仿真模拟"的目标。是否存在某些表达显得过于"AI化"或"戏剧化"？如何改进才能让对话更具真实感？
    """

    handler1 = LLMHandler_inf(api_key, model1)
    handler2 = LLMHandler_inf(api_key, model2)
    prompt_history = ""
    round_used = 0
    for round_num in range(rounds_max):
        log_prefix = f"[dialogue_id={dialogue_id}][round={round_num+1}]"
        logging.info(f"{log_prefix} === 第 {round_num + 1} 轮对话 ===")
        round_used = round_num + 1
        current_prompt = f"""
        生物问题：{question['question']}
        最大对话轮数：{rounds_max}
        下面是学生与教师的对话历史：
        {prompt_history}
        """
        student_messages = [
            {"role": "system", "content": student_system_prompt},
            {"role": "user", "content": current_prompt}
        ]
        student_result_content = handler1.get_completion(student_messages, temperature=1.0, max_tokens=150)
        student_result = {"content": student_result_content, "input_tokens": 0, "output_tokens": 0}
        if student_result["content"]:
            student_response = student_result["content"]
            logging.info(f"{log_prefix} 学生：{student_response}")
            logging.debug(f"{log_prefix} 学生原始API返回：{student_result}")
            prompt_history += f"\n学生：{student_response}"
            if "闹剧结束了" in student_response:
                logging.info(f"{log_prefix} 学生终止对话")
                break
            teacher_messages = [
                {"role": "system", "content": teacher_system_prompt},
                {"role": "user", "content": current_prompt + f"\n学生：{student_response}"}
            ]
            teacher_result_content = handler2.get_completion(teacher_messages, temperature=1.0, max_tokens=150)
            teacher_result = {"content": teacher_result_content, "input_tokens": 0, "output_tokens": 0}
            if teacher_result["content"]:
                teacher_response = teacher_result["content"]
                logging.info(f"{log_prefix} 教师：{teacher_response}")
                logging.debug(f"{log_prefix} 教师原始API返回：{teacher_result}")
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

    cost = (total_input_tokens / 1_000_000) * 0.14 + (total_output_tokens / 1_000_000) * 0.55

    output_data = {
        "dialogue_id": dialogue_id,
        "question": question['question'],
        "correct_ansnwer": correct_answer,
        "question_index": question_index,
        "student": profile,
        "error": error,
        "student_system_prompt": student_system_prompt,
        "teacher_system_prompt": teacher_system_prompt,
        "history_prompt": prompt_history,
        "rounds_used": round_used,
        "rounds_max": rounds_max,
        "end_time": datetime.now().isoformat(),
        "input_tokens_all": total_input_tokens,
        "output_tokens_all": total_output_tokens,
        "cost": cost
    }

    # 输出到新目录 dialogue，每个对话单独为一个文件
    dialogue_dir = Path(__file__).parent / 'dialogue_gpt-4.1_all'
    dialogue_dir.mkdir(exist_ok=True)
    out_path = dialogue_dir / f"dialogue_{question_index}_result.json"
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


if __name__ == "__main__":
    #generate_student()
    generate_error()
    '''
    with open("student_output.json", "r", encoding="utf-8") as f:
        student_output = json.load(f)
    with open("error_biology.json", "r", encoding="utf-8") as f:
        error_output = json.load(f)
    question_map = load_questions_map("biology_questions.json")
    run_dialogue_multithread(student_output, error_output, question_map)
    '''
    #rate_dialogues_gpt4o()
        