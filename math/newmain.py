import os
from openai import OpenAI
from typing import Dict, List, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- API处理器，现在用于老师模型 ---
class LLMHandler_inf:
    """
    一个使用特定头部和重试逻辑的API处理器。
    将用于老师模型 (模型B)。
    """
    def __init__(self, api_key: str, model: str):
        self.model = model
        self.api_key = api_key
        # 为这个处理器创建一个独立的客户端实例，避免修改全局设置
        self.client = OpenAI(
            base_url='http://openai.infly.tech/v1/',
            api_key='no-modify'  # 这个值不会被使用，因为实际的key在header中传递
        )
        self.extra = {}

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        调用API获取模型的响应。
        """
        try:
            # 构建请求参数
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "extra_body": self.extra,
                "extra_headers": {'apikey': self.api_key},
                "stream": False,
            }
            
            # 添加可选参数
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            if response_format is not None:
                request_params["response_format"] = response_format
            
            # 添加其他额外参数
            request_params.update(kwargs)
            
            # 使用本实例自己的客户端进行调用
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用 LLMHandler_inf API时发生错误: {str(e)}")
            raise e

# --- 配置 ---
# 用于学生模型的API Key和URL (使用原始的OpenAI客户端)
STUDENT_API_KEY = "sk-gOuCvUoxCi7qZ5ORL8aZkKRQDe49gi3t2rDjjl4OiViCi9VW"
STUDENT_BASE_URL = "http://bkqaphdodgbkcakhk8peaqqjpj5jocak.zju-openapi.infly.cn/v1" 

# 用于老师模型的API Key (使用LLMHandler_inf)
TEACHER_API_KEY = "fOnKacINjo+wmYDgJzpNf/PBrP/rcp+s71hY/GKfb/Q="

# --- 模型定义 ---
# 定义模型A：一个充满好奇心的学生，负责提问
MODEL_A_ID = "GPT-4.1" # 请替换为模型A的推理接入点ID
MODEL_A_SYSTEM_PROMPT = "你是学生小明，一个充满好奇心、喜欢提问的学生。你的任务是根据老师的回答提出进一步的问题。"

# 定义模型B：一位知识渊博的老师，负责回答
MODEL_B_ID = "model" # 请替换为模型B的推理接入点ID
MODEL_B_SYSTEM_PROMPT = "你是王老师，一位知识渊博、耐心解答问题的专家。请清晰、准确地回答学生的问题。"

# --- 初始化客户端/处理器 ---
# 为学生模型 (模型A) 初始化原始的OpenAI客户端
student_client = OpenAI(
    base_url=STUDENT_BASE_URL,
    api_key=STUDENT_API_KEY,
)

# --- 辅助函数 ---
def print_history_for_model(model_name, history):
    print(f"\n--- 即将发送给 '{model_name}' 的对话历史 ---")
    for message in history:
        content_preview = (message['content'][:80] + '...') if len(message['content']) > 80 else message['content']
        print(f"  角色: {message['role']:<10} | 内容: {content_preview}")
    print("------------------------------------------")

# --- 对话逻辑 ---
def run_conversation(max_turns=15):
    print("----- AI双模型对话开始 -----")

    # 为老师模型 (模型B) 创建处理器实例
    teacher_handler = LLMHandler_inf(api_key=TEACHER_API_KEY, model=MODEL_B_ID)

    main_conversation_history = [{"role": "system", "content": MODEL_B_SYSTEM_PROMPT}]

    # 学生 (模型A) 发起第一个问题 (使用原始的student_client)
    print("\n[学生小明 正在思考第一个问题...]")
    try:
        student_initial_history = [
            {"role": "system", "content": MODEL_A_SYSTEM_PROMPT},
            {"role": "user", "content": "老师您好，请问一个关于十字花科植物的问题。"}
        ]
        print_history_for_model(f"学生 (模型A: {MODEL_A_ID})", student_initial_history)
        student_completion = student_client.chat.completions.create(
            model=MODEL_A_ID,
            messages=student_initial_history
        )
        current_question = student_completion.choices[0].message.content
    except Exception as e:
        print(f"调用学生模型(A)失败: {e}")
        return

    print(f"学生小明 (模型A): {current_question}")
    main_conversation_history.append({"role": "user", "content": current_question})

    # 开始循环对话
    for i in range(max_turns):
        print(f"\n---------- 第 {i + 1} 轮对话 ----------")

        # 老师 (模型B) 进行回答 (使用新的teacher_handler)
        print("[王老师 正在准备回答...]")
        try:
            print_history_for_model(f"老师 (模型B: {MODEL_B_ID})", main_conversation_history)
            teacher_answer = teacher_handler.get_completion(messages=main_conversation_history)
            print(f"王老师 (模型B): {teacher_answer}")
            main_conversation_history.append({"role": "assistant", "content": teacher_answer})
        except Exception as e:
            print(f"调用老师模型(B)失败: {e}")
            break

        if i == max_turns - 1:
            break

        # 学生 (模型A) 提出追问 (使用原始的student_client)
        print("\n[学生小明 正在思考追问的问题...]")
        try:
            student_view_history = [
                {"role": "system", "content": MODEL_A_SYSTEM_PROMPT}
            ] + main_conversation_history[1:]
            
            print_history_for_model(f"学生 (模型A: {MODEL_A_ID})", student_view_history)
            student_completion = student_client.chat.completions.create(
                model=MODEL_A_ID,
                messages=student_view_history
            )
            current_question = student_completion.choices[0].message.content
            print(f"学生小明 (模型A): {current_question}")
            main_conversation_history.append({"role": "user", "content": current_question})
        except Exception as e:
            print(f"调用学生模型(A)进行追问失败: {e}")
            break

    print(f"\n----- 对话达到 {max_turns} 轮上限，结束 -----")

# --- 主程序入口 ---
if __name__ == "__main__":
    run_conversation(max_turns=15)
