import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-a068e5aab8cd444ea14f397a95f24306", # 请使用您的有效API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-8b",  # 或您正在使用的其他qwen模型
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    # 解决方案：添加下面这行代码
    extra_body={"enable_thinking": False}
)

print(completion.model_dump_json(indent=2))