from langchain_openai import ChatOpenAI

from env_utils import LOCAL_BASE_URL, LOCAL_API_KEY, ZAI_API_KEY

# 连接 qwen
qw_llm = ChatOpenAI(
    model="qwen3.5-0.8b",   # 私有化部署时，自定的模型名称
    temperature=0.9,        # 控制模型输出的随机性和创造性，数值越高，输出越多样、发散、有创意
    presence_penalty=0.8,   # 控制模型是否重复使用已经出现过的词汇/概念，等于0不惩罚，大于0如果某个词已经出现过，它再次被选中的概率会大幅降低。这会强迫模型使用新词汇，拓展话题，避免车轱辘话。
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)

# 链接 deepseek
ds_llm = ChatOpenAI(
    model="ds-qwen3-8b",
    temperature=0.9,
    presence_penalty=0.8,
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)


message = [
    ("system", "你是一个助手，请回答我的问题。"),
    ("human", "请用中文回答我的问题：如何使用 langchain？")
]

# 测试连接是否成功
# qw_resp = qw_llm.invoke(message)
# print(f"千问回答：{qw_resp}")

# ds_resp = ds_llm.invoke(message)
# print(f"deepseek回答：{ds_resp}")


# 智谱大模型连接
from zai import ZhipuAiClient

# Initialize client
zai_llm = ZhipuAiClient(api_key=ZAI_API_KEY)

# Create chat completion
# zai_resp = zai_llm.chat.completions.create(
#     model="glm-5",
#     messages=[
#         {"role": "user", "content": "你好，请介绍一下自己, Z.ai!"}
#     ]
# )
# print(zai_resp.choices[0].message.content)


multiModel_llm = ChatOpenAI(
    model="qwen2.5-omni-3b",
    base_url=LOCAL_BASE_URL,
    api_key=LOCAL_API_KEY,
    temperature=0.9,
    presence_penalty=0.8,
)