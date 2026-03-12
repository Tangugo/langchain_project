from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from connect_llm import qw_llm

# 定义示例数据
examples = [
    {"input": "快乐的", "output": "正面"},
    {"input": "悲伤的", "output": "负面"},
    {"input": "愤怒的", "output": "负面"},
]

# 创建示例提示模板 (定义示例的格式)
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "情感分析输入: {input}"),
        ("ai", "情感分类: {output}"),
    ]
)

# 创建少样本消息提示 (自动选取最相关的 k 个示例，这里全选)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input"], # 这里不需要额外变量，因为例子是固定的
)

# 主模板：包含系统指令 + 少样本示例 + 用户最终输入
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个情感分析助手。请根据示例判断用户输入的情感。"),
        few_shot_prompt, # 插入示例
        ("human", "{input}"), # 用户实际输入
    ]
)

chain = final_prompt | qw_llm
response = chain.invoke({"input": "开心的"})
print(f"Few-Shot: {response.content}")