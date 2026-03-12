from langchain_core.prompts import PromptTemplate, FewShotChatMessagePromptTemplate, ChatPromptTemplate

from connect_llm import qw_llm

# 方法 A: 直接指令法 (最简单)
cot_template = PromptTemplate.from_template(
    "问题：{question}\n"
    "请一步步进行推理，最后给出答案。\n"
    "推理过程："
)

# 方法 B: 少样本 CoT (效果更好，提供带推理的示例)
cot_examples = [
    {
        "q": "罗杰有5个网球，又买了2筒(每筒3个)，共有几个？",
        "a": "罗杰原有5个。2筒每筒3个，即2*3=6个。总共5+6=11个。答案是11。"
    },
    {
        "q": "食堂有23个苹果，用掉20个，又买来6个，现在有几个？",
        "a": "原有23个。用掉20个剩3个。买来6个，3+6=9个。答案是9。"
    }
]

# 构建带推理示例的 Few-Shot 模板
cot_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{q}"),
        ("ai", "{a}")
    ]),
    examples=cot_examples,
    input_variables=["q"]
)

final_cot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个数学专家。请先展示推理步骤，再给出最终答案。"),
    cot_few_shot,
    ("human", "{q}")
])

chain = final_cot_prompt | qw_llm
response = chain.invoke({"q": "我有3个苹果，吃了1个，又分了剩下的一半给朋友，我还剩几个？"})
print(f"CoT: {response.content}")