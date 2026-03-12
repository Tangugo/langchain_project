from langchain_core.prompts import ChatPromptTemplate

from connect_llm import qw_llm

role_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一位拥有20年经验的资深 {role_name}。\n"
            "你的性格特点：{personality}。\n"
            "你的回答风格：{style}。\n"
            "请严格遵守以下约束：{constraints}"
        ),
        ("human", "{user_input}")
    ]
)

chain = role_template | qw_llm

response = chain.invoke({
    "role_name": "Python 架构师",
    "personality": "严谨、直接、喜欢最佳实践",
    "style": "使用技术术语，提供代码片段",
    "constraints": "不要解释基础概念，直接指出性能瓶颈",
    "user_input": "这段代码有什么性能问题？\nfor i in range(1000000): pass"
})
print(f"Role-Play: {response.content}")