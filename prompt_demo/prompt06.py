from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from connect_llm import qw_llm

# 第一步：生成初始答案
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个解题助手。"),
    ("human", "问题：{question}\n请给出一个初步解决方案。")
])

# 第二步：反思/验证 (Critic)
reflector_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严格的审查员。检查以下方案是否有逻辑漏洞或错误。如果有，请指出并提供修正后的方案。"),
    ("human", "原始方案：{initial_answer}\n\n请审查并修正：")
])

# 构建反思链 (Generator -> Reflector)
# 注意：这里需要把第一步的输出作为第二步的输入
reflection_chain = (
    generator_prompt
    | qw_llm
    | StrOutputParser() # 获取纯文本
    | (lambda initial_answer: {"initial_answer": initial_answer, "question": None}) # 构造第二步的输入 (这里简化处理，实际需传递原问题)
    # 上面 lambda 写法为了演示结构，实际完整写法如下：
)

# ✅ 正确的完整反射链写法 (使用 RunnablePassthrough 传递上下文)
from langchain_core.runnables import RunnablePassthrough

full_reflection_chain = (
    {
        "initial_answer": (generator_prompt | qw_llm | StrOutputParser()),
        "question": RunnablePassthrough() # 保留原始问题传入下一步（如果需要）
    }
    | reflector_prompt
    | qw_llm
)

# 由于 reflector_prompt 定义中只需要 initial_answer，我们调整一下 prompt 定义以匹配
reflector_prompt_simple = ChatPromptTemplate.from_messages([
    ("system", "审查以下方案，如有错误请修正："),
    ("human", "方案：{initial_answer}")
])

final_self_consistency_chain = (
    {
        "initial_answer": (generator_prompt | qw_llm | StrOutputParser())
    }
    | reflector_prompt_simple
    | qw_llm
)

response = final_self_consistency_chain.invoke("鸡兔同笼，头35，脚94，求鸡兔各几只？")
print(f"Self-Reflection: {response.content}")