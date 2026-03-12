from langchain_core.prompts import PromptTemplate

from connect_llm import qw_llm

# 定义模板
zero_shot_template = PromptTemplate.from_template(
    "请将以下文本翻译成 {language}：\n{text}"
)

# 构建链
chain = zero_shot_template | qw_llm

# 调用
response = chain.invoke({"language": "英语", "text": "你好，世界"})
print(f"Zero-Shot: {response.content}")