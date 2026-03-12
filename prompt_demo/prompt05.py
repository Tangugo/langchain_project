from langchain_core.prompts import PromptTemplate

from connect_llm import qw_llm

# 优化后的 Prompt
structured_template = PromptTemplate.from_template(
    """# 角色
你是一名精准的数据提取助手。

# 任务
从下方的 <data> 标签中提取**所有被提及的人名**及其**文中明确提到或通过简单上下文可知的年龄**。

# 重要规则
1. **不要遗漏**：即使某个人物只是作为比较的参照物（例如“比王五大”中的王五），只要文中提到了他的名字和年龄，必须提取。
2. **直接提取**：优先提取文中直接给出的数字。如果年龄需要通过简单加减法得出（如“比...大2岁”），也可以计算后提取，但必须确保所有提及的人都出现在结果中。
3. **格式严格**：仅输出 JSON，无其他文字。

# 输出格式
{{"people": [{{"name": "姓名", "age": 年龄数字}}]}}

# 数据
<data>
{raw_text}
</data>

# 开始提取
"""
)

chain = structured_template | qw_llm
text_data = "张三今年25岁，李四比王五（30岁）大2岁。"
response = chain.invoke({"raw_text": text_data})
print(f"Optimized: {response.content}")