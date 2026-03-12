from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from connect_llm import qw_llm

prompt_template = ChatPromptTemplate.from_template(
    "{topic}"
    '你必须始终输出一个包含"answer"和"followup_question"键的 json 对象，其中"answer"代表：对用户问题的回答，"followup_question"代表：用户可能提出的后续问题'
)

chain = prompt_template | qw_llm | SimpleJsonOutputParser()
resp = chain.invoke({"topic": "细胞的动力源是什么？"})
print(resp)
