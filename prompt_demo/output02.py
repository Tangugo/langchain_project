from pydantic import Field, BaseModel

from connect_llm import qw_llm


class ResponseFormater(BaseModel):
    """
    返回结果结构类
    """
    answer: str = Field(description="对用户问题的回答")
    followup_question: str = Field(description="用户可能提出的后续问题")


runnable = qw_llm.bind_tools([ResponseFormater])
resp = runnable.invoke("细胞的动力源是什么？")
print(resp)

resp.pretty_print()