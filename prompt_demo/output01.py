from pydantic import BaseModel, Field

from connect_llm import qw_llm


class User(BaseModel):
    name: str = Field(description="用户姓名")
    age: int = Field(description="用户年龄")

# 一行代码，自动处理所有底层细节
structured_llm = qw_llm.with_structured_output(User)

result = structured_llm.invoke([{"role": "user", "content": "小明今年 5 岁啦"}])
# result 是 User 对象，不是字典，也不是字符串
print(result)