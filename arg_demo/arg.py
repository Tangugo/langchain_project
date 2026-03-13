import sqlite3
from os import path

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from connect_llm import qw_llm
from custom_embeding import CustomQwen3Embedding

# ================= 配置部分 =================
# 加载向量模型
local_model_path = "/Users/zhu/projects/ai/langchain_project/models/Qwen3-Embedding-0.6B"
qwen_embedding = CustomQwen3Embedding(local_model_path)

# 构建向量数据库
vector_db_path = "./chroma_db"
vector_store = Chroma(
    persist_directory=vector_db_path,
    embedding_function=qwen_embedding,
    collection_name="t_agent_blog",
)

def select_one():
    """判断向量数据库是否已经存在数据"""

    # 连接数据库
    conn = sqlite3.connect(path.join(vector_db_path, "chroma.sqlite3"))

    try:
        # 创建游标 (Cursor) - 用于执行 SQL 语句
        cursor = conn.cursor()

        # 执行查询 SQL
        cursor.execute("SELECT embedding_id FROM embeddings where embedding_id='doc_1'")

        # 获取结果
        one_row = cursor.fetchone()
        if one_row:
            print("向量数据库已存在数据。")
            return True
        else:
            print("向量数据库不存在数据。")
            return False

    except sqlite3.Error as err:
        print(f"数据库错误: {err}")
        raise err

    finally:
        # 关闭连接 (重要！释放资源)
        if conn:
            conn.close()

def create_dense_db():
    """把网络上关于agent的博客数据写入向量数据库"""

    # 查询向量数据库，如果存在数据则表示该博客内容向量数据已经存在，不再创建
    if select_one():
        return

    loader = WebBaseLoader(
        web_path=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    )
    docs_list = loader.load()

    # 切割文本
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs_list)

    print(f"切割后，文档数量为：{len(docs)}")

    ids = [f"doc_{i}" for i in range(len(docs))]
    vector_store.add_documents(documents=docs, ids=ids)

create_dense_db() # 如果需要重新构建数据库，取消注释

# ================= Prompt 定义 =================

# 1. 重写问题的 Prompt (用于处理多轮对话的历史上下文)
contextualize_q_system_prompt = (
    "给定聊天历史和最新的用户问题（可能引用聊天历史中的上下文），"
    "将其重新表述为一个独立的问题（不需要聊天历史也能理解）。"
    "不要回答问题，只需在需要时重新表述问题，否则保持原样。"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 2. 最终回答的 Prompt (RAG 核心)
system_prompt = (
    "你是一个问答任务助手。"
    "使用以下检索到的上下文来回答问题。"
    "如果不知道答案，就说你不知道。"
    "回答最多三句话，保持简洁。"
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ================= 构建链条 (LCEL 方式 - 纯串行稳定版) =================

# 1. 基础检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 文档格式化辅助函数
def format_docs(docs):
    if not docs:
        return "未找到相关上下文。"
    return "\n\n".join(doc.page_content for doc in docs)

# --- 步骤 1: 重写问题 ---
# 输入: {"input": str, "chat_history": list}
# 输出: str (重写后的问题)
rewrite_step = (
    contextualize_q_prompt
    | qw_llm
    | StrOutputParser()
)

# --- 步骤 2: 构造检索所需的完整上下文 ---
# 输入: {"input": str, "chat_history": list}
# 逻辑: 先运行 rewrite_step 得到新问题，同时保留原始 input 和 history
# 我们使用 RunnablePassthrough.assign() 来安全地添加新字段，而不是用并行字典
from langchain_core.runnables import RunnableLambda

def prepare_retrieval_input(data):
    # data 包含: input, chat_history
    # 我们需要手动调用 rewrite_step 来获取 rewritten_question
    # 注意：这里必须在同步上下文中调用，或者使用 invoke
    rewritten_q = rewrite_step.invoke(data)
    return {
        "rewritten_question": rewritten_q,
        "original_input": data["input"],
        "chat_history": data.get("chat_history", [])
    }

# 将准备逻辑封装为一个 Runnable
prepare_step = RunnableLambda(prepare_retrieval_input)

# --- 步骤 3: 执行检索并格式化 ---
def retrieve_and_format(data):
    docs = retriever.invoke(data["rewritten_question"])
    formatted_context = format_docs(docs)
    return {
        "context": formatted_context,
        "input": data["original_input"],
        "chat_history": data["chat_history"]
    }

retrieve_step = RunnableLambda(retrieve_and_format)

# --- 步骤 4: 组装最终链 (纯串行) ---
# 流程: 原始输入 -> 准备数据(含重写) -> 检索并格式化 -> 填充Prompt -> LLM -> 解析
rag_chain_logic = (
    prepare_step      # 输出: {rewritten_question, original_input, chat_history}
    | retrieve_step   # 输出: {context, input, chat_history}
    | qa_prompt       # 输出: PromptMessages
    | qw_llm          # 输出: AIMessage
    | StrOutputParser() # 输出: str
)

# ================= 添加会话记忆包装器 =================

store = {}

def get_session_chat_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain_logic,
    get_session_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ================= 运行测试 =================

config = RunnableConfig()
config["configurable"] = {"session_id": "user123"}

print("--- 第一轮问答 ---")
try:
    resp1 = conversational_rag_chain.invoke(
        input={"input": "What is Task Decomposition?"},
        config=config
    )
    print(resp1)
except Exception as e:
    import traceback
    print(f"发生错误: {e}")
    traceback.print_exc()

print("\n--- 第二轮问答 (带上下文) ---")
try:
    resp2 = conversational_rag_chain.invoke(
        input={"input": "What are common ways of doing it?"},
        config=config
    )
    print(resp2)
except Exception as e:
    import traceback
    print(f"发生错误: {e}")
    traceback.print_exc()