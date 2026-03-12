import base64
import uuid
from io import BytesIO

import gradio as gr
from PIL import Image
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig

# 导入自定义的 LLM 连接模块
from connect_llm import multiModel_llm

# 定义提示词模板：包含系统指令和消息占位符
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个多模态AI助手，可以处理文本、音频、图像输入"),
    MessagesPlaceholder(variable_name="message")  # 用于插入历史对话消息
])

# 构建执行链：提示词 -> LLM 模型
chain = prompt | multiModel_llm


def get_session_chat_history(session_id):
    """
    获取或创建指定 session_id 的聊天历史记录对象。
    使用 SQLite 数据库持久化存储聊天记录。
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db",
    )


# 包装 Chain 以支持自动历史记忆
# get_session_chat_history: 用于获取历史记录的函数
chain_history = RunnableWithMessageHistory(
    chain,
    get_session_chat_history,
)

# 初始化全局配置，生成一个唯一的 session_id
config = RunnableConfig()
config["configurable"] = {"session_id": str(uuid.uuid4())}


def get_last_user_after_assistant(chat_history):
    """
    辅助函数：获取最后一个 Assistant 回复之后的所有用户消息。
    (注：当前主逻辑中未直接使用此函数，可用于特定上下文截取场景)
    """
    if not chat_history:
        return None
    if chat_history[-1]["role"] == "assistant":
        return None

    last_assistant_idx = -1
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i]["role"] == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx == -1:
        return chat_history
    else:
        return chat_history[last_assistant_idx + 1:]


def transcribe_audio(audio_file, audio_type="wav"):
    """
    将音频文件读取并转换为 Base64 编码的 Data URI 格式。
    符合多模态模型输入的 JSON 格式要求。
    """
    try:
        with open(audio_file, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode("utf-8")
        audio_message = {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/{audio_type};base64,{audio_data}",
            }
        }
        return audio_message
    except Exception as e:
        print(e)
        return None


def transcribe_image(image_file):
    """
    将图片文件读取、转换为 JPEG 格式并编码为 Base64 Data URI。
    """
    with Image.open(image_file) as img:
        buff = BytesIO()
        img.save(buff, format="JPEG")
        img_data = base64.b64encode(buff.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_data}",
                "detail": "low"
            }
        }

def transcribe_video(video_file, max_frames=10):
    """
    处理视频文件：均匀抽取指定数量（默认 10 帧）的关键帧，转换为图片列表。
    依赖库：opencv-python (cv2)

    参数:
        video_file: 视频文件路径
        max_frames: 要抽取的最大帧数（默认 10），确保覆盖全视频
    """
    try:
        import cv2
    except ImportError:
        print("错误: 未安装 opencv-python。请运行: pip install opencv-python")
        return []

    content_list = []
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_file}")
        return []

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"警告: 视频帧数为 {total_frames}，无法抽帧。")
        cap.release()
        return []

    # 计算均匀抽帧的步长（向上取整，确保至少抽 1 帧）
    # 例如：100 帧 → 步长 = ceil(100/10) = 10；95 帧 → ceil(95/10)=10；10 帧 → 步长=1
    step = max(1, (total_frames + max_frames - 1) // max_frames)  # 等价于 math.ceil(total_frames / max_frames)

    # 预分配目标帧索引（避免重复读取）
    target_indices = [i * step for i in range(max_frames)]
    # 确保最后一个索引不超过 total_frames-1
    target_indices = [idx for idx in target_indices if idx < total_frames]

    # 重置帧计数器，逐帧读取并只在目标索引处处理
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in target_indices:
            # OpenCV 读取的是 BGR，需转为 RGB 才能被 PIL 正确处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            buff = BytesIO()
            img_pil.save(buff, format="JPEG", quality=50)  # 适当压缩
            img_data = base64.b64encode(buff.getvalue()).decode("utf-8")

            content_list.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_data}",
                    "detail": "low"
                }
            })
            saved_count += 1

            # 如果已达到最大帧数，提前退出
            if saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"视频处理完成：共抽取 {saved_count} 帧（目标 {max_frames} 帧）")
    return content_list



def process_message(chat_history, user_input, audio_file, file_objs):
    """
    处理用户输入的主函数：支持文本、语音、图片。
    1. 收集并转换所有输入媒体为模型可理解的格式。
    2. 调用 LLM 获取回复。
    3. 更新聊天记录并清空输入框。
    """
    content = []


    # 处理图片、视频文件列表
    if file_objs:
        for obj_path in file_objs:
            if obj_path.endswith((".png", ".jpg", ".jpeg")):
                img_msg = transcribe_image(obj_path)
                if img_msg:
                    content.append(img_msg)
                    # 记录到前端历史显示 (此处记录路径，实际显示依赖 Gradio 自动处理或后续逻辑)
                    chat_history.append({"role": "user", "content": {"path": obj_path}})
            elif obj_path.endswith((".mp4", ".avi")):
                video_frames = transcribe_video(obj_path, 5)
                content.extend(video_frames)
                if video_frames:
                    chat_history.append({"role": "user", "content": {"path": obj_path, "type": "video"}})
            else:
                print(f"不支持的文件类型: {obj_path}")

    # 处理音频文件
    if audio_file:
        audio_msg = transcribe_audio(audio_file)
        if audio_msg:
            content.append(audio_msg)
            # 记录到前端历史显示
            chat_history.append({"role": "user", "content": {"path": audio_file}})

    # 处理文本输入
    if user_input and user_input.strip():
        content.append({"type": "text", "text": user_input})
        chat_history.append({"role": "user", "content": user_input})

    # 如果没有有效内容，直接返回，不触发模型调用
    if not content:
        return chat_history, "", None, []

    # 构造人类消息对象
    input_message = HumanMessage(content=content)

    # 调用带历史记忆的 Chain
    # config 中包含了 session_id，用于区分不同用户的会话
    resp = chain_history.invoke({"message": [input_message]}, config=config)

    # 将 AI 回复添加到聊天记录
    chat_history.append({"role": "assistant", "content": resp.content})

    # 返回更新后的聊天记录，并清空输入组件 (文本="", 音频=None, 图片=[])
    return chat_history, "", None, []


# 构建 Gradio 界面
with gr.Blocks(title="多模态聊天机器人") as block:
    # 聊天展示区域
    chatbot = gr.Chatbot(max_height=500, label="聊天机器人")

    with gr.Row():
        # 文本输入框
        text_input = gr.Textbox(
            placeholder="请输入消息...",
            show_label=False,
            scale=3
        )
        # 语音输入组件 (麦克风)
        # type="filepath" 表示后端接收的是文件路径字符串
        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="录音",
            show_label=False,
            scale=1
        )
        # 图片上传组件
        # file_count="multiple" 允许一次上传多张图片
        file_input = gr.File(
            file_types=["image", "video"],
            file_count="multiple",
            label="支持图片(png, jpg, jpeg)和视频(mp4, avi)文件",
            show_label=True,
            scale=1,
            height=300
        )
        # 发送按钮
        submit_btn = gr.Button("发送", scale=1)

    # 绑定事件：点击发送按钮触发 process_message
    submit_btn.click(
        process_message,
        inputs=[chatbot, text_input, audio_input, file_input],
        outputs=[chatbot, text_input, audio_input, file_input]
    )

    # 绑定事件：在文本框按回车键触发 process_message
    text_input.submit(
        process_message,
        inputs=[chatbot, text_input, audio_input, file_input],
        outputs=[chatbot, text_input, audio_input, file_input]
    )

if __name__ == "__main__":
    # 启动应用
    # theme="soft" 使用柔和主题
    # debug=True 开启调试模式，显示详细日志
    block.launch(theme="soft", debug=True)