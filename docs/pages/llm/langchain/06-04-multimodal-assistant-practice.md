---
title: 实战：构建一个多模态智能助手
description: 整合视觉理解 + 语音交互，搭建能看图、听语音、语音回复的完整应用
---
# 实战：构建一个多模态智能助手

前面两节我们分别学习了图像理解和语音交互的能力。这一节我们将它们全部整合起来，构建一个**完整的多模态智能助手**——它能看懂用户发的图片、能听懂用户的语音、还能用语音回复。同时它还保留了前几章学到的记忆能力，是一个真正意义上的"全能"助手。

## 功能规划

我们的多模态助手将支持以下交互方式：

| 输入方式 | 处理方式 | 输出方式 |
|---------|---------|---------|
| 纯文本 | LLM 直接处理 | 文本 / 语音 |
| 语音输入 | STT → LLM | 文本 / 语音 |
| 图片 + 文字 | 多模态 LLM | 文本 / 语音 |
| 图片 + 语音 | STT + 多模态 LLM | 文本 / 语音 |

## 第一步：核心类设计

```python
"""
multimodal_assistant.py — 多模态智能助手
"""
import os
import base64
import mimetypes
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_history import InMemoryChatMessageHistory
from langchain.runnables.history import RunnableWithMessageHistory

load_dotenv()


class MultimodalAssistant:
    """多模态智能助手 — 能看图、能听、能说、有记忆"""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        system_prompt: str = None,
        tts_voice: str = "nova"
    ):
        # OpenAI 客户端（用于 STT 和 TTS）
        self.openai_client = OpenAI()
        self.tts_voice = tts_voice

        # LangChain Chat Model（用于对话和多模态）
        self.chat = ChatOpenAI(model=model_name, temperature=0)

        self.system_prompt = system_prompt or (
            "你是一个多功能 AI 助手。你能理解文字、图片和语音。"
            "回答要准确、简洁、有帮助。"
        )

        # 基础对话 Chain
        base_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        self.base_chain = base_prompt | self.chat

        # 记忆管理
        self.store = {}

        # 带记忆的 Chain
        from langchain.runnables.history import RunnableWithMessageHistory
        self.chain_with_memory = RunnableWithMessageHistory(
            runnable=self.base_chain,
            get_session_history=self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
```

### 各个能力模块的实现

```python
    # ========== 语音能力 ==========

    def listen(self, audio_path: str) -> str:
        """STT: 音频文件 → 文字"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        with open(audio_path, "rb") as f:
            result = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        return result.text

    def speak(self, text: str, output_path: str = "response.mp3") -> str:
        """TTS: 文字 → 音频文件"""
        if len(text) > 4000:
            text = text[:4000]

        resp = self.openai_client.audio.speech.create(
            model="tts-1",
            voice=self.tts_voice,
            input=text
        )
        resp.stream_to_file(output_path)
        return output_path

    # ========== 视觉能力 ==========

    def _encode_image(self, image_path: str) -> str:
        """本地图片转 Base64 Data URI"""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        return f"data:{mime};base64,{b64}"

    def see(self, image_source: str, question: str) -> str:
        """图像理解: 图片 + 问题 → 回答"""
        if os.path.exists(image_source):
            url = self._encode_image(image_source)
        else:
            url = image_source

        message = HumanMessage(content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": url}}
        ])

        return self.chat.invoke([message]).content

    # ========== 记忆能力 ==========

    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
```

### 统一入口方法

```python
    # ========== 核心交互方法 ==========

    def chat_text(self, text: str, session_id: str = "default") -> str:
        """纯文字对话（带记忆）"""
        response = self.chain_with_memory.invoke(
            {"input": text},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

    def chat_voice(self, audio_path: str, session_id: str = "default") -> tuple[str, str]:
        """语音对话：听 → 想(带记忆) → 说。返回 (文字回复, 音频路径)"""
        text = self.listen(audio_path)
        reply = self.chat_text(text, session_id=session_id)
        audio_out = f"reply_{os.getpid()}.mp3"
        self.speak(reply, output_path=audio_out)
        return reply, audio_out

    def chat_image(
        self,
        image_source: str,
        question: str,
        session_id: str = "default",
        voice_reply: bool = False
    ) -> str | tuple:
        """图片问答（可选带记忆和语音输出）"""
        if session_id and voice_reply:
            # 带记忆的多模态对话
            history = self._get_session_history(session_id)

            if os.path.exists(image_source):
                url = self._encode_image(image_source)
            else:
                url = image_source

            message = HumanMessage(content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": url}}
            ])

            history.add_message(message)
            response = self.base_chain.invoke({
                "input": [message],
                "chat_history": history.messages[:-1]   # 排除刚加入的消息
            })
            history.add_message(AIMessage(content=response.content))

            reply = response.content
            if voice_reply:
                audio = self.speak(reply)
                return reply, audio
            return reply
        else:
            # 不需要记忆的简单图片问答
            reply = self.see(image_source, question)
            if voice_reply:
                audio = self.speak(reply)
                return reply, audio
            return reply
```

注意 `chat_image` 方法中的两种模式：
- **简单模式**（无 session）：直接把图片+问题发给模型，不做记忆管理。适合一次性的图片分析任务。
- **完整模式**（有 session）：先把图片消息加入会话历史，再调用带记忆的 Chain，最后保存回复到历史中。适合连续的多轮图片对话场景。

## 第二步：交互式主程序

```python
def main():
    assistant = MultimodalAssistant(
        model_name="gpt-4o",
        system_prompt="你是一个友好的多模态助手。回答简洁有用。",
        tts_voice="alloy"
    )
    current_session = "default"

    print("=" * 56)
    print("   🤖👁️🎤 多模态助手 (支持文字/图片/语音)")
    print("=" * 56)
    print("  /img <path> <question>     发送图片提问")
    print("  /voice <path>              发送语音问题")
    print("  /tts on/off               开关语音回复")
    print("  /session <name>            切换会话")
    print("  /exit                      退出")
    print("=" * 56)

    voice_mode = False

    while True:
        try:
            user_input = input("\n❓ 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # 处理命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd in ["/exit", "/quit"]:
                break
            elif cmd == "/tts":
                voice_mode = len(parts) > 1 and parts[1].lower() == "on"
                print(f"🔊 语音回复: {'开启' if voice_mode else '关闭'}")
            elif cmd == "/session":
                if len(parts) > 1:
                    current_session = parts[1].strip()
                    print(f"✅ 切换到会话: {current_session}")
                else:
                    print(f"📌 当前会话: {current_session}")
            elif cmd == "/img" and len(parts) >= 3:
                img_path = parts[1]
                question = parts[2]
                print(f"\n🖼️  分析图片: {img_path}")
                try:
                    if voice_mode:
                        reply, audio = assistant.chat_image(
                            img_path, question, current_session, voice_reply=True
                        )
                        print(f"\n💬 {reply}")
                        print(f"🔊 语音已生成: {audio}")
                    else:
                        reply = assistant.chat_image(img_path, question, current_session)
                        print(f"\n💬 {reply}")
                except Exception as e:
                    print(f"⚠️  错误: {e}")
            elif cmd == "/voice" and len(parts) >= 2:
                audio_path = parts[1]
                print(f"\n🎤 处理语音: {audio_path}")
                try:
                    reply, audio = assistant.chat_voice(audio_path, current_session)
                    print(f"\n🎤 用户说: {assistant.listen(audio_path)}")
                    print(f"💬 {reply}")
                    if voice_mode:
                        print(f"🔊 语音已生成: {audio}")
                except Exception as e:
                    print(f"⚠️  错误: {e}")
            else:
                print('❓ 输入 /help 查看命令')
            continue

        # 普通文字对话
        try:
            reply = assistant.chat_text(user_input, current_session)
            if voice_mode:
                audio = assistant.speak(reply)
                print(f"\n💬 {reply}\n🔊 语音: {audio}")
            else:
                print(f"\n💬 {reply}")
        except Exception as e:
            print(f"⚠️  错误: {e}")

    print("\n👋 再见！")


if __name__ == "__main__":
    main()
```

## 第三步：运行效果演示

启动后，你可以尝试以下几种交互：

### 场景一：纯文字对话（带记忆）

```
❓ 你: 我叫小明

💬 你好，小明！很高兴认识你。

❓ 你: 我在学Python

💬 Python 是一门很好的入门语言！有什么具体想了解的吗？
```

### 场景二：发送图片提问

```
❓ you: /img photos/error_screenshot.png 这是什么错误？怎么修？

🖼️  分析图片: photos/error_screenshot.png

💬 这是 TypeError，发生在第12行。
原因是你尝试用 + 号连接整数和字符串。
修复方法：使用 f-string 或 str() 转换。
```

### 场景三：语音输入 + 语音回复

```
❓ you: /tts on
🔊 语音回复: 开启

❓ you: /voice recordings/question.mp3

🎤  处理语音: recordings/question.mp3
🎤 用户说: 帮我用Python写一个装饰器的例子
💬 这是一个计时装饰器示例：
```python
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.time()-start:.4f}s")
        return result
    return wrapper
```
🔊 语音已生成: reply_12345.mp3
```

### 场景四：混合模式——发语音问图片相关的问题

```
❓ you: /voice recordings/ask_about_chart.mp3

🎤  处理语音: recordings/ask_about_chart.mp3
🎤 用户说: 请看一下我刚才发的那个图表，Q3的数据怎么样？

💬 根据之前讨论的图表，Q3 的数据显示...
（模型回忆起了该会话中之前的图片内容）
```

这就是多模态 + 记忆组合后的威力——**用户可以随时切换输入方式（打字/说话/发图），而助手始终保持着完整的上下文**。

## 第四步：项目结构总结

```
multimodal-assistant/
├── .env                          # OPENAI_API_KEY
├── .gitignore
├── requirements.txt
├── multimodal_assistant.py       # 主程序（MultimodalAssistant 类 + CLI）
├── photos/                       # 测试图片目录
│   ├── error_screenshot.png
│   └── sales_chart.png
└── recordings/                   # 录音文件目录
    └── test_question.mp3
```

`requirements.txt`：

```
langchain>=0.3
langchain-openai>=0.2
langchain-core>=0.3
openai>=1.20
python-dotenv>=1.0
Pillow>=10.0          # 图片缩放（可选）
sounddevice>=0.5      # 录音功能（可选）
scipy>=1.12           # WAV 文件保存（可选）
```

## 扩展方向

完成基础版后，有几个值得探索的扩展方向：

**方向一：Web 界面集成**

上面的例子是命令行交互。在实际产品中，你需要一个 Web 界面来让用户上传图片、录制语音、播放回复音频。可以用 Streamlit 快速搭建原型：

```python
import streamlit as st

st.title("🤖 多模态助手")

tab1, tab2, tab3 = st.tabs(["💬 文字", "🖼️ 图片", "🎤 语音"])

with tab1:
    user_text = st.text_input("输入你的问题")
    if st.button("发送"):
        reply = assistant.chat_text(user_text)
        st.write(reply)

with tab2:
    uploaded = st.file_uploader("上传图片", type=["jpg", "png"])
    question = st.text_input("关于这张图片的问题")
    if uploaded and question and st.button("分析"):
        temp_path = f"temp_{uploaded.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getvalue())
        reply = assistant.chat_image(temp_path, question)
        st.write(reply)

with tab3:
    audio_input = st.audio_input("录音或上传音频")
    if audio_input and st.button("识别并回答"):
        temp_audio = "temp_voice.wav"
        with open(temp_audio, "wb") as f:
            f.write(audio_input.getvalue())
        reply, audio_out = assistant.chat_voice(temp_audio)
        st.write(reply)
        st.audio(audio_out)
```

**方向二：本地化部署降低成本**

如果对 API 成本敏感，可以考虑用开源替代方案：

```python
# 本地 Whisper（免费）
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3")

# 本地 TTS（免费）
from TTS.api import TTS
tts = TTS(model_name="tts_models/zh/baker/tacotron-DDC-GST")
tts.tts_to_file(text="你好世界", file_path="output.wav")

# 本地多模态模型（通过 Ollama）
# ollama run llava      # 图像理解
# ollama run whisper    # 语音识别
```

这些本地方案零 API 成本，但质量和速度通常不如 OpenAI 的云端服务。适合数据隐私要求高或预算有限的场景。

到这里，第六章的全部内容就结束了。我们从多模态的基本概念出发，深入学习了图像理解的多种实战用法（描述/OCR/图表分析/代码诊断），掌握了语音交互的双向链路（STT + TTS），最终把它们与记忆能力整合在一起，构建了一个功能完整的**多模态智能助手**。这个助手已经具备了一个现代 AI 产品所需的核心交互能力——接下来的章节我们将学习更高级的主题：Chain 链式编排、Agent 智能体等，让应用从"被动应答"进化为"主动思考和行动"。
