---
title: 语音交互：语音转文字与文字转语音
description: OpenAI Whisper STT、TTS 文字转语音、LangChain 集成与完整对话流程
---
# 语音交互：语音转文字与文字转语音

上一节我们让 AI 助手学会了"看"图片。这一节我们让它学会"听"和"说"——通过 **STT（Speech-to-Text，语音转文字）** 和 **TTS（Text-to-Speech，文字转语音）** 技术，实现完整的语音交互能力。

## 为什么需要语音交互

在很多人机交互场景中，语音比文字更自然：

- **驾驶场景**：司机无法打字，只能用语音
- **智能家居**：对着空气说话比掏出手机打字方便得多
- **无障碍访问**：视障用户依赖语音交互来使用数字服务
- **效率场景**：说话的速度（每分钟约 150 字）远快于打字（平均每分钟 40-60 字）

一个完整的语音交互流程是这样的：

```
用户说话 → [麦克风录音] → 音频文件 → [STT] → 文本 → [LLM] → 回复文本 → [TTS] → 音频 → [扬声器播放]
```

这条链路中有两个关键环节——STT 和 TTS。我们分别来看。

## STT：语音转文字

OpenAI 提供了 **Whisper** 模型来做语音识别。它是目前最先进的开源语音识别模型之一，支持超过 50 种语言，对中文的识别效果尤其出色。

### 基础用法

```python
from openai import OpenAI

client = OpenAI()

with open("audio/user_voice.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="zh"   # 可选：指定语言以提升准确率
    )

print(transcript.text)
# 你好，我想知道今天北京的天气怎么样
```

就这么简单——把音频文件传给 `transcriptions.create()`，返回的就是识别出的文本。Whisper 支持常见的音频格式：MP3、M4A、WAV、FLAC、OGG 等。

### 支持的音频格式与限制

| 格式 | 支持 | 推荐度 |
|------|------|--------|
| MP3 | ✅ | ⭐⭐⭐ 兼容性好 |
| M4A | ✅ | ⭐⭐⭐ iPhone 录音默认格式 |
| WAV | ✅ | ⭐⭐ 无损但文件大 |
| FLAC | ✅ | ⭐ 无损压缩 |
| OGG | ✅ | ⭐ 开源格式 |

限制条件：
- 文件大小不超过 **25 MB**
- 音频时长建议不超过 **10 分钟**（更长的音频可以先分段处理）
- 采样率推荐 **16kHz 或以上**

### 在 LangChain 中集成 STT

把 STT 能力封装成一个 LangChain 可以使用的工具函数：

```python
import os
from openai import OpenAI

class SpeechToText:
    """语音转文字封装"""

    def __init__(self, model: str = "whisper-1"):
        self.client = OpenAI()
        self.model = model

    def transcribe(self, audio_path: str, language: str = "zh") -> str:
        """
        将音频文件转换为文本
        
        Args:
            audio_path: 音频文件路径
            language: 语言代码（zh=中文, en=英文, ja=日文等）
        
        Returns:
            识别出的文本内容
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        file_size = os.path.getsize(audio_path)
        if file_size > 25 * 1024 * 1024:   # 25MB 限制
            raise ValueError(f"文件过大 ({file_size/1024/1024:.1f}MB)，上限 25MB")

        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                language=language,
                response_format="text"
            )
        
        return response.text


# 使用示例
stt = SpeechToText()
text = stt.transcribe("recordings/question.mp3")
print(f"识别结果: {text}")
# 识别结果: 帮我用Python写一个快速排序算法
```

### 实际录制音频的方法

在开发测试阶段，你需要一些音频样本来验证 STT 效果。最简单的方式是用 Python 直接录制：

```python
import sounddevice as sd
import scipy.io.wavfile as wav
import os

def record_audio(duration: int = 5, output_path: str = "recording.wav", sample_rate: int = 16000):
    """
    录制一段音频并保存为 WAV 文件
    
    Args:
        duration: 录制时长（秒）
        output_path: 输出文件路径
        sample_rate: 采样率（16000 适合语音）
    """
    print(f"🎤 开始录制 ({duration}秒)...")
    print("请现在开始说话")
    
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()   # 等待录制完成
    
    wav.write(output_path, sample_rate, recording)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ 录制完成！保存到 {output_path} ({size_kb:.0f} KB)")
    return output_path


# 使用
audio_file = record_audio(duration=5, output_path="test_voice.wav")

# 立即转录
stt = SpeechToText()
text = stt.transcribe(audio_file)
print(f"\n📝 转录结果: {text}")
```

> **安装依赖**：`pip install sounddevice scipy`。`sounddevice` 用于录音，`scipy` 用于保存 WAV 文件。

## TTS：文字转语音

OpenAI 还提供了 TTS（Text-to-Speech）服务，能把文本转换成自然的语音输出：

```python
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",       # 合成声音风格
    input="你好！我是你的 AI 助手，很高兴为你服务。"
)

# 保存为音频文件
output_path = "output_response.mp3"
response.stream_to_file(output_path)
print(f"✅ 语音已保存到 {output_path}")
```

双击播放这个 MP3 文件，你会听到一段清晰自然的中文语音。

### 声音风格选择

OpenAI TTS 提供 6 种预置声音：

| 声音 | 特点 | 适用场景 |
|------|------|---------|
| `alloy` | 中性平衡 | 通用场景（默认推荐） |
| `echo` | 低沉回声感 | 新闻播报、纪录片 |
| `fable` | 温暖叙事感 | 有声书、故事讲述 |
| `onyx` | 深沉有力 | 正式场合、商务 |
| `nova` | 活泼明亮 | 年轻化产品、助手 |
| `shimmer` | 柔和轻快 | 轻松场景 |

你可以试听每种声音后选择最适合你应用风格的那个。

### TTS 封装类

```python
import os
from openai import OpenAI

class TextToSpeech:
    """文字转语音封装"""

    def __init__(self, model: str = "tts-1", voice: str = "alloy"):
        self.client = OpenAI()
        self.model = model
        self.voice = voice

    def speak(self, text: str, output_path: str = "response.mp3") -> str:
        """
        将文本转换为语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频文件路径
        
        Returns:
            生成的音频文件路径
        """
        # TTS 对单次输入长度有限制（约 4096 字符）
        if len(text) > 4000:
            text = text[:4000] + "（内容过长，已截断）"

        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )

        response.stream_to_file(output_path)
        
        size_kb = os.path.getsize(output_path) / 1024
        print(f"🔊 语音已生成: {output_path} ({size_kb:.0f} KB)")
        return output_path


# 使用示例
tts = TextToSpeech(voice="nova")   # 用活泼的声音
tts.speak("今天天气真不错，适合出去走走！", output_path="greeting.mp3")
```

### TTS-1 vs TTS-1-HD

OpenAI 提供两个版本的 TTS 模型：

| 模型 | 质量 | 速度 | 成本 |
|------|------|------|------|
| `tts-1` | 标准 | 快 | 低 |
| `tts-1-hd` | 高品质 | 稍慢 | 约 2 倍 |

对于大多数应用场景，`tts-1` 的质量已经足够好。如果你在做有声书或高端客服产品，可以升级到 `tts-1-hd` 获得更细腻的语音表现。

## 组装完整的语音对话链路

有了 STT 和 TTS，我们可以把它们和 LLM 串起来，形成一个完整的语音对话系统：

```python
"""
voice_assistant.py — 语音对话助手
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class VoiceAssistant:
    """带语音能力的对话助手"""

    def __init__(self, system_prompt: str = "你是一个简洁有用的助手"):
        # LLM
        chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        self.chain = self.prompt | chat

        # STT & TTS
        from openai import OpenAI
        self.openai_client = OpenAI()
        self.tts_voice = "nova"

    def listen(self, audio_path: str) -> str:
        """听：语音 → 文字"""
        with open(audio_path, "rb") as f:
            result = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="zh"
            )
        text = result.text
        print(f"🎤 用户说: {text}")
        return text

    def think(self, text: str) -> str:
        """想：文字 → LLM → 回复"""
        response = self.chain.invoke({"input": text})
        reply = response.content
        print(f"🤖 助手回复: {reply}")
        return reply

    def speak(self, text: str, output_path: str = "reply.mp3") -> str:
        """说：文字 → 语音"""
        resp = self.openai_client.audio.speech.create(
            model="tts-1",
            voice=self.tts_voice,
            input=text
        )
        resp.stream_to_file(output_path)
        print(f"🔊 语音已保存: {output_path}")
        return output_path

    def converse(self, audio_path: str) -> str:
        """完整的一次对话：听 → 想 → 说"""
        text = self.listen(audio_path)
        reply = self.think(text)
        audio_out = self.speak(reply)
        return audio_out


# 使用示例
assistant = VoiceAssistant(system_prompt="你是一个Python编程助教，回答要简短")

# 模拟：用户发送了一段语音问题
output = assistant.converse("recordings/python_question.mp3")
# 🎤 用户说: 装饰器是什么意思？
# 🤖 助手回复: 装饰器是一种语法糖，用于在不修改原函数代码的前提下扩展其功能。
# 🔊 语音已保存: reply.mp3
```

整个流程只有三个方法调用——`listen()`、`think()`、`speak()`——每个方法的职责清晰单一，非常容易理解和维护。

## 流式 TTS：边生成边播放

上面的例子中，TTS 是先生成完整个音频文件再保存。如果回复很长（比如几百字的详细解释），用户可能要等好几秒才能开始听到声音。**流式 TTS** 可以解决这个问题——它一边生成一边推送音频数据，实现几乎实时的语音输出：

```python
def stream_speak(self, text: str, output_path: str = "stream_reply.mp3"):
    """流式 TTS：边生成边写入文件"""
    with self.openai_client.audio.speech.create(
        model="tts-1",
        voice=self.tts_voice,
        input=text,
        response_format="mp3"
    ) as response:
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=1024):
                f.write(chunk)
                # 实时播放的逻辑可以在这里加入
```

`response.iter_bytes()` 让你可以逐块读取生成的音频数据。在 Web 应用中，这些 chunk 可以实时推送到前端浏览器播放；在桌面应用中，可以用 `pygame` 或 `pyaudio` 实现边接收边播放。

## 成本参考

语音 API 的成本比文本 API 高不少，了解价格有助于做技术选型：

| 服务 | 模型 | 价格 |
|------|------|------|
| STT (Whisper) | whisper-1 | $0.006 / 分钟 |
| TTS (标准) | tts-1 | $15 / 百万字符 |
| TTS (高清) | tts-1-hd | $30 / 百万字符 |

对比一下：一次普通的文本问答大约花费 $0.0001-0.001（取决于 token 数），而一次 30 秒的语音交互（STT + LLM + TTS）总成本大约在 $0.01-0.03 左右——贵了约 30-100 倍。所以在设计产品功能时，应该让用户**自主选择是否开启语音模式**，默认使用文字交互。

下一节我们将把视觉理解（图像）和语音交互整合在一起，构建一个完整的多模态助手——它能看图、能听懂语音、还能用语音回答。
