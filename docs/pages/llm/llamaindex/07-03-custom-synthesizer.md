---
title: 自定义 Synthesizer 与 Prompt 工程
description: 完全控制合成流程、Prompt 模板定制、多阶段合成管道、领域适配与风格控制
---
# 自定义 Synthesizer 与 Prompt 工程

前两节我们学习了响应合成的挑战和四种内置模式。但内置模式的 Prompt 模板是通用的——它们不知道你的应用是法律咨询还是技术支持，不知道用户偏好简洁回答还是详细解释，也不知道答案应该用什么语言风格。

这一节我们来学习如何通过**自定义 Synthesizer 和 Prompt 工程**来精确控制合成的行为和输出质量。这是让 RAG 系统从"能用"到"好用"的关键一步。

## 为什么需要自定义 Prompt？

让我们看看 LlamaIndex 默认的 REFINE 模式使用的 Prompt（简化版）：

```
System Prompt:
你是一个有帮助的助手，请根据上下文信息回答问题。
如果你不知道答案，就说不知道。

Initial Query Prompt:
上下文信息:
{context_str}

问题:
{query_str}

Refine Prompt:
我们有一个关于'{query_str}'的现有答案：
'{existing_answer}'
以下是新的参考信息（可能有助于改进答案）：
'{context_msg}'
如果新信息有用，请精炼原有答案；如果没有用，保持不变。
```

这个默认 Prompt 的问题在于：

**问题一：缺少领域约束。** 它没有告诉 LLM "你是法律顾问"或"你是技术支持工程师"，导致回答风格可能不符合业务场景的需求。

**问题二：没有格式要求。** 它没有要求输出 Markdown 表格、JSON 结构或特定的分段方式。

**问题三：没有安全边界。** 它没有明确禁止某些行为（如编造文档中没有的信息、泄露内部系统细节等）。

**问题四：没有引用规范。** 它没有要求在答案中标注信息来源，导致用户无法追溯。

通过自定义 Prompt，我们可以解决所有这些问题。

## 基础自定义：替换 Prompt 模板

```python
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts.prompt_template import PromptTemplate
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

index = VectorStoreIndex.from_documents(documents)

text_qa_template = PromptTemplate(
    "你是公司智能客服助手'小智'，专门负责解答客户关于产品和服务的问题。\n\n"
    "【重要规则】\n"
    "1. 只使用以下参考资料中的信息来回答问题，不要编造任何内容\n"
    "2. 如果参考资料中没有相关信息，请诚实告知并建议联系人工客服\n"
    "3. 回答时请在适当位置标注来源，格式为 (来源: 文件名)\n"
    "4. 使用友好、专业的语气，避免过于口语化或过于正式\n"
    "5. 如果涉及操作步骤，请用编号列表清晰列出\n\n"
    "参考资料:\n{context_str}\n\n"
    "客户问题:\n{query_str}\n"
)

refine_template = PromptTemplate(
    "我们有一个关于'{query_str}'的现有回答：\n"
    "'{existing_answer}'\n\n"
    "以下是从知识库中检索到的新的参考信息：\n"
    "'{context_msg}'\n\n"
    "请基于新信息精炼原有回答。具体要求：\n"
    "- 如果新信息与已有内容矛盾，以新信息为准并注明变化\n"
    "- 如果新信息补充了有用细节，将其自然地融入回答中\n"
    "- 保持原有的友好语气和格式规范\n"
    "- 更新来源标注（如有）\n"
    "- 如果新信息没有帮助，保持原样不变\n"
)

synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    text_qa_template=text_qa_template,
    refine_template=refine_template,
)

query_engine = index.as_query_engine(response_synthesizer=synthesizer)
response = query_engine.query("怎么申请退款？")
print(response.response)
```

### 自定义前后的效果对比

**使用默认 Prompt 的输出：**
```
根据文档，您可以登录账户进入订单管理页面，
点击申请退款按钮。退款将在3-5个工作日内退回。
```

**使用自定义 Prompt 的输出：**
```
您好！关于退款申请的问题，我为您整理了以下信息 (来源: FAQ.md)：

📋 **退款条件**
• 支持购买后30天内无理由退款 (来源: refund_policy.pdf)
• 定制类产品（如刻字、特殊配色）不适用此政策 (来源: refund_policy.pdf)

🔄 **操作步骤** (来源: user_guide.md)
1. 登录您的账户 → 进入「我的订单」页面
2. 找到需要退款的订单 → 点击「申请退款」按钮
3. 选择退款原因（必填）→ 提交审核
4. 审核通过后 3-5 个工作日内原路退回

💡 **温馨提示**
如果您购买的是定制版产品或有其他特殊情况，
建议先联系人工客服（400-xxx-xxxx）确认是否适用无理由退款政策。
```

差异是显而易见的——自定义 Prompt 让输出从"一段平铺直叙的文字"变成了"结构清晰、有来源标注、有温馨提示的专业回复"。

## 高级技巧：条件化 Prompt

有时候你想根据查询类型动态调整 Prompt 策略：

```python
from typing import List
from llama_index.core.schema import TextNode


class AdaptivePromptSynthesizer(BaseResponseSynthesizer):
    """根据查询特征自适应选择 Prompt 的 Synthesizer"""

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or Settings.llm

        self.templates = {
            "factual": PromptTemplate(
                "请严格基于资料回答，给出精确的事实性答案。\n\n"
                "资料:{context_str}\n问题:{query_str}"
            ),
            "procedural": PromptTemplate(
                "请按步骤详细说明操作方法。\n\n"
                "资料:{context_str}\n问题:{query_str}"
            ),
            "comparative": PromptTemplate(
                "请从多个角度比较分析。\n\n"
                "资料:{context_str}\n问题:{query_str}"
            ),
        }

    def _classify_query(self, query: str) -> str:
        """简单分类查询类型"""
        factual_keywords = ["多少", "什么", "是否", "有没有", "价格", "时间"]
        procedural_keywords = ["怎么", "如何", "步骤", "流程", "操作"]
        comparative_keywords = ["比较", "区别", "差异", "哪个好", "优缺点"]

        if any(k in query for k in factual_keywords):
            return "factual"
        elif any(k in query for k in procedural_keywords):
            return "procedural"
        elif any(k in query for k in comparative_keywords):
            return "comparative"
        return "factual"  # 默认

    def synthesize(self, query, nodes, **kwargs):
        qtype = self._classify_query(query)
        template = self.templates[qtype]

        context = "\n\n".join([f"[{i+1}] {n.text}" for i, n in enumerate(nodes)])
        prompt = template.format(context_str=context, query_str=query)

        response_text = self.llm.complete(prompt).text

        return Response(
            response=response_text,
            source_nodes=[NodeWithScore(node=n) for n in nodes],
            metadata={"query_type": qtype},
        )
```

这种"先判断再选择"的模式在生产环境中非常实用——不同类型的查询确实需要不同的处理策略。

## 多阶段合成管道

对于特别复杂的合成需求，你可以构建一个多阶段的处理管道：

```python
class MultiStageSynthesizer(BaseResponseSynthesizer):
    """多阶段合成：草稿 → 审核 → 优化 → 格式化"""

    def __init__(self, llm=None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm or Settings.llm

    def synthesize(self, query, nodes, **kwargs):
        context = self._build_context(nodes)

        # Stage 1: 草稿生成
        draft_prompt = (
            f"基于以下资料生成一份完整的回答草稿。\n"
            f"不需要完美，但尽量覆盖所有关键点。\n\n"
            f"资料:\n{context}\n\n问题: {query}"
        )
        draft = self.llm.complete(draft_prompt).text
        print(f"[Stage 1] 草稿: {len(draft)} 字符")

        # Stage 2: 自检与修正
        review_prompt = (
            f"以下是针对问题'{query}'的一份回答草稿：\n{draft}\n\n"
            f"原始参考资料:\n{context}\n\n"
            f"请审核这份草稿：\n"
            f"1. 是否所有事实都来自参考资料？（标记任何编造的内容）\n"
            f"2. 是否遗漏了重要信息？\n"
            f"3. 是否有逻辑矛盾或不一致之处？\n"
            f"4. 输出修改后的版本，并附上修改说明。"
        )
        reviewed = self.llm.complete(review_prompt).text
        print(f"[Stage 2] 审核完成: {len(reviewed)} 字符")

        # Stage 3: 格式化输出
        format_prompt = (
            f"将以下回答转换为最终输出格式。\n\n"
            f"{reviewed}\n\n"
            f"输出要求:\n"
            f"- 使用 Markdown 格式\n"
            f"- 关键信息用加粗标注\n"
            f"- 操作步骤用有序列表\n"
            f"- 在每段重要信息后标注(来源: 文件名)\n"
            f"- 开头给出一句简短总结"
        )
        final = self.llm.complete(format_prompt).text
        print(f"[Stage 3] 最终版本: {len(final)} 字符")

        return Response(
            response=final,
            source_nodes=[NodeWithScore(node=n) for n in nodes],
            metadata={"stages": 3},
        )
```

这个三阶段管道虽然成本较高（3 次 LLM 调用），但对于**高质量要求的场景**（如法律意见书、医疗建议、金融报告等）能显著提升输出质量。

## 领域适配示例：不同行业的 Prompt

### 法律行业

```python
legal_template = PromptTemplate(
    "您是一位资深法律顾问。请基于提供的法规和政策文件回答法律咨询。\n\n"
    "【职业准则】\n"
    "- 仅依据提供的法律法规文本进行分析\n"
    "- 明确区分'法律规定'和'实务建议'（后者应标注'仅供参考'）\n"
    "- 引用时注明具体的法条名称和条款号\n"
    "- 对于模糊或存在争议的法律问题，说明不同观点\n"
    "- 如资料不足以得出结论，明确说明并建议咨询专业律师\n\n"
    "法规资料:\n{context_str}\n\n"
    "法律咨询:\n{query_str}"
)
```

### 技术支持

```python
tech_support_template = PromptTemplate(
    "你是公司的技术支持工程师。帮助用户解决产品使用中的技术问题。\n\n"
    "【回答规范】\n"
    "- 先确认用户的产品型号和版本号\n"
    "- 操作步骤要具体到菜单路径和按钮名称\n"
    "- 给出命令行/代码示例时要完整可运行\n"
    "- 区分'正常现象'和'故障现象'\n"
    "- 提供排查思路而非直接给答案（引导用户自主解决）\n"
    "- 无法解决时提供升级渠道（工单/电话）\n\n"
    "技术文档:\n{context_str}\n\n"
    "用户问题:\n{query_str}"
)
```

### 医疗健康

```python
healthcare_template = PromptTemplate(
    "您是一位医疗健康顾问助手。请基于提供的医学资料回答健康相关问题。\n\n"
    "【⚠️ 重要免责声明】\n"
    "- 以下回答不能替代专业医生的诊断和治疗建议\n"
    "- 如涉及症状描述，请强调'请及时就医' \n"
    "- 不提供处方药推荐或剂量建议\n"
    "- 引用的医学信息应标注数据来源和发布日期\n"
    "- 对紧急情况提供急救指导并强烈建议拨打急救电话\n\n"
    "医学参考资料:\n{context_str}\n\n"
    "健康咨询:\n{query_str}"
)
```

## 常见误区

**误区一:"Prompt 越长越好。"** 不是的。过长的 Prompt 会：占用宝贵的上下文窗口空间、增加 LLM 处理时间、可能引入相互矛盾的指令（指令越多越容易冲突）。**好的 Prompt 应该精炼且聚焦——每个指令都应该有明确的目的。**

**误区二:"Prompt 写好后就不需要改了。"** Prompt 的效果会随着 LLM 模型的更新而变化（新模型可能对不同的指令格式更敏感），也会随着用户反馈的积累而发现可优化的点。**把 Prompt 当作需要持续迭代的代码来管理，而不是一次性写完就封存。**

**误区三:"自定义 Synthesizer 很难实现。"** 从上一节的代码框架可以看到，最简单的自定义只需要实现 `synthesize()` 一个方法。真正复杂的是设计好的 Prompt 和处理逻辑——但这更多是领域知识和写作能力的问题，不是编程难题。
