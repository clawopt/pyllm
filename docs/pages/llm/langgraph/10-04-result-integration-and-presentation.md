# 4. 结果整合与展示

## 从原始数据到洞察：让研究成果真正有用

经过前面三节的努力，DeepResearch 已经具备了完整的研究能力——它能解析用户查询、制定研究计划、通过 Reactor 模式自主调用多种工具进行多轮迭代搜索、提取结构化事实、生成和验证假设。但所有这些努力的最终产出，目前只是一份基础的 Markdown 报告，里面罗列了提取到的事实和来源链接。

这显然不够好。一个真正有价值的研究报告应该像人类专家写的那样——有清晰的逻辑主线、有数据支撑的论点、对不同观点的平衡讨论、有可操作的结论建议、以及专业的排版和引用格式。这一节我们要解决的问题就是：**如何把研究中积累的大量碎片化信息（几十个事实、十几个来源、若干假设），整合成一份高质量、易阅读、有深度的研究报告**。

### 报告生成的挑战与设计原则

在深入代码之前，让我们先想清楚"好的研究报告"到底长什么样，以及用 AI 自动生成这样的报告面临哪些挑战。

**挑战一：信息过载 vs 信息不足的矛盾。** 研究过程中可能收集了几十条事实和十几个来源，如果全部堆进报告里会变成一篇没人愿意读的长篇大论；但如果过度精简又可能丢失关键信息。解决方案是**分层呈现**——核心发现放在最前面（执行摘要），详细论证放在中间章节，原始数据和参考资料放在附录。

**挑战二：逻辑连贯性的保持。** LLM 生成的每个段落单独看可能都不错，但放在一起可能缺乏逻辑衔接——前后矛盾、重复论述、跳跃性太大。解决方案是**先规划大纲再逐章撰写**，确保每一节都有明确的目的且与其他章节形成有机整体。

**挑战三：准确性与可追溯性的保证。** AI 最容易犯的错误就是"幻觉"——编造不存在的事实或引用。对于研究报告来说这是致命的。解决方案是**强制引用绑定**——每一个论点都必须关联到至少一个已验证的事实/来源，不允许"凭空出现"的信息。

基于这些分析，我们确立以下报告生成的核心设计原则：

1. **结构驱动**：先生成大纲，再填充内容
2. **证据为本**：每段论述必须有事实支撑
3. **冲突透明**：不同观点要并列呈现
4. **引用完整**：所有结论都可追溯到原始来源
5. **格式专业**：支持 Markdown / HTML / PDF 多种输出

### 增强版报告生成器

让我们重新实现 `generate_report_node`，把它从一个简单的模板填充升级为一个智能的报告生成引擎。

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

REPORT_PLANNER_PROMPT = """你是一个资深的研究报告架构师。根据给定的研究数据和发现，设计一份高质量的研究报告大纲。

## 研究主题
{topic}

## 子问题及覆盖情况
{sub_questions_coverage}

## 核心发现（高置信度事实）
{key_findings}

## 已验证的假设
{hypotheses}

## 来源统计
- 总来源数: {source_count}
- 学术论文: {academic_count}
- 新闻报道: {news_count}
- 官方文档: {official_count}
- 其他: {other_count}

## 设计要求
请输出 JSON 格式的大纲：
{
    "title": "报告标题",
    "executive_summary_points": ["要点1", "要点2", ...],
    "sections": [
        {
            "title": "章节标题",
            "purpose": "本章目的",
            "key_arguments": ["论点1", "论点2"],
            "supporting_facts_indices": [0, 3, 7],
            "conflicts_to_address": ["需要讨论的分歧点"],
            "estimated_length": "short|medium|long"
        }
    ],
    "conclusion": {
        "main_takeaways": ["核心结论1", "核心结论2"],
        "limitations": "研究的局限性说明",
        "recommendations": ["建议1", "建议2"]
    }
}"""

report_planner_llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

@dataclass
class ReportSection:
    title: str
    purpose: str
    key_arguments: List[str]
    supporting_fact_indices: List[int]
    conflicts_to_address: List[str]
    estimated_length: str
    content: str = ""
    
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "purpose": self.purpose,
            "key_arguments": self.key_arguments,
            "supporting_fact_indices": self.supporting_fact_indices,
            "conflicts_to_address": self.conflicts_to_address,
            "estimated_length": self.estimated_length,
            "content": self.content
        }

def plan_report_structure(state: ResearchState) -> dict:
    topic = state["topic"]
    sub_questions = state["sub_questions"]
    coverage_map = state["coverage_map"]
    all_facts = state["extracted_facts"]
    all_sources = state["collected_sources"]
    hypotheses = state.get("active_hypotheses", [])
    
    sq_coverage_text = "\n".join(
        f"- {sq}: {coverage_map.get(sq, 0):.0%}"
        for sq in sub_questions
    )
    
    high_conf_facts = sorted(
        [f for f in all_facts if f.get("confidence", 0) >= 0.7],
        key=lambda x: x.get("confidence", 0),
        reverse=True
    )[:25]
    
    facts_text = "\n".join(
        f"[{i}] {f.get('subject', '?')} {f.get('predicate', '?')} {f.get('object_value', '?')} "
        f"(置信度:{f.get('confidence', 0):.0%}, 来源:{len(f.get('source_urls', []))}个)"
        for i, f in enumerate(high_conf_facts)
    )
    
    hypotheses_text = "\n".join(
        f"- '{h['statement']}' [{h['status']}, 置信度:{h['confidence']:.0%}]"
        for h in hypotheses if h.get("status") != "pending"
    ) or "(无已验证假设)"
    
    type_counts = {}
    for s in all_sources:
        stype = s.get("source_type", "other")
        type_counts[stype] = type_counts.get(stype, 0) + 1
    
    prompt = REPORT_PLANNER_PROMPT.format(
        topic=topic,
        sub_questions_coverage=sq_coverage_text,
        key_findings=facts_text,
        hypotheses=hypotheses_text,
        source_count=len(all_sources),
        academic_count=type_counts.get("academic_paper", 0),
        news_count=type_counts.get("news_article", 0),
        official_count=type_counts.get("official_doc", 0),
        other_count=len(all_sources) - sum(type_counts.get(k, 0) 
                     for k in ["academic_paper", "news_article", "official_doc"])
    )
    
    messages = [
        SystemMessage(content=REPORT_PLANNER_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = report_planner_llm.invoke(messages)
    
    try:
        import json
        plan = json.loads(response.content.strip())
    except:
        plan = generate_fallback_plan(topic, sub_questions, high_conf_facts)
    
    return plan

def generate_fallback_plan(topic: str, sub_questions: List[str], facts: List[Dict]) -> dict:
    sections = []
    for i, sq in enumerate(sub_questions):
        sections.append({
            "title": f"{i+1}. {sq}",
            "purpose": f"深入分析 {sq} 相关的关键发现",
            "key_arguments": [f.get("object_value", "")[:80] for f in facts[i*5:(i+1)*5]],
            "supporting_fact_indices": list(range(i*5, min((i+1)*5, len(facts)))),
            "conflicts_to_address": [],
            "estimated_length": "medium"
        })
    
    return {
        "title": f"深度研究报告：{topic}",
        "executive_summary_points": [
            f"本报告围绕 {topic} 进行了系统性研究",
            f"共收集 {len(facts)} 条关键事实",
            f"覆盖了 {len(sub_questions)} 个核心子问题"
        ],
        "sections": sections,
        "conclusion": {
            "main_takeaways": [f"关于 {topic} 的综合研究发现..."],
            "limitations": "本研究基于公开可获得的信息源...",
            "recommendations": ["建议进一步深入研究..."]
        }
    }
```

`plan_report_structure` 函数是报告生成的第一步——它把所有的研究数据（主题、子问题覆盖率、高置信度事实、已验证假设、来源类型分布）打包成一个精心设计的 prompt 发送给 LLM，让它输出一份结构化的报告大纲。这个大纲定义了报告的标题、执行摘要要点、各章节的目的和论点、每章应引用哪些事实索引、需要讨论哪些冲突点等。

注意其中的 `generate_fallback_plan` 函数——当 LLM 的输出无法解析为合法 JSON 时（这种情况在实际使用中并不罕见），我们用一个基于规则的 fallback 方案来确保报告生成不会完全失败。这种**优雅降级**的思想在生产系统中非常重要。

有了大纲之后，下一步是逐章生成内容：

```python
SECTION_WRITER_PROMPT = """你是一位资深的研究报告撰稿人。请根据以下信息撰写报告的一个章节。

## 章节要求
- 标题：{section_title}
- 目的：{section_purpose}
- 需要阐述的核心论点：{key_arguments}
- 需要讨论的争议/分歧点：{conflicts}

## 可用的证据材料
{evidence_material}

## 写作要求
1. 开头用 1-2 句话概括本章主旨
2. 每个论点都要有具体的事实或数据支撑
3. 如果存在争议，要客观地呈现各方观点
4. 使用清晰的小标题组织内容
5. 语言专业但不晦涩，适合目标读者理解
6. 段落之间要有逻辑过渡
7. 总长度控制在 {length_guide} 字左右

## 重要约束
- 只使用上面提供的证据材料，不要编造任何新信息
- 引用事实时使用 [来源N] 格式的上标标注
- 如果某个论点缺乏足够证据，明确指出"现有信息不足以充分论证此点"

直接输出 Markdown 格式的正文内容，不要包含章节标题（标题会由系统自动添加）。"""

section_writer_llm = ChatOpenAI_model="gpt-4o", temperature=0.5)

def write_section(section_plan: dict, all_facts: List[Dict], 
                  all_sources: List[Dict], fact_index_map: Dict[int, int]) -> str:
    fact_indices = section_plan.get("supporting_fact_indices", [])
    
    evidence_parts = []
    source_counter = {}
    
    for idx in fact_indices:
        actual_idx = fact_index_map.get(idx, idx)
        if actual_idx < len(all_facts):
            fact = all_facts[actual_idx]
            
            src_urls = fact.get("source_urls", [])
            source_nums = []
            for url in src_urls:
                if url not in source_counter:
                    source_counter[url] = len(source_counter) + 1
                source_nums.append(source_counter[url])
            
            evidence_parts.append(
                f"**事实[{idx+1}]**: {fact.get('subject', '')} "
                f"{fact.get('predicate', '')} {fact.get('object_value', '')} "
                f"(置信度: {fact.get('confidence', 0):.0%}) "
                f"来源: {','.join(f'[{n}]' for n in source_nums)}"
            )
    
    length_guide = {
        "short": "300-500",
        "medium": "500-800",
        "long": "800-1200"
    }.get(section_plan.get("estimated_length", "medium"), "500-800")
    
    prompt = SECTION_WRITER_PROMPT.format(
        section_title=section_plan["title"],
        section_purpose=section_plan.get("purpose", ""),
        key_arguments="\n".join(f"- {a}" for a in section_plan.get("key_arguments", [])),
        conflicts="\n".join(f"- {c}" for c in section_plan.get("conflicts_to_address", [])) or "(无)",
        evidence_material="\n\n".join(evidence_parts) or "(暂无直接证据)",
        length_guide=length_guide
    )
    
    messages = [
        SystemMessage(content=SECTION_WRITER_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = section_writer_llm.invoke(messages)
    return response.content.strip()

def generate_enhanced_report(state: ResearchState) -> str:
    plan = plan_report_structure(state)
    
    all_facts = state["extracted_facts"]
    all_sources = state["collected_sources"]
    
    high_conf_facts = sorted(
        [f for f in all_facts if f.get("confidence", 0) >= 0.5],
        key=lambda x: x.get("confidence", 0),
        reverse=True
    )
    
    fact_index_map = {i: idx for idx, f in enumerate(high_conf_facts)}
    
    report_lines = []
    
    report_lines.append(f"# {plan.get('title', state['topic'])}\n")
    report_lines.append(f"> **DeepResearch 自动生成报告** | "
                       f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    report_lines.append("---\n")
    report_lines.append("## 目录\n")
    for i, section in enumerate(plan.get("sections", [])):
        report_lines.append(f"{i+1}. [{section['title']}](#{slugify(section['title'])})\n")
    report_lines.append(f"{len(plan.get('sections', []))+1}. [结论与建议](#结论与建议)\n")
    report_lines.append(f"{len(plan.get('sections', []))+2}. [参考资料](#参考资料)\n")
    
    report_lines.append("\n---\n")
    report_lines.append("## 执行摘要\n")
    for point in plan.get("executive_summary_points", []):
        report_lines.append(f"- {point}\n")
    
    report_lines.append(f"\n> **研究概览**: 本报告基于 {len(all_sources)} 个独立信息源和 "
                       f"{len(all_facts)} 条关键事实，覆盖了 {len(state['sub_questions'])} 个核心研究维度。\n")
    
    for i, section_plan in enumerate(plan.get("sections", [])):
        slug = slugify(section_plan["title"])
        report_lines.append(f"\n---\n")
        report_lines.append(f"<a id='{slug}'></a>\n")
        report_lines.append(f"## {section_plan['title']}\n")
        
        content = write_section(section_plan, high_conf_facts, all_sources, fact_index_map)
        report_lines.append(content)
        report_lines.append("\n")
    
    conclusion = plan.get("conclusion", {})
    report_lines.append("\n---\n")
    report_lines.append("<a id='结论与建议'></a>\n")
    report_lines.append("## 结论与建议\n")
    
    report_lines.append("### 核心发现\n")
    for takeaway in conclusion.get("main_takeaways", []):
        report_lines.append(f"- {takeaway}\n")
    
    if conclusion.get("limitations"):
        report_lines.append("\n### 研究局限性\n")
        report_lines.append(f"{conclusion['limitations']}\n")
    
    if conclusion.get("recommendations"):
        report_lines.append("\n### 建议\n")
        for rec in conclusion.get("recommendations", []):
            report_lines.append(f"- {rec}\n")
    
    verified_hyps = [h for h in state.get("active_hypotheses", []) 
                    if h.get("status") in ["confirmed", "partial"]]
    if verified_hyps:
        report_lines.append("\n### 已验证的研究假设\n")
        for hyp in verified_hyps:
            status_icon = "✅" if hyp["status"] == "confirmed" else "🔶"
            report_lines.append(f"- {status_icon} **{hyp['statement']}** "
                               f"(置信度: {hyp['confidence']:.0%})\n")
            if hyp.get("supporting_evidence"):
                report_lines.append(f"  - 支持证据: {len(hyp['supporting_evidence'])} 条\n")
            if hyp.get("refuting_evidence"):
                report_lines.append(f"  - 反对证据: {len(hyp['refuting_evidence'])} 条\n")
    
    report_lines.append("\n---\n")
    report_lines.append("<a id='参考资料'></a>\n")
    report_lines.append("## 参考资料\n")
    
    source_number_map = {}
    seen_urls = set()
    
    for source in all_sources:
        url = source.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            num = len(source_number_map) + 1
            source_number_map[url] = num
            
            type_icon = {
                "academic_paper": "📄",
                "official_doc": "📋",
                "news_article": "📰",
                "wikipedia": "📖",
                "blog_post": "📝",
                "forum": "💬"
            }.get(source.get("source_type", ""), "📎")
            
            title = source.get("title", "Untitled")
            published = source.get("metadata", {}).get("published_date", "")
            date_str = f" ({published})" if published else ""
            
            report_lines.append(
                f"[{num}] {type_icon} [{title}]({url})"
                f" [{source.get('source_type', '?')}]{date_str}\n"
            )
    
    report_lines.append("\n---\n")
    report_lines.append(f"*本报告由 DeepResearch v2.0 自动生成 | "
                       f"研究引擎: LangGraph + GPT-4o | "
                       f"共 {state['task_context']['current_round']} 轮迭代*\n")
    
    return "\n".join(report_lines)

import re

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    return text.strip('-')
```

增强版报告生成器的流程分为三个阶段：

**阶段一：规划（plan_report_structure）**。LLM 基于全部研究数据生成一份详细的报告大纲，包含各章节的标题、目的、论点、证据索引、需讨论的冲突点、预估长度。

**阶段二：逐章撰写（write_section）**。对大纲中的每个章节，构建一个专门的 prompt，包含该章的目的、论点、可用证据材料（带编号的事实列表和来源编号）、写作要求和约束条件。然后调用 LLM 撰写该章内容。特别注意证据材料的处理方式——每个事实都被分配了一个唯一的编号，来源也有独立的编号体系，这样正文中就可以用 `[来源N]` 的方式精确引用。

**阶段三：组装（generate_enhanced_report）**。把所有部分组装成完整的 Markdown 文档——包括封面元信息、目录、执行摘要、各章节正文、结论与建议（含已验证假设）、参考资料清单、页脚声明。其中目录使用了锚点链接（`<a id>` + `#slug`），方便在 HTML 渲染后点击跳转。

### 数据可视化描述生成

纯文本的报告虽然信息量大，但有时候一张图表胜过千言万语。由于我们的系统运行在后端，无法直接渲染图形，但可以在报告中嵌入**图表描述**——告诉用户"这里应该有一张什么样的图"，并附上可用于绑定的数据。

```python
VISUALIZATION_GENERATOR_PROMPT = """根据以下研究数据，判断是否需要可视化图表。
如果需要，生成图表的具体规格描述和数据。

## 研究主题
{topic}

## 数值型事实（可用于制图）
{numeric_facts}

## 输出要求
如果适合制作图表，返回 JSON：
{
    "chart_type": "bar|line|pie|scatter|table|timeline",
    "title": "图表标题",
    "description": "图表说明了什么",
    "data": {
        "labels": [...],
        "values": [...],
        "series": [...]  // 用于多系列图表
    },
    "x_axis_label": "...",
    "y_axis_label": "...",
    "insight": "从图表中可以得出的关键洞察"
}
如果不适合制作图表，返回：{"chart_type": null}"""

viz_generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def extract_numeric_facts(facts: List[Dict]) -> List[Dict]:
    numeric = []
    import re
    
    for f in facts:
        obj_val = str(f.get("object_value", ""))
        numbers = re.findall(r'[\d,]+(?:\.\d+)?(?:%|亿|万|B|M|K)?', obj_val)
        if numbers:
            numeric.append({
                "subject": f.get("subject", ""),
                "predicate": f.get("predicate", ""),
                "value_text": obj_val,
                "numbers": numbers,
                "confidence": f.get("confidence", 0)
            })
    
    return numeric[:20]

def generate_visualizations(state: ResearchState) -> List[dict]:
    numeric_facts = extract_numeric_facts(state["extracted_facts"])
    
    if not numeric_facts:
        return []
    
    prompt = VISUALIZATION_GENERATOR_PROMPT.format(
        topic=state["topic"],
        numeric_facts="\n".join(
            f"- {nf['subject']} {nf['predicate']}: {nf['value_text']} (数值: {nf['numbers']})"
            for nf in numeric_facts
        )
    )
    
    messages = [
        SystemMessage(content=VISUALIZATION_GENERATOR_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = viz_generator_llm.invoke(messages)
    
    try:
        import json
        viz_spec = json.loads(response.content.strip())
    except:
        return []
    
    visualizations = []
    
    if viz_spec.get("chart_type"):
        visualizations.append(viz_spec)
    
    coverage_data = generate_coverage_chart_data(state)
    if coverage_data:
        visualizations.append(coverage_data)
    
    source_type_data = generate_source_type_chart_data(state)
    if source_type_data:
        visualizations.append(source_type_data)
    
    return visualizations

def generate_coverage_chart_data(state: ResearchState) -> dict:
    coverage = state["coverage_map"]
    if not coverage:
        return None
    
    labels = list(coverage.keys())
    values = [round(v * 100, 1) for v in coverage.values()]
    
    short_labels = [l[:20] + "..." if len(l) > 20 else l for l in labels]
    
    return {
        "chart_type": "horizontal_bar",
        "title": "各研究维度的信息覆盖率",
        "description": "展示了每个子问题的信息充足程度，100% 表示信息完全充分",
        "data": {"labels": short_labels, "values": values},
        "x_axis_label": "覆盖率 (%)",
        "y_axis_label": "研究维度",
        "insight": f"平均覆盖率: {sum(values)/len(values):.1f}%，"
                   f"最薄弱维度: {labels[values.index(min(values))]}"
    }

def generate_source_type_chart_data(state: ResearchState) -> dict:
    sources = state["collected_sources"]
    if not sources:
        return None
    
    type_counts = {}
    for s in sources:
        stype = s.get("source_type", "other")
        type_name = {
            "academic_paper": "学术论文",
            "official_doc": "官方文档",
            "news_article": "新闻报道",
            "wikipedia": "维基百科",
            "blog_post": "博客文章",
            "forum": "论坛讨论"
        }.get(stype, stype)
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
    
    return {
        "chart_type": "pie",
        "title": "信息来源类型分布",
        "description": "展示了各类信息源的数量占比，反映研究的多样性",
        "data": {"labels": list(type_counts.keys()), "values": list(type_counts.values())},
        "insight": f"共 {len(sources)} 个来源，涵盖 {len(type_counts)} 种类型"
    }

def insert_visualizations_into_report(markdown_report: str, visualizations: List[dict]) -> str:
    if not visualizations:
        return markdown_report
    
    viz_section = "\n## 📊 数据可视化\n\n"
    
    for i, viz in enumerate(visualizations):
        chart_type = viz.get("chart_type", "")
        title = viz.get("title", f"图表 {i+1}")
        description = viz.get("description", "")
        data = viz.get("data", {})
        insight = viz.get("insight", "")
        
        viz_section += f"### {title}\n\n"
        viz_section += f"{description}\n\n"
        
        if chart_type == "table":
            viz_section += render_table_from_data(data)
        elif chart_type in ("bar", "horizontal_bar", "line", "pie"):
            viz_section += render_ascii_chart(data, chart_type)
        
        if insight:
            viz_section += f"\n> 💡 **洞察**: {insight}\n"
        
        viz_section += f"\n*图表数据（供前端渲染使用）:* `{json.dumps(data, ensure_ascii=False)}`\n\n"
    
    conclusion_pos = markdown_report.find("## 结论与建议")
    if conclusion_pos > 0:
        return markdown_report[:conclusion_pos] + viz_section + "\n---\n" + markdown_report[conclusion_pos:]
    else:
        return markdown_report + "\n---\n" + viz_section

def render_table_from_data(data: dict) -> str:
    labels = data.get("labels", [])
    values = data.get("values", [])
    
    lines = ["| 指标 | 值 |", "|------|-----|"]
    for label, value in zip(labels, values):
        lines.append(f"| {label} | {value} |")
    
    return "\n".join(lines) + "\n"

def render_ascii_chart(data: dict, chart_type: str) -> str:
    labels = data.get("labels", [])
    values = data.get("values", [])
    
    if not values:
        return ""
    
    max_val = max(values) if values else 1
    bar_width = 30
    
    lines = []
    for label, val in zip(labels, values):
        bar_len = int(val / max_val * bar_width) if max_val > 0 else 0
        bar = "█" * bar_len
        lines.append(f"`{val:>8}` │{bar} {label}")
    
    return "\n".join(lines) + "\n"
```

这套可视化模块的工作原理是：

1. **自动识别数值型事实**：`extract_numeric_facts` 用正则表达式从所有事实中筛选出包含数字的那些——这些是可以用来做图表的数据基础。

2. **LLM 判断图表类型**：把数值型事实发给 LLM，让它决定最适合用什么类型的图表（柱状图/折线图/饼图/散点图/表格），并生成具体的图表规格（标题、坐标轴标签、数据映射）。

3. **内置标准图表**：除了 LLM 生成的自定义图表外，还自动生成两张标准图表——**信息覆盖率条形图**（展示各子问题的研究充分程度）和**来源类型饼图**（展示信息多样性）。这两张图几乎对所有研究报告都有价值。

4. **ASCII 艺术渲染**：由于我们输出的是 Markdown 文本，无法嵌入真正的图片，所以用 Unicode 字符（█）绘制 ASCII 柱状图。同时以 JSON 格式附带原始数据，前端可以据此渲染真正的交互式图表。

### 交互式研究回顾界面

研究报告不是终点——用户往往会在阅读报告后产生新的疑问："这个数据是从哪来的？""能不能再深入看一下这个问题？""有没有遗漏什么重要角度？"一个好的研究助手应该提供**交互式的回顾能力**，让用户能够探索研究的全过程。

```python
class ResearchSessionViewer:
    def __init__(self, state: ResearchState):
        self.state = state
        
    def get_full_timeline(self) -> list:
        timeline = []
        for log_entry in self.state.get("research_log", []):
            parsed = self._parse_log_entry(log_entry)
            timeline.append(parsed)
        return timeline
    
    def _parse_log_entry(self, entry: str) -> dict:
        entry_type = "info"
        if "[Reactor" in entry:
            entry_type = "reactor"
        elif "[Hypothesis" in entry:
            entry_type = "hypothesis"
        elif "[Verify" in entry:
            entry_type = "verify"
        elif "[DONE]" in entry or "完成" in entry:
            entry_type = "milestone"
        elif "评估完成" in entry:
            entry_type = "evaluation"
        elif "搜索完成" in entry or "提取完成" in entry:
            entry_type = "action"
        
        round_match = re.search(r'R?(\d+)', entry)
        round_num = int(round_match.group(1)) if round_match else 0
        
        return {
            "type": entry_type,
            "round": round_num,
            "content": entry,
            "timestamp_hint": f"Round {round_num}"
        }
    
    def get_fact_details(self, fact_index: int) -> dict:
        facts = self.state["extracted_facts"]
        if 0 <= fact_index < len(facts):
            fact = facts[fact_index]
            return {
                "index": fact_index,
                "subject": fact.get("subject", ""),
                "predicate": fact.get("predicate", ""),
                "object_value": fact.get("object_value", ""),
                "confidence": fact.get("confidence", 0),
                "sources": fact.get("source_urls", []),
                "context": fact.get("context", "")[:300],
                "extracted_round": fact.get("round_number", 0),
                "related_sub_question": fact.get("sub_question", "")
            }
        return {"error": "索引超出范围"}
    
    def get_source_details(self, source_index: int) -> dict:
        sources = self.state["collected_sources"]
        if 0 <= source_index < len(sources):
            src = sources[source_index]
            related_facts = [
                i for i, f in enumerate(self.state["extracted_facts"])
                if src.get("url") in f.get("source_urls", [])
            ]
            return {
                "index": source_index,
                "url": src.get("url", ""),
                "title": src.get("title", ""),
                "source_type": src.get("source_type", ""),
                "quality_score": src.get("quality_score", 0),
                "relevance_score": src.get("relevance_score", 0),
                "content_summary": src.get("content_summary", ""),
                "collected_in_round": src.get("collected_in_round", 0),
                "contributed_facts_count": len(related_facts),
                "contributed_fact_indices": related_facts[:10]
            }
        return {"error": "索引超出范围"}
    
    def trace_argument_chain(self, claim_text: str) -> list:
        """追踪一个论点的完整证据链"""
        chain = []
        facts = self.state["extracted_facts"]
        
        claim_lower = claim_text.lower()
        words = set(claim_lower.split())
        
        for i, fact in enumerate(facts):
            fact_text = f"{fact.get('subject','')} {fact.get('predicate','')} {fact.get('object_value','')}".lower()
            relevance = len(words & set(fact_text.split())) / max(len(words), 1)
            
            if relevance > 0.3:
                chain.append({
                    "fact_index": i,
                    "relevance": round(relevance, 2),
                    "fact_preview": (
                        f"{fact.get('subject','')} {fact.get('predicate','')} "
                        f"{fact.get('object_value','')[:100]}"
                    ),
                    "confidence": fact.get("confidence", 0),
                    "source_count": len(fact.get("source_urls", []))
                })
        
        chain.sort(key=lambda x: x["relevance"], reverse=True)
        return chain[:15]
    
    def get_statistics(self) -> dict:
        task_ctx = self.state["task_context"]
        facts = self.state["extracted_facts"]
        sources = self.state["collected_sources"]
        hypotheses = self.state.get("active_hypotheses", [])
        
        confidences = [f.get("confidence", 0) for f in facts]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        type_dist = {}
        for s in sources:
            stype = s.get("source_type", "other")
            type_dist[stype] = type_dist.get(stype, 0) + 1
        
        hypothesis_stats = {
            "total": len(hypotheses),
            "confirmed": sum(1 for h in hypotheses if h.get("status") == "confirmed"),
            "refuted": sum(1 for h in hypotheses if h.get("status") == "refuted"),
            "partial": sum(1 for h in hypotheses if h.get("status") == "partial"),
            "pending": sum(1 for h in hypotheses if h.get("status") == "pending")
        }
        
        return {
            "total_rounds": task_ctx.get("current_round", 0),
            "total_searches": task_ctx.get("total_searches_performed", 0),
            "total_sources": len(sources),
            "total_facts": len(facts),
            "avg_fact_confidence": round(avg_confidence, 3),
            "high_confidence_facts": sum(1 for c in confidences if c >= 0.8),
            "source_type_distribution": type_dist,
            "coverage_by_dimension": self.state.get("coverage_map", {}),
            "hypothesis_breakdown": hypothesis_stats,
            "estimated_cost_usd": task_ctx.get("estimated_cost_usd", 0),
            "token_usage": task_ctx.get("token_usage", {})
        }

viewer = None

def create_viewer_for_state(state: ResearchState) -> ResearchSessionViewer:
    global viewer
    viewer = ResearchSessionViewer(state)
    return viewer
```

`ResearchSessionViewer` 类提供了一套丰富的交互式查询 API：

- **`get_full_timeline()`**：返回完整的研究过程时间线，按时间顺序排列每一步操作（搜索、提取、评估、假设生成等），每条记录都有类型标记便于前端做不同的样式渲染。

- **`get_fact_details(index)`** 和 **`get_source_details(index)`**：查看任意一条事实或来源的详细信息，包括置信度、相关联的其他事实等。

- **`trace_argument_chain(claim_text)`**：这是最有意思的功能——给定一段论点陈述（比如从报告中复制的某句话），它能在所有已收集的事实中找到最相关的那些，形成一个**证据链**。这让用户可以验证报告中的任何论断是否有据可查。

- **`get_statistics()`**：返回整个研究过程的统计数据汇总——轮次、搜索次数、来源数、事实数、平均置信度、来源类型分布、覆盖率、假设状态、预估成本等。这些数据可以直接用于仪表盘展示。

### 报告质量校验

最后但同样重要的是，在报告输出之前进行质量检查，确保没有明显的错误或缺陷。

```python
class ReportQualityChecker:
    def __init__(self):
        self.checks = [
            self._check_factual_consistency,
            self._check_citation_completeness,
            self._check_coverage_adequacy,
            self._check_language_quality,
            self._check_length_appropriateness
        ]
    
    def validate(self, report: str, state: ResearchState) -> dict:
        results = []
        total_score = 0
        max_score = len(self.checks) * 20
        
        for check_fn in self.checks:
            result = check_fn(report, state)
            results.append(result)
            total_score += result["score"]
        
        overall_grade = self._calculate_grade(total_score, max_score)
        critical_issues = [r for r in results if r["severity"] == "critical"]
        
        return {
            "overall_score": total_score,
            "max_score": max_score,
            "grade": overall_grade,
            "passes": total_score >= max_score * 0.6,
            "check_results": results,
            "critical_issues": critical_issues,
            "recommendations": self._generate_recommendations(results)
        }
    
    def _check_factual_consistency(self, report: str, state: ResearchState) -> dict:
        facts = state["extracted_facts"]
        report_lower = report.lower()
        
        contradictions = []
        for fact in facts:
            if fact.get("confidence", 0) >= 0.8:
                obj_val = fact.get("object_value", "").lower()
                negation_patterns = [f"not {obj_val[:20]}", f"没有{obj_val[:10]}"]
                for pattern in negation_patterns:
                    if pattern in report_lower:
                        contradictions.append({
                            "fact": f"{fact.get('subject')} {fact.get('predicate')} {obj_val[:50]}",
                            "contradiction": pattern,
                            "location": report_lower.find(pattern)
                        })
        
        score = 20 - len(contradictions) * 5
        return {
            "name": "事实一致性",
            "score": max(0, score),
            "severity": "critical" if contradictions else "ok",
            "details": f"发现 {len(contradictions)} 处潜在矛盾" if contradictions else "未发现明显矛盾"
        }
    
    def _check_citation_completeness(self, report: str, state: ResearchState) -> dict:
        citation_pattern = r'\[(\d+)\]'
        citations_found = re.findall(citation_pattern, report)
        
        unique_citations = set(citations_found)
        expected_min = min(len(state["collected_sources"]), 5)
        
        score = min(20, len(unique_citations) * 2)
        severity = "warning" if len(unique_citations) < expected_min else "ok"
        
        return {
            "name": "引用完整性",
            "score": score,
            "severity": severity,
            "details": f"找到 {len(unique_citations)} 个独立引用，建议至少 {expected_min} 个"
        }
    
    def _check_coverage_adequacy(self, report: str, state: ResearchState) -> dict:
        coverage = state.get("coverage_map", {})
        if not coverage:
            return {"name": "覆盖度检查", "score": 15, "severity": "ok", "details": "无覆盖度数据"}
        
        undercovered = [sq for sq, cov in coverage.items() if cov < 0.5]
        avg_cov = sum(coverage.values()) / len(coverage)
        
        score = 20 if avg_cov >= 0.7 else (15 if avg_cov >= 0.5 else 10)
        severity = "warning" if undercovered else "ok"
        
        return {
            "name": "覆盖度充分性",
            "score": score,
            "severity": severity,
            "details": f"平均覆盖率 {avg_cov:.0%}，"
                      f"{'以下维度覆盖不足: ' + ', '.join(undercovered) if undercovered else '各维度覆盖良好'}"
        }
    
    def _check_language_quality(self, report: str, state: ResearchState) -> dict:
        issues = []
        
        very_long_paragraphs = [p for p in report.split('\n\n') if len(p) > 1500]
        if very_long_paragraphs:
            issues.append(f"发现 {len(very_long_paragraphs)} 个超长段落")
        
        empty_sections = re.findall(r'## .+\n\n\*(.*?)\*', report)
        if empty_sections:
            issues.append(f"发现 {len(empty_sections)} 个空章节")
        
        repetitive_phrases = re.findall(r'(.{50,})\1', report)
        if repetitive_phrases:
            issues.append(f"发现重复内容")
        
        score = 20 - len(issues) * 3
        return {
            "name": "语言质量",
            "score": max(5, score),
            "severity": "warning" if issues else "ok",
            "details": "; ".join(issues) if issues else "语言表达正常"
        }
    
    def _check_length_appropriateness(self, report: str, state: ResearchState) -> dict:
        word_count = len(report.split())
        fact_count = len(state["extracted_facts"])
        
        ideal_min = fact_count * 30
        ideal_max = fact_count * 150
        
        if ideal_min <= word_count <= ideal_max:
            score = 20
            severity = "ok"
        elif word_count < ideal_min:
            score = 12
            severity = "warning"
        else:
            score = 16
            severity = "ok"
        
        return {
            "name": "篇幅适当性",
            "score": score,
            "severity": severity,
            "details": f"报告约 {word_count} 词，基于 {fact_count} 条事实"
        }
    
    def _calculate_grade(self, score: int, max_score: int) -> str:
        ratio = score / max_score
        if ratio >= 0.9:
            return "A"
        elif ratio >= 0.75:
            return "B"
        elif ratio >= 0.6:
            return "C"
        else:
            return "D"
    
    def _generate_recommendations(self, results: list) -> list:
        recommendations = []
        for r in results:
            if r["severity"] == "critical":
                recommendations.append(f"🔴 必须: {r['details']}")
            elif r["severity"] == "warning":
                recommendations.append(f"🟡 建议: {r['details']}")
        return recommendations

quality_checker = ReportQualityChecker()

async def generate_final_report_with_qa(state: ResearchState) -> tuple:
    enhanced_report = generate_enhanced_report(state)
    
    visualizations = generate_visualizations(state)
    report_with_viz = insert_visualizations_into_report(enhanced_report, visualizations)
    
    qa_result = quality_checker.validate(report_with_viz, state)
    
    final_metadata = {
        **state.get("report_metadata", {}),
        "quality_check": qa_result,
        "visualization_count": len(visualizations),
        "word_count": len(report_with_viz.split()),
        "generated_at": datetime.now().isoformat(),
        "version": "2.0-enhanced"
    }
    
    log_entry = (
        f"[DONE] 增强版报告生成完成。"
        f"质量评分: {qa_result['grade']} ({qa_result['overall_score']}/{qa_result['max_score']})，"
        f"字数: {final_metadata['word_count']}，"
        f"图表: {len(visualizations)}"
    )
    
    if not qa_result["passes"]:
        log_entry += f" ⚠️ 未通过质量检查: {'; '.join(qa_result['recommendations'][:3])}"
    
    return report_with_viz, final_metadata, qa_result, [log_entry]
```

`ReportQualityChecker` 实现了五项自动化质量检查：

1. **事实一致性**：检查报告中是否存在与高置信度事实相矛盾的表述
2. **引用完整性**：检查是否有足够的引用标注（至少 5 个独立来源）
3. **覆盖度充分性**：确认各研究维度都有基本的信息覆盖（不低于 50%）
4. **语言质量**：检测超长段落、空章节、重复内容等问题
5. **篇幅适当性**：验证报告长度与事实数量的比例是否合理

每项检查给出 0-20 分的评分和严重程度（ok/warning/critical），最终汇总为 A/B/C/D 等级。只有 C 级以上（60%）才被认为"通过"。未通过的检查会生成具体的改进建议，这些信息既可以反馈给用户，也可以在未来版本中触发自动重试机制。

到这里，DeepResearch 的结果整合与展示能力已经非常完善了。从简单的文本罗列进化到了具有大纲规划、逐章撰写、数据可视化、交互式回顾、质量校验等全套能力的专业报告生成系统。最后一节我们将把这个强大的研究助手部署到生产环境，让它能真正为用户提供服务。
