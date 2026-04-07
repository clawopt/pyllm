# 6.3 子图的测试策略

> 子图最大的工程价值之一就是可独立测试性——因为子图是一个完整的 CompiledGraph，你不需要启动整个系统就能对它进行端到端的测试。但如何高效地组织这些测试？如何模拟父图的环境？如何验证子图之间的集成是否正确？这一节我们会建立一套完整的子图测试方法论。

## 子图单元测试：隔离验证每个模块

子图单元测试的目标是验证单个子图在隔离环境下的行为是否符合预期。这种测试不依赖任何外部系统（如数据库、API、其他子图），只关注子图内部的逻辑正确性。

```python
import pytest
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class TextProcessorState(TypedDict):
    input_text: str
    normalized_text: str
    word_count: int
    sentiment: str
    keywords: list[str]

def normalize(state: TextProcessorState) -> dict:
    text = state["input_text"].strip().lower()
    text = " ".join(text.split())
    return {"normalized_text": text}

def count_words(state: TextProcessorState) -> dict:
    words = state["normalized_text"].split()
    return {"word_count": len(words)}

def analyze_sentiment(state: TextProcessorState) -> dict:
    text = state["normalized_text"]
    positive = {"好", "棒", "优秀", "喜欢", "good", "great", "love"}
    negative = {"差", "烂", "讨厌", "bad", "terrible", "hate"}

    pos_count = sum(1 for w in positive if w in text)
    neg_count = sum(1 for w in negative if w in text)

    if pos_count > neg_count * 1.5:
        sentiment = "positive"
    elif neg_count > pos_count * 1.5:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment}

def extract_keywords(state: TextProcessorState) -> dict:
    from collections import Counter
    text = state["normalized_text"]
    stop_words = {"the", "is", "a", "an", "and", "or", "in", "on", "at"}
    words = [w for w in text.split() if len(w) > 2 and w not in stop_words]
    top_keywords = [w for w, _ in Counter(words).most_common(5)]
    return {"keywords": top_keywords}

processor_graph = StateGraph(TextProcessorState)
processor_graph.add_node("normalize", normalize)
processor_graph.add_node("count_words", count_words)
processor_graph.add_node("analyze_sentiment", analyze_sentiment)
processor_graph.add_node("extract_keywords", extract_keywords)

processor_graph.add_edge(START, "normalize")
processor_graph.add_edge("normalize", "count_words")
processor_graph.add_edge("count_words", "analyze_sentiment")
processor_graph.add_edge("analyze_sentiment", "extract_keywords")
processor_graph.add_edge("extract_keywords", END)

compiled_processor = processor_graph.compile()

class TestTextProcessorSubgraph:
    def test_normal_input(self):
        result = compiled_processor.invoke({
            "input_text": "  Hello World, this is GREAT!  ",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert result["normalized_text"] == "hello world, this is great!"
        assert result["word_count"] == 6

    def test_empty_input(self):
        result = compiled_processor.invoke({
            "input_text": "",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert result["normalized_text"] == ""
        assert result["word_count"] == 0
        assert result["keywords"] == []

    def test_positive_sentiment(self):
        result = compiled_processor.invoke({
            "input_text": "This product is great and I love it",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert result["sentiment"] == "positive"

    def test_negative_sentiment(self):
        result = compiled_processor.invoke({
            "input_text": "This is terrible and I hate it so bad",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert result["sentiment"] == "negative"

    def test_neutral_sentiment(self):
        result = compiled_processor.invoke({
            "input_text": "The meeting is at 3pm tomorrow",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert result["sentiment"] == "neutral"

    def test_keyword_extraction(self):
        result = compiled_processor.invoke({
            "input_text": "Python programming language for data science and machine learning",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert len(result["keywords"]) <= 5
        assert isinstance(result["keywords"], list)
        assert all(isinstance(k, str) for k in result["keywords"])

    def test_special_characters_handling(self):
        result = compiled_processor.invoke({
            "input_text": "Hello!!!   World???  \n\n  Test...",
            "normalized_text": "", "word_count": 0,
            "sentiment": "", "keywords": []
        })
        assert "!!!" not in result["normalized_text"]
        assert "..." in result["normalized_text"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

这套测试覆盖了文本处理子图的所有关键路径：正常输入、空输入、三种情感倾向、关键词提取、特殊字符处理。每个测试方法都构造了完整的初始状态（包括所有必需字段的默认值），调用编译后的子图，然后断言输出的正确性。注意一个重要的实践：**每次测试都应该提供完整的状态字典，即使某些字段在本次测试中不会被用到**——这保证了测试的独立性和可重复性。

## 子图集成测试：验证数据流正确性

单元测试验证了每个子图内部的正确性，但还需要集成测试来验证多个子图组装在一起时数据流是否正确。特别是要验证前一个子图的输出是否能正确地映射为后一个子图的输入。

```python
class TestPipelineIntegration:
    def setup_method(self):
        self.app = pipeline_graph.compile()

    def test_full_pipeline_data_flow(self):
        result = self.app.invoke({
            "raw_text": "Python async programming tutorial",
            "classification": {}, "retrieval_results": [],
            "generated_answer": "", "quality_report": {},
            "pipeline_log": []
        })

        assert result["classification"]["category"] == "technical"
        assert len(result["retrieval_results"]) > 0
        assert len(result["generated_answer"]) > 0
        assert len(result["pipeline_log"]) >= 3

    def test_classification_output_feeds_retrieval(self):
        result = self.app.invoke({
            "raw_text": "How to cook pasta?",
            "classification": {}, "retrieval_results": [],
            "generated_answer": "", "quality_report": {},
            "pipeline_log": []
        })

        cls_category = result["classification"]["category"]
        assert cls_category in ["technical", "cooking", "general", "other"]

        retrieval_results = result["retrieval_results"]
        for r in retrieval_results:
            assert "title" in r
            assert "relevance" in r

    def test_empty_input_handled_gracefully(self):
        result = self.app.invoke({
            "raw_text": "",
            "classification": {}, "retrieval_results": [],
            "generated_answer": "", "quality_report": {},
            "pipeline_log": []
        })

        assert isinstance(result["generated_answer"], str)
        assert isinstance(result["pipeline_log"], list)

    def test_long_input_does_not_crash(self):
        long_text = "test sentence. " * 1000
        result = self.app.invoke({
            "raw_text": long_text,
            "classification": {}, "retrieval_results": [],
            "generated_answer": "", "quality_report": {},
            "pipeline_log": []
        })
        assert result is not None
```

集成测试的关键在于它验证的是**接口契约的满足程度**——分类子图产生的输出格式是否满足检索子图的期望？检索子图的输出是否包含生成子图需要的字段？这些跨子图的交互问题只有在集成测试中才能被发现。

## Mock 外部依赖进行测试

当子图内部依赖外部服务（LLM API、数据库、第三方 HTTP 接口）时，直接运行测试会带来问题：测试速度慢、需要网络连接、可能产生费用、结果不确定（LLM 输出有随机性）。解决方案是用 mock 对象替代真实的外部依赖。

```python
from unittest.mock import MagicMock, patch

class TestWithMockedDependencies:

    @patch("__main__.llm.invoke")
    def test_llm_subgraph_with_mock(self, mock_llm_invoke):
        mock_response = MagicMock()
        mock_response.content = "这是模拟的 LLM 回复内容"
        mock_llm_invoke.return_value = mock_response

        result = compiled_generate.invoke({
            "question": "什么是 Python 装饰器？",
            "context": [{"title": "Python 文档"}],
            "answer": ""
        })

        assert "模拟" in result["answer"]
        mock_llm_invoke.assert_called_once()

    @patch("__main__.requests.get")
    def test_api_call_subgraph_with_mock(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"id": "1", "score": 0.95},
                {"id": "2", "score": 0.88}
            ]
        }
        mock_get.return_value = mock_response

        # 运行包含 API 调用的子图
        result = some_api_subgraph.invoke({...})

        assert len(result["results"]) == 2
        mock_get.assert_called_once()

    def test_database_subgraph_with_fake_db(self):
        fake_db = {
            "user_001": {"name": "张三", "tier": "vip"},
            "user_002": {"name": "李四", "tier": "normal"}
        }

        with patch.dict("os.environ", {"TEST_MODE": "true"}):
            result = user_lookup_subgraph.invoke({
                "user_id": "user_001",
                "_db_override": fake_db
            })
            assert result["name"] == "张三"
            assert result["tier"] == "vip"
```

Mock 测试的核心思想是：**让测试只验证逻辑的正确性，而不依赖于外部服务的可用性和行为**。通过 mock，你可以精确控制外部依赖的返回值，从而覆盖各种边界情况（成功、失败、超时、异常格式等）。

## 测试覆盖率分析

和普通代码一样，子图的测试也需要关注覆盖率。但由于子图是图结构而非线性代码，传统的代码行覆盖率工具无法直接适用。我们需要从图的角度来定义覆盖率：

**节点覆盖率**：图中定义的所有节点中，有多少被至少一次执行路径经过？

**边覆盖率**：图中定义的所有边（包括条件边的各个分支）中，有多少被实际触发？

**状态字段覆盖率**：状态类型定义的所有字段中，有多少被实际写入或读取？

```python
def analyze_subgraph_coverage(graph_app, test_cases: list[dict]) -> dict:
    graph_structure = graph_app.get_graph()
    all_nodes = set(graph_structure.nodes.keys()) - {START, END}
    executed_nodes = set()
    triggered_edges = set()
    touched_fields = set()

    for case in test_cases:
        try:
            captured_nodes = []

            for event in graph_app.stream(case, stream_mode="updates"):
                for node_name in event:
                    captured_nodes.append(node_name)
                    update = event[node_name]
                    touched_fields.update(update.keys())

                    prev = None
                    for n in captured_nodes[:-1]:
                        prev = n
                    if prev:
                        triggered_edges.add((prev, node_name))

            executed_nodes.update(captured_nodes)

        except Exception as e:
            pass

    missing_nodes = all_nodes - executed_nodes
    total_edges = count_all_edges(graph_structure)
    
    return {
        "total_nodes": len(all_nodes),
        "executed_nodes": len(executed_nodes),
        "node_coverage": f"{len(executed_nodes)/len(all_nodes)*100:.1f}%" if all_nodes else "N/A",
        "missing_nodes": sorted(missing_nodes),
        "edges_triggered": len(triggered_edges),
        "fields_touched": sorted(touched_fields),
        "unique_paths": len(set(tuple(n) for n in [get_path_for_case(case) for case in test_cases]))
    }

# 使用示例
coverage = analyze_subgraph_coverage(compiled_processor, [
    {"input_text": "Hello world", ...},
    {"input_text": "", ...},
    {"input_text": "Great product!", ...},
])
print(f"节点覆盖率: {coverage['node_coverage']}")
print(f"未覆盖节点: {coverage['missing_nodes']}")
```

这个覆盖率分析工具通过运行一组测试用例并追踪实际的执行路径来统计覆盖率指标。在实际项目中，建议把这类分析作为 CI/CD 流水线的一部分——如果节点覆盖率低于某个阈值（比如 80%），就让构建失败提醒开发者补充测试用例。
