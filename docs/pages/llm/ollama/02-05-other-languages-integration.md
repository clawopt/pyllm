# 02-5 其他语言接入速查

## 语言选择的现实考量

在实际项目中，你不太可能同时用五种语言来调用 Ollama——通常你的技术栈已经确定了主要语言。这一节的目标是：**让你知道如果某一天需要用其他语言（比如维护一个用 Go 写的旧系统，或者给 Java 团队提供 AI 能力），该怎么快速接入 Ollama。**

每个语言的示例都会遵循相同的模式：
1. 安装依赖（如果有官方/社区库）
2. 发起 HTTP 请求到 `http://## Go

Go 是云原生生态的首选语言之一，很多基础设施工具（Kubernetes、Docker、Prometheus）都是 Go 写的。如果你的后端服务是 Go 架构，直接在进程内调用 Ollama 比起跨语言 RPC 高效得多。

```go
package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	apiBase = "http://localhost:11434"
	defaultModel = "qwen2.5:7b"
	timeout = 120 * time.Second
)

// ===== 基础数据结构 =====

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatResponse struct {
	Content string `json:"message.content"`
	Model   string `json:"model"`
	Done     bool   `json:"done"`
}

type ChatStreamChunk struct {
	Content *string `json:"message.content,omitempty"`
	Done     bool    `json:"done"`
}

// ===== 核心客户端 =====

type Client struct {
	BaseURL    string
	DefaultModel string
	HTTPClient  *http.Client
}

func NewClient(opts ... ) *Client {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = apiBase
	}
	return &Client{
		BaseURL:    baseURL,
		DefaultModel: opts.DefaultModel,
		HTTPClient: &http.Client{Timeout: timeout},
	}
}

// ===== Chat（同步非流式）=====

func (c *Client) Chat(ctx context.Context, prompt string, model ... ) (string, error) {
	m := model
	if m == "" {
		m = c.DefaultModel
	}
	
	resp, err := c.HTTPClient.Post(c.BaseURL+"/api/chat",
		"application/json",
		bytes.NewBufferString(json.Marshal(map[string]interface{}{
			"model": m,
			"messages": []ChatMessage{
				{Role: "user", Content: prompt},
			}},
		})),
	)
	if err != nil {
		return "", fmt.Errorf("请求失败: %w", err)
	}
	
	defer resp.Body.Close()
	
	var result ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("解析响应失败: %w", err)
	}
	
	return result.Content, nil
}

// ===== ChatStream（流式）=====

func (c *Client) ChatStream(ctx context.Context, prompt string, onChunk func(string) error) error {
	m := c.DefaultModel
	
	resp, err := c.HTTPClient.Post(c.BaseURL+"/api/chat",
		"application/json",
		bytes.NewBufferString(json.Marshal(map[string]interface{}{
			"model": m,
			"messages": []ChatMessage{
				{Role: "user", Content: prompt},
			}},
			"stream": true,
		})),
	)
	if err != nil {
		return fmt.Errorf("启动流式请求失败: %w", err)
	}
	defer resp.Body.Close()
	
	reader := bufio.NewReader(resp.Body)
	scanner := bufio.NewScanner(reader)
	
	for {
		line, err := scanner.ReadString('\n')
		if err != nil {
			if err == io.EOF { break }
			return fmt.Errorf("读取流结束异常: %w", err)
		}
		
		if line == "" {
			continue // 空行或心跳包
		}
		
		if line == "data: [DONE]" {
			break // 流结束标记
		}
		
		if !hasPrefix(line, "data: ") {
			continue // 非 SSE 数据行
			}
		
		dataStr := trimPrefix(line, "data: ")
		
		var chunk ChatStreamChunk
		if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
			continue // 解析失败跳过
			}
		
		if chunk.Message != nil && chunk.Message.Content != "" {
			if err := onChunk(chunk.Message.Content); err != nil {
				return err
			}
		}
		
		if chunk.Done {
			break
		}
	}
	
	return nil
}

// ===== Embedding =====

func (c *Client) Embed(ctx context.Context, texts []string, model ... ) ([][]float64, error) {
	m := model
	if m == "" {
		m = "nomic-embed-text"
	}
	
	var embeddings [][]float64
	
	for _, text := range texts {
		resp, err := c.HTTPClient.Post(c.BaseURL+"/api/embeddings",
			"application/json",
			bytes.NewBufferString(json.Marshal(map[string]interface{}{
				"model": m,
				"input": text,
			})),
		)
		if err != nil {
			return nil, err
		}
		
		var result struct {
			Embeddings [][]float64 `json:"embeddings"`
		}
		
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, err
			}
		
		embeddings = append(embeddings, result.Embeddings...)
	}
	
	return embeddings, nil
}

// ===== 使用示例 =====

func main() {
	client := NewClient(WithOpts{
		DefaultModel: "qwen2.5:7b",
	})
	
	// 场景一：CLI 工具
	if len(os.Args) > 1 && os.Args[1] == "chat" {
		prompt := strings.Join(os.Args[2:], " ")
		answer, err := client.Chat(context.Background(), prompt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "错误: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(answer)
		return
	}
	
	// 场景二：Web 服务（HTTP Handler）
	http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Prompt string `json:"prompt"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		
		answer, err := client.Chat(r.Context(), req.Prompt)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"answer": answer,
			"model": client.DefaultModel,
		})
	})
	
	// 启动服务
	fmt.Println("🚀 Ollama Go 服务运行在 :8080")
		log.Fatal(http.ListenAndServe(":8080", nil))
}

// ===== 辅助函数 =====

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func trimPrefix(s, prefix string) string {
	if hasPrefix(s, prefix) {
		return s[len(prefix):]
	}
	return s
}
```

使用方式：

```bash
# 构建
go build -o ollama-cli ./cmd/

# CLI 模式
./ollama-chat chat "什么是闭包？"

# 或作为库导入到现有项目
```

Go 的优势在于**编译为单一二进制、部署简单、并发性能极强**（goroutine 轻量级轻量）。适合构建需要长期运行的后台服务。

## Java / Spring AI 集成

Java 企业级应用集成 Ollama 通常通过 Spring AI 或 Spring WebFlux 来完成：

```java
// OllamaService.java
package com.example.ollama;

import com.fasterxml.jackson.annotation.*;
import org.springframework.stereotype.*;
import org.springframework.web.client.*;

@Service
public class OllamaService {
    
    private final RestTemplate restTemplate;
    private final String apiBase;
    
    public OllamaServiceImpl(
            @Value("${ollama.api.base-url:http://localhost:11434}") String apiBase) {
        this.restTemplate = new RestTemplate();
        this.apiBase = apiBase;
    }
    
    /**
     * 同步聊天
     */
    public String chat(String userMessage) {
        var request = Map.of(
            "model", "qwen2.5:7b",
            "messages", List.of(
                Map.of("role", "user"),
                "content", userMessage)
            )
        );
        
        var response = restTemplate.postForObject(
                apiBase + "/api/chat",
                request,
                Map.class,
                String.class
        );
        
        @SuppressWarnings("unchecked")
        Map<String, Object> body = (Map<String, Object>) response;
        return (String) ((Map<String, Object>) body.get("message")).get("content");
    }
    
    /**
     * 流式聊天（Server-Sent Events）
     */
    public Flux<String> chatStream(String userMessage) {
        var request = Map.of(
            "model", "qwen2.5:7b",
            "messages", List.of(
                Map.of("role", "user"),
                "content", userMessage)
            ),
            "stream", true
        );
        
        return restTemplate.postForFlux(
                apiBase + "/api/chat",
                request,
                String.class,
                String.class
        ).flatMapMany(response -> {
            // 解析 SSE 数据块
            Flux<DataBuffer> dataBuffer = DataBuffer.wrap(
                    response.getBodyAs(StandardCharsets.UTF_8),
                StandardCharsets.UTF_8
            );
            
            return dataBuffer
                    .map(chunk -> {
                        String line = StandardCharsets.UTF_8.decode(
                                java.nio.ByteBuffer.wrap(
                                        chunk.asByteBuffer(), 
                                        StandardCharsets.UTF_8
                                )
                        ).toString();
                        
                        if (line.startsWith("data: ")) {
                            try {
                                String json = line.substring(6);
                                if (!"[DONE]".equals(json)) {
                                    return Mono.just(
                                            com.fasterxml.jackson.databind.JsonNodeFactory
                                                    .instance(json)
                                                    .get("message").asText();
                                }
                            } catch (Exception e) {
                                return Mono.empty();
                            }
                        }
                        }
                        return Mono.empty();
                    }
                    return Mono.just(line);
                })
                .filter(line -> !line.isEmpty())
                .doOnNext();  // 过滤空行和心跳包
        });
    }
    
    /**
     * 文本向量化
     */
    public float[] embed(String text) {
        var request = Map.of(
            "model", "nomic-embed-text",
            "input", text
        );
        
        var response = restTemplate.postForObject(
                apiBase + "/api/embeddings",
                request,
                Map.class,
                OllamaEmbeddingResponse.class
        );
        
        OllamaEmbeddingResponse embeddingResult = (OllamaEmbeddingResponse) response;
        return embeddingResult.getEmbeddings()[0];
    }
    
    // ===== DTO 类 =====
    @Data
    static class OllamaChatResponse {
        private String message;
        private String model;
        private boolean done;
        private long totalDuration;
        private int evalCount;
        private long promptEvalCount;
    }
    
    @Data
    static class OllamaEmbeddingResponse {
        private List<float[]> embeddings;
    }

    // ===== Controller 层示例 =====
    @RestController
    @RequestMapping("/api/ai/chat")
    public class AiChatController {
        
        private final OllamaService ollamaService;
        
        public AiChatController(OllamaService ollamaService) {
            this.ollamaService = ollamaService;
        }
        
        @PostMapping
        public ResponseEntity<Map<String, Object>> chat(@RequestBody Map<String, Object> body) {
            String answer = ollamaService.chat((String) body.get("message"));
            return ResponseEntity.ok(Map.of(
                "answer", answer,
                "model", ollamaService.getDefaultModel(),
                "timestamp", System.currentTimeMillis(),
            ));
        }
        
        @PostMapping(value = "/stream")
        public Flux<ServerSentEvent> chatStream(@RequestBody Map<String, Object> body) {
            Flux<ServerSentEvent> events = ollamaService.chatStream((String) body.get("message"));
            return ServerSentEvent.flux(events);
        }
    }
}
```

Java 方案的关键设计决策：
- **`RestTemplate` vs `WebClient`**：RestTemplate 更适合结构化 API 调用；WebClient 在需要更灵活的 HTTP 控制（如超时、拦截器）
- **`@ConfigurationProperties` 配置 Ollama 地址**：让模型名、API 地址都可以外部化配置，不用硬编码
- **SSE (Flux)**：Spring 的 `Flux<ServerSentEvent>` 天然适配 SSE 协议，非常适合流式输出

### Spring AI 自动发现

如果你用的是 Spring AI（`org.springframework.ai:spring-ai`），Ollama 可以被自动识别为一个 ChatModel provider：

```java
@Configuration
public class OllamaConfig {

    @Bean
    public ChatModel ollamaChatModel() {
        // Spring AI 会自动扫描 classpath 下所有 ChatModel 实现
        // 我们只需要返回一个包装 Ollama 的 ChatModel 即可
        return new OllamaChatModel(
            "http://localhost:11434/v1",  # Ollama 兼容 OpenAI 格式
            "qwen2.5:7b"              // 默认模型
        );
    }
}

@Component
class OllamaChatModel implements ChatModel {
    
    private final String baseUrl;
    private final String defaultModel;
    
    public OllamaChatModel(String baseUrl, String defaultModel) {
        this.baseUrl =  = baseUrl;
        this.defaultModel = defaultModel;
    }
    
    @Override
    public ChatResponse call(Prompt userPrompt) {
        // Spring AI 的标准接口 —— 只需实现这一个方法！
        var messages = convertToMessages(userPrompt);
        
        // 内部调用 Ollama API
        var response = callOllama(messages);
        return new ChatResponse(response);
    }
    
    // ... 其他接口实现 ...
}
```

这样你在 Controller 中就可以像使用 OpenAI 一样使用 Ollama 了——而且切换回 GPT-4o 只需改一行配置。

## Rust / Axum 接入

Rust 的高性能特性使其成为推理服务的热门选择。如果你正在构建一个需要极低延迟的 AI 服务端：

```rust
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value};

const API: &str = "http://localhost:11434";
const MODEL: &str = "qwen2.5:7b";

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ApiResponse {
    message: MessageContent,
    model: String,
    done: bool,
    #[serde(rename = "eval_count")]
    eval_count: u32,
    #[serde(rename = "prompt_eval_count")]
    pub prompt_eval_count: u32,
}

#[derive(Debug, Serialize)]
struct MessageContent {
    content: String,
}

struct OllamaClient {
    client: Client,
}

impl OllamaClient {
    fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client");
        }
    }
    
    fn chat(&self, message: str) -> Result<String, Box<dyn std::error::Error>> {
        let payload = serde_json::json!({
            "model": MODEL,
            "messages": [ChatMessage {
                role: "user".into(),
                content: message.into(),
            }],
            "stream": false,
        });
        
        let resp = self.client
            .post(format!("{}/api/chat", API))
            .header("Content-Type", "application/json")
            .body(&payload)
            .send()?;
        
        if !resp.status().is_success() {
            return Err(format!("HTTP {}: {}", resp.status(), resp.status()?));
        }
        
        let result: ApiResponse = serde_json::from_str(&resp.text()?)?;
        Ok(result.message.content)
    }
    
    async fn chat_stream(&self, message: str) -> Result<String, Box<dyn std::error::Error>> {
        let payload = serde_json::json!({
            "model": "MODEL",
            "messages": [ChatMessage {
                role: "user".into(),
                content: message.into(),
            }],
            "stream": true,
        });
        
        let resp = self.client
            .post(format!("{}/api/chat", API))
            .header("content-type", "application/json")
            .json(&payload)
            .send()?
            .await
            .map_err(|e| e.to_string())?;
        
        let mut full_response = String::new();
        
        use futures_util::StreamExt;  // 需要 tokio-stream 库
        
        let mut resp_stream = resp.bytes_stream()?.expect("Failed to get byte stream")?;
        
        loop {
            let chunk = resp_stream.chunk().await.map_err(|e| format!("SSE read error: {e}"))?;
            
            let line = match std::str::from_utf8(&chunk) {
                Ok(s) => s,
                Err(_) => continue,
            };
            
            if line.starts_with("data: ") {
                let json_str = &line[6..]; // 去掉 "data: "
                
                if json_str == "[DONE]" {
                    break;
                }
                
                if let Ok(chunk) = serde_json::from_str::<ChatStreamChunk>(json_str) {
                    if let Some(content) = chunk.message.content {
                        full_response.push_str(content.as_str());
                        print!("{}", content.as_str()); // 实时打印
                    }
                }
            }
        }
        
        Ok(full_response)
    }
    
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let payload = serde_json::json!({
            "model": "nomic-embed-text",
            "input": text,
        });
        
        let resp = self.client
            .post(format!("{}/api/embeddings", API))
            .header("Content-Type", "application/json")
            .body(&payload)
            .send()?;
        
        if !resp.status().is_success() {
            return Err(format!("Embedding failed: {}", resp.status()));
        }
        
        let result: OllamaEmbedResponse = serde_json::from_str(&resp.text()?)?;
        Ok(result.embeddings.into_iter().map(|v| v.clone()).collect())
    }
}

#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    message: Option<MessageContent>,
    done: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = OllamaClient::new();
    
    println!("=== 同步模式 ===");
    let answer = client.chat("Rust 怎么写一个 HTTP server？")?;
    println!("{}\n", answer);
    
    println!("=== 流式模式 ===");
    client.chat_stream("用 Rust 写一个快速排序").await?;
    
    println!("\n=== Embedding ===");
    let emb = client.embed("Hello world")?;
    println!("维度: {}", emb.len());
    
    Ok(())
}
```

Rust 版本的特点：
- **零运行时开销**：编译后的二进制体积极小
- **内存安全**：Rust 所有权检查在编译期完成
- **并发性能**：tokio::spawn 的异步 I/O 效率极高
- **错误类型安全**：`Result<T, E>` 强制处理每种可能的错误路径

## C# / .NET 集成

对于 Windows 企业环境（尤其是大量 legacy .NET Framework 项目），C# 接入 Ollama 是最自然的选择：

```csharp
using System.Text.Json;
using System.Net.Http;

namespace Ollama.Csharp;

public class OllamaClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly Uri _baseUri;
    private readonly string _defaultModel;
    
    public OllamaClient(
        string baseUri = "http://localhost:11434",
        string defaultModel = "qwen2.5:7b"
    )
    {
        _baseUri = new Uri(baseUri.TrimEndsWith('/'));
        _defaultModel = defaultModel;
        _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(120) };
    }
    
    /// <summary>
    /// 发送聊天消息并获取完整回复
    /// </summary>
    public async Task<string> ChatAsync(string message, string? model = null)
    {
        model ??= _defaultModel;
        
        var payload = new
        {
            model = model,
            messages = new[]
            {
                new { role = "user", content = message },
            },
            stream = false,
        };
        
        var resp = await _httpClient.PostAsJsonAsync(
            new Uri(_baseUri, "/api/chat"),
            payload,
            new MediaTypeHeaderValue("application/json")
        );
        
        resp.EnsureSuccessStatusCode();
        
        var result = await resp.Content.ReadFromJsonAsync<ChatResponse>();
        return result.Message.Content;
    }
    
    /// <summary>
    /// 流式聊天——实时逐 token 返回
    /// </summary>
    public async IAsyncEnumerable<string> ChatStreamAsync(string message, string? model = null)
    {
        model ??= _defaultModel;
        
        var payload = new
        {
            model = model,
            messages = new[]
            {
                new { role = "user", content = message },
            },
            stream = true,
        };
        
        var resp = await _httpClient.SendAsync(
            new HttpRequestMessage(HttpMethod.Post, new Uri(_baseUri, "/api/chat"))
        {
            Content = JsonContent.Create(payload, 
                new MediaTypeHeaderValue("application/json")),
            Headers = {
                { HeaderNames.Accept, "*/*" },
            },
            CompletionOption = HttpCompletionOption.ResponseHeadersRead,
        };
        
        var stream = resp.Content.ReadFromStreamAsync(
            new Progress<>(
                new ReadAheadSize(256),
                // 一次读取 256 字节
                // 对于大模型回复，这个值可以适当调大
            ),
        );
        
        await foreach (var line in stream.ReadAllLinesAsync())  // 逐行读取
        {
            var line = line.Trim();
            
            if (line.StartsWith("data: "))
            {
                var json = line[6..];  // 去掉 "data: " 前缀
                
                if (json == "[DONE]") break;
                
                var chunk = JsonSerializer.Deserialize<
                    OllamaStreamChunk>(json);
                    
                if (chunk.Message?.Content is not null)
                {
                    yield return chunk.Message.Content;
                }
            }
            // 忽略空行和心跳包
        }
    }
    
    /// <summary>
    /// 文本向量化
    /// </summary>
    public async Task<float[]> EmbedAsync(string text, string? model = null)
    {
        model ??= "nomic-embed-text";
        
        var payload = new { model, input = text };
        
        var resp = await _httpClient.PostAsJsonAsync(
            new Uri(_baseUri, "/api/embeddings"),
            payload,
            new MediaTypeHeaderValue("application/json")
        );
        
        resp.EnsureSuccessStatusCode();
        
        var result = await resp.Content.ReadFromJsonAsync<OllamaEmbedResponse>();
        return result.Embeddings.First();
    }
    
    public void Dispose() => _httpClient.Dispose();
}

// ===== 数据模型 =====

public record OllamaStreamChunk
{
    [JsonPropertyName("message"), NullValueHandling = Ignore]
    public OllamaMessage? Message { get; set; }
    
    [JsonPropertyName("done"), NullValueHandling = Ignore]
    public bool Done { get; set; }
}

public record OllamaMessage
{
    [JsonPropertyName("content")]
    public string Content { get; set; }
}

public record OllamaEmbeddingResponse
{
    [JsonPropertyName("embeddings")]
    public List<float[]> Embeddings { get; }
}

// ===== 使用示例 =====
public class Program
{
    static async Task Main(string[] args)
    {
        using var client = new OllamaClient();
        
        Console.WriteLine("=== 同步问答 ===");
        string answer = await client.ChatAsync("什么是装饰器？");
        Console.WriteLine($"A: {answer}\n");
        
        Console.WriteLine("=== 流式输出 ===");
        await foreach (var token in client.ChatStreamAsync("写一首七言绝句")) 
        {
            Console.Write(token);
        }
        Console.WriteLine("\n");
        
        Console.WriteLine("=== Embedding ===");
        var vec = await client.EmbedAsync("机器学习");
        Console.WriteLine($"维度: {vec.Length}, 前3个值: [{string.Join(", ", vec[..3].Select(v => v.ToString("F4")))}]\n");
    }
}
```

C# / .NET 的额外优势：
- **Visual Studio 调试友好**：断点、监视窗口、即时变量查看
- **与 ASP.NET Core 深度集成**：中间件、过滤器、DI 注入
- **企业认证体系**：Windows Auth / JWT / OAuth2
- **Serilog / 结构化日志**：天然支持

## Ruby / Rails 集成

```ruby
require 'net/http'
require 'json'

module Ollama
  API_BASE = 'http://localhost:11434'.freeze
  DEFAULT_MODEL = 'qwen2.5:7b'
  
  class Error < StandardError; end
  
  class Client
    def initialize(base_url: API_BASE, default_model: DEFAULT_MODEL)
      @base_url = base_url
      @default_model = default_model
      @http = Net::HTTP.new(:write_timeout => 30)
    end
    
    def chat(message:, model: nil)
      model ||= @default_model
      
      payload = { model: model, messages: [{ role: "user", content: message }] }
      
      resp = @http.post("#{@base_url}/api/chat",
                 headers: { 'Content-Type' => 'application/json' },
                 body: JSON.generate(payload))
      
      raise unless resp.success?
      
      JSON.parse(resp.body)['message']['content']
    rescue => StandardError, "请求失败: #{resp.code}: #{resp.body}"
  end
  
  def chat_stream(message:, model: nil, &block)
    model ||= @default_model
      
    payload = { model: model, messages: [{ role: "user", content: }], stream: true }
    
    resp = @http.post("#{@base_url}/api/chat",
                 headers: { 'Content-Type' => 'application/json' },
                 body: JSON.generate(payload),
                 stream: # 关键！
               )
    
    full = ""
    resp.body.each_line do |line|
      next if line.strip.empty?
      
      case line
      when /\Adata:\s/
        break if line == "[DONE]"
        
        begin
          chunk = JSON.parse(line[6..])
          full << chunk.dig('message&.content') if chunk['message']
        rescue
          next
        end
      end
      
      full
    end
  
  embed(texts:, model: "nomic-embed-text")
    resp = @http.post("#{@base_url}/api/embeddings",
                 headers: { 'Content-Type' => 'application/json' },
                 body: JSON.generate({ model: model, input: texts }))
    
    JSON.parse(resp.body)['embeddings'].first
  end
end

# ===== 使用示例 =====
if __FILE__ == $PROGRAM_NAME
  client = Ollama.new
  
  puts "=== 同步 ==="
  puts client.chat("Ruby 中怎么发 HTTP 请求?")
  
  puts "\n=== 流式 ==="
  client.chat_stream("用 Ruby 写个快速排序") { |c| print c }
  puts
  
  puts "\n=== Embedding ==="
  puts client.embed("Hello World").size
end
```

## 快速参考表：各语言关键差异

| 语言 | 安装方式 | HTTP 库 | 流式支持 | 异步 | 适用场景 |
|------|---------|----------|---------|------|------|---------|
| **Python** | `pip install ollama` | requests | ✅ | asyncio/aiohttp | ✅ | 全栈首选 |
| **JavaScript/TS** | `npm install ollama` | fetch / node-fetch | ✅ | ✅ | Web / Electron / Next.js |
| **Go** | `go get ...` | net/http | 手动实现 SSE | goroutine/channel | 微服务 / K8s |
| **Java** | Maven spring-ai | RestTemplate/WebClient | Flux (SSE) | WebClient | 企业级应用 |
| **Rust** | cargo add ollama | reqwest | tokio-stream | async/await | 高性能服务端 |
| **C#** | NuGet / RestClient | HttpClient | IAsyncEnumerable | async/await | WinForms / ASP.NET Core |
| **Ruby** | gem install ollama | net/http | each_line | block | Rails / Sinatra |

## 统一封装思路：多语言项目的架构建议

如果你的团队使用多种语言（比如前端 TS + 后端 Go + 维护脚本用 Python），最干净的方案是**把 Ollama 调用集中在一个独立的微服务中**，然后各语言通过统一的 gRPC/REST API 与它通信：

```
┌─────────────────────────────────────────────┐
│         TypeScript (Frontend)              │
│         Python (Scripts/MLOps)              │
│         Go (Backend Service)                  │
│         Ruby (DevOps)                       │
│                                          │
│                          ▼              │
│                   ollama-gateway           │
│                   (统一 Ollama 服务层)       │
│                                          │
└─────────────────────────────────────────────┘
```

这个网关层可以用任何一种语言编写（推荐 Go 或 Rust），然后提供统一的 REST API 给上层应用。这样做的好处是：
- 各业务模块不需要关心底层是大模型还是小模型、是 Ollama 还是 vLLM
- 切换底层 LLM 提供商只需改 gateway 的配置
- 网关层可以做统一的日志、限流、计费、缓存

本章我们完成了从 Python 到 JavaScript/TypeScript 到 Go/Java/Rust/C#/Ruby 全主流编程语言的 Ollama 接入覆盖。下一章将进入第三章：模型生态与选择指南。
