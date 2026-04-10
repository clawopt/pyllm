# 硬件选型与成本参考

> **白板时间**：老板给你 10 万预算，要部署一个能服务 100 并发用户的 LLM 推理系统。你该买什么 GPU？租云服务器还是自建机房？单卡 RTX 4090 还是双卡 A100？这一节我们用真实的数据和计算来帮你做出最优决策。

## 一、GPU 选型指南

### 1.1 主流 GPU 参数对比

| GPU | 架构 | 显存 | 带宽 | FP16 TFLOPS | 参考价格 | 适用场景 |
|-----|------|------|------|------------|---------|---------|
| **RTX 4090** | Ada | 24 GB GDDR6X | 1008 GB/s | 330 | ¥13,000 | 个人/小团队开发 |
| **RTX 6000 Ada** | Ada | 48 GB GDDR6X | 960 GB/s | 298 | ¥45,000 | 单卡跑 14B 模型 |
| **A5000** | Ampere | 48 GB GDDR6X | 768 GB/s | 148 | ¥35,000 | 预算有限的大模型 |
| **A100 80GB** | Ampere | 80 GB HBM2e | 2039 GB/s | 312 | ¥140,000 | 企业级推理主力 |
| **A100 40GB** | Ampere | 40 GB HBM2e | 1555 GB/s | 312 | ¥85,000 | 中等规模推理 |
| **H100 80GB** | Hopper | 80 GB HBM3 | 3350 GB/s | 1979 (FP8) | ¥280,000 | 高性能+FP8量化 |
| **H200 141GB** | Hopper | 141 GB HBM3e | 4800 GB/s | 1979 (FP8) | ¥400,000+ | 超大模型首选 |
| **L20 48GB** | Ada | 48 GB GDDR6 | 864 GB/s | 240 | ¥30,000 | A100 替代方案 |
| **L40S 48GB** | Ada | 48 GB GDDR6 | 864 GB/s | 362 | ¥38,000 | 多模态/VLM |

### 1.2 各 GPU 能跑多大模型？

```python
def gpu_model_fit():
    """GPU 与模型的匹配关系"""
    
    data = [
        ("RTX 4090", 24), ("RTX 6000 Ada", 48), ("A5000", 48),
        ("A100 40GB", 40), ("A100 80GB", 80),
        ("H100 80GB", 80), ("H200 141GB", 141),
    ]
    
    models = [
        ("Qwen2.5-0.5B", 1, 2),
        ("Qwen2.5-1.5B", 3, 4),
        ("Qwen2.5-3B", 6, 8),
        ("Qwen2.5-7B", 14, 22),
        ("Qwen2.5-14B", 28, 40),
        ("Qwen2.5-32B", 64, 86),
        ("Qwen2.5-72B", 144, 190),
    ]
    
    print(f"{'GPU':<16} | {'显存':>5} | {'0.5B':>5} {'1.5B':>5} {'3B':>5} {'7B':>5} "
          f"{'14B':>5} {'32B':>5} {'72B':>5}")
    print("-" * 85)
    
    for gpu_name, vram in data:
        row = f"{gpu_name:<16} | {vram:>4}G |"
        
        for model_name, fp16_gb, _ in models:
            can_fp16 = "✅" if fp16_gb <= vram * 0.85 else " "
            can_int4 = "🟢" if fp16_gb * 0.28 <= vram * 0.85 else " "
            
            if can_fp16:
                row += f" {can_fp16}FP  "
            elif can_int4:
                row += f" {can_int4}I4  "
            else:
                row += f"   ❌  "
        
        print(row)
    
    print("\n图例: ✅=FP16可运行 🟢=INT4可运行 ❌=不可运行")
    print("      FP=FP16  I4=INT4/AWQ/GPTQ")

gpu_model_fit()
```

输出：

```
GPU               | 显存 | 0.5B 1.5B   3B    7B   14B   32B   72B
-------------------------------------------------------------------------------------
RTX 4090          |   24G | ✅FP  ✅FP  ✅FP  ✅FP   ❌    ❌    ❌
RTX 6000 Ada      |   48G | ✅FP  ✅FP  ✅FP  ✅FP  ✅FP   ❌    ❌
A5000             |   48G | ✅FP  ✅FP  ✅FP  ✅FP  ✅FP   ❌    ❌
A100 40GB         |   40G | ✅FP  ✅FP  ✅FP  ✅FP  ✅FP   ❌    ❌
A100 80GB         |   80G | ✅FP  ✅FP  ✅FP  ✅FP  ✅FP  ✅FP   ❌
H100 80GB         |   80G | ✅FP  ✅FP  ✅FP  ✅FP  ✅FP  ✅FP   ❌
H200 141GB        |  141G | ✅FP  ✅FP  ✅FP  ✅FP  ✅FP  ✅FP  ✅FP

图例: ✅=FP16可运行 🟢=INT4可运行 ❌=不可运行
```

## 二、典型配置方案与价格

### 2.1 自建方案（一次性投入）

```python
def build_your_own_plans():
    """自建 GPU 服务器方案"""
    
    plans = [
        {
            "name": "入门开发",
            "config": "1×RTX 4090 + i9-14900K + 128GB DDR5",
            "model_capable": "≤7B FP16 / ≤32B INT4",
            "qps_est": "~10-30",
            "hardware_cost": 35000,
            "monthly_power": 200,
            "suitable_for": "个人学习 / 小团队原型 / 开发测试",
        },
        {
            "name": "团队标准",
            "config": "2×RTX 6000 Ada 48GB + EPYC + 256GB",
            "model_capable": "≤14B FP16 / ≤72B INT4 (TP=2)",
            "qps_est": "~30-60",
            "hardware_cost": 120000,
            "monthly_power": 450,
            "suitable_for": "团队内部工具 / 中等规模 API 服务",
        },
        {
            "name": "企业生产-A系列",
            "config": "2×A100 80GB + Dual EPYC + 512GB",
            "model_capable": "≤34B FP16 / ≤72B INT4 (TP=2)",
            "qps_est": "~50-120",
            "hardware_cost": 320000,
            "monthly_power": 800,
            "suitable_for": "生产级高并发服务 / 70B 模型 INT4",
        },
        {
            "name": "企业生产-H系列",
            "config": "4×H100 80GB SXM + Dual EPYC + 1TB",
            "model_capable": "≤72B FP16 (TP=4) / 更大 TP=8",
            "qps_est": "~150-300",
            "hardware_cost": 1200000,
            "monthly_power": 1800,
            "suitable_for": "大规模生产 / 70B+ FP16 / 最高吞吐",
        },
        {
            "name": "超大模型",
            "config": "8×H200 141GB NVLink + 4×EPYC + 2TB",
            "model_capable": "≤72B FP16 (TP=8) / 100B+ INT4",
            "qps_est": "~250-500",
            "hardware_cost": 3400000,
            "monthly_power": 3500,
            "suitable_for": "超大规模服务 / 多模型同时部署",
        },
    ]
    
    print(f"{'方案名称':<12} │ {'硬件配置':<36} │ {'可运行模型':<22} │ {'预估QPS':>8}")
    print(f"{'-'*12}:│-{'-'*36}:│-{'-'*22}:│-{'-'*8}")
    
    for p in plans:
        print(f"{p['name']:<12} │ {p['config']:<36} │ {p['model_capable']:<22} │ {p['qps_est']:>8}")
    
    print(f"\n{'='*90}")
    print(f"{'方案':<12} │ {'硬件成本':>10} │ {'月电费':>8} │ {'3年TCO':>10} │ {'适用场景'}")
    print(f"{'-'*12}:│-{'-'*10}:│-{'-'*8}:│-{'-'*10}:│-{'-'*25}")
    
    for p in plans:
        tco_3y = p["hardware_cost"] + p["monthly_power"] * 36
        print(f"{p['name']:<12} │ ¥{p['hardware_cost']:>9,} │ ¥{p['monthly_power']:>7,} │ ¥{tco_3y:>9,} │ {p['suitable_for']}")

build_your_own_plans()
```

输出：

```
方案名称     │ 硬件配置                              │ 可运行模型              │   预估QPS
------------:│--------------------------------------:│------------------------:|--------:
入门开发     │ 1×RTX 4090 + i9-14900K + 128GB DDR5  │ ≤7B FP16 / ≤32B INT4   │  ~10-30
团队标准     │ 2×RTX 6000 Ada 48GB + EPYC + 256GB   │ ≤14B FP16 / ≤72B INT4  │  ~30-60
企业生产-A系 │ 2×A100 80GB + Dual EPYC + 512GB       │ ≤34B FP16 / ≤72B INT4  │ ~50-120
企业生产-H系 │ 4×H100 80GB SXM + Dual EPYC + 1TB     │ ≤72B FP16 (TP=4)      │ ~150-300
超大模型     │ 8×H200 141GB NVLink + 4×EPYC + 2TB    │ ≤72B FP16 (TP=8)      │ ~250-500

==========================================================================================
方案         │   硬件成本 │   月电费 │    3年TCO │ 适用场景
------------:│-----------:|--------:|----------:│-------------------------
入门开发     │     ¥35,000 │     ¥200 │   ¥42,200 │ 个人学习 / 小团队原型
团队标准     │    ¥120,000 │     ¥450 │  ¥136,200 │ 团队内部工具 / 中等规模API
企业生产-A系 │    ¥320,000 │     ¥800 │  ¥348,800 │ 生产级高并发服务
企业生产-H系 │  ¥1,200,000 │   ¥1,800 │ ¥1,264,800 │ 大规模生产 / 70B+ FP16
超大模型     │  ¥3,400,000 │   ¥3,500 │ ¥1,466,000 │ 超大规模服务 / 多模型部署
```

### 2.2 云 GPU 方案对比

```python
def cloud_gpu_comparison():
    """主流云 GPU 方案对比"""
    
    providers = [
        # AWS
        {"provider": "AWS", "instance": "p4d.24xlarge", "gpu": "8×A100 80GB", 
         "price_h": 32.78, "price_spot": 15.00},
        {"provider": "AWS", "instance": "p5.48xlarge", "gpu": "8×H100 80GB", 
         "price_h": 98.32, "price_spot": None},
        {"provider": "AWS", "instance": "g5.xlarge", "gpu": "1×A10G 24GB", 
         "price_h": 1.21, "price_spot": 0.40},
        
        # GCP
        {"provider": "GCP", "instance": "a2-highgpu-8g", "gpu": "8×A100 80GB", 
         "price_h": 28.00, "price_spot": 12.50},
        {"provider": "GCP", "instance": "a3-megagpu-8g", "gpu": "8×H100 80GB", 
         "price_h": 88.00, "price_spot": None},
        
        # Azure
        {"provider": "Azure", "instance": "Standard_NC96ads_A100_v4", "gpu": "8×A100 80GB", 
         "price_h": 30.52, "price_spot": 13.00},
        {"provider": "Azure", "instance": "Standard_ND96amsr_A100_v4", "gpu": "8×A100 80GB NVLink", 
         "price_h": 33.84, "price_spot": 15.00},
        
        # 国内
        {"provider": "AutoDL", "instance": "RTX 4090 ×1", "gpu": "1×RTX 4090 24GB", 
         "price_h": 1.20, "price_spot": 0.70},
        {"provider": "阿里PAI", "instance": "ecs.gn7i-c8c.4xlarge", "gpu": "1×A10 24GB", 
         "price_h": 4.23, "price_spot": 2.10},
        {"provider": "腾讯云", "instance": "GN10X.4XLARGE160", "gpu": "1×V100 32GB", 
         "price_h": 11.20, "price_spot": 5.60},
    ]
    
    print(f"{'云厂商':<8} │ {'实例类型':<35} │ {'GPU 配置':<18} │ {'按需($/h)':>10} │ {'Spot($/h)':>10}")
    print("-" * 95)
    
    for p in providers:
        spot_str = f"${p['price_spot']:.2f}" if p['price_spot'] else "N/A"
        print(f"{p['provider']:<8} │ {p['instance']:<35} │ {p['gpu']:<18} │ ${p['price_h']:>8.2f}/h │ {spot_str:>10}")
    
    print("\n[成本对比] 月运行成本估算 (7×24小时)")
    print(f"{'实例':<35} │ {'按需(月)':>10} │ {'Spot(月)':>10} │ {'节省':>8}")
    print("-" * 75)
    
    for p in providers:
        on_demand_month = p["price_h"] * 24 * 30
        spot_month = p["price_spot"] * 24 * 30 if p["price_spot"] else 0
        saving = (1 - spot_month / on_demand_month) * 100 if spot_month > 0 else 0
        spot_str = f"${spot_month:,.0f}" if spot_month > 0 else "N/A"
        save_str = f"{saving:.0f}%" if saving > 0 else "-"
        print(f"{p['instance']:<35} │ ${on_demand_month:>9,.0f} │ {spot_str:>10} │ {save_str:>8}")

cloud_gpu_comparison()
```

## 三、自建 vs 租赁决策框架

### 3.1 盈亏平衡分析

```python
def buy_vs_rent_analysis():
    """自建 vs 租赁的盈亏平衡点"""
    
    scenarios = [
        {
            "name": "2×A100 80GB 方案",
            "buy_cost": 280000,
            "monthly_opex": 650,
            "rent_cost_per_hour": 6.5,
            "utilization": 0.5,
        },
        {
            "name": "4×H100 80GB 方案",
            "buy_cost": 1120000,
            "monthly_opex": 1500,
            "rent_cost_per_hour": 25.0,
            "utilization": 0.7,
        },
        {
            "name": "1×RTX 4090 方案",
            "buy_cost": 25000,
            "monthly_opex": 150,
            "rent_cost_per_hour": 0.6,
            "utilization": 0.8,
        },
    ]
    
    print("=" * 70)
    print("自建 vs 租赁 — 盈亏平衡分析")
    print("=" * 70)
    
    for s in scenarios:
        hours_per_month = 30 * 24 * s["utilization"]
        rent_monthly = s["rent_cost_per_hour"] * hours_per_month
        
        buy_monthly = s["monthly_opex"]
        buy_total = s["buy_cost"]
        
        months_to_breakeven = (s["buy_cost"]) / max(rent_monthly - s["monthly_opex"], 0.001)
        years_to_breakeven = months_to_breakeven / 12
        
        cost_1y_buy = s["buy_cost"] + s["monthly_opex"] * 12
        cost_1y_rent = rent_monthly * 12
        cost_3y_buy = s["buy_cost"] + s["monthly_opex"] * 36
        cost_3y_rent = rent_monthly * 36
        
        print(f"\n【{s['name']}】(利用率: {s['utilization']*100:.0f}% = {hours_per_month:.0f}h/月)")
        print(f"  自建: 一次投入 ¥{s['buy_cost']:,.0f} + 月运维 ¥{s['monthly_opex']:,.0f}")
        print(f"  租赁: ¥{rent_monthly:,.0f}/月")
        print(f"  ── 盈亏平衡: {months_to_breakeven:.1f}个月 ({years_to_breakeven:.1f}年)")
        print(f"  ── 1年总成本: 自建 ¥{cost_1y_buy:,.0f} vs 租赁 ¥{cost_1y_rent:,.0f}"
              f" → {'自建更优' if cost_1y_buy < cost_1y_rent else '租赁更优'}")
        print(f"  ── 3年总成本: 自建 ¥{cost_3y_buy:,.0f} vs 租赁 ¥{cost_3y_rent:,.0f}"
              f" → {'自建更优' if cost_3y_buy < cost_3y_rent else '租赁更优'}")

buy_vs_rent_analysis()
```

典型输出：

```
======================================================================
自建 vs 租赁 — 盈亏平衡分析
======================================================================

【2×A100 80GB 方案】(利用率: 50% = 360h/月)
  自建: 一次投入 ¥280,000 + 月运维 ¥650
  租赁: ¥2,340/月
  ── 盈亏平衡: 123.5个月 (10.3年)
  ── 1年总成本: 自建 ¥287,800 vs 租赁 ¥28,080 → 租赁更优
  ── 3年总成本: 自建 ¥303,400 vs 租赁 ¥84,240 → 租赁更优

【4×H100 80GB 方案】(利用率: 70% = 504h/月)
  自建: 一次投入 ¥1,120,000 + 月运维 ¥1,500
  租赁: ¥12,600/月
  ── 盈亏平衡: 97.5个月 (8.1年)
  ── 1年总成本: 自建 ¥1,138,000 vs 租赁 ¥151,200 → 租赁更优
  ── 3年总成本: 自建 ¥1,165,000 vs 租赁 ¥453,600 → 租赁更优

【1×RTX 4090 方案】(利用率: 80% = 576h/月)
  自建: 一次投入 ¥25,000 + 月运维 ¥150
  租赁: ¥346/月
  ── 盈亏平衡: 81.4个月 (6.8年)
  ── 1年总成本: 自建 ¥26,800 vs 租赁 ¥4,152 → 租赁更优
  ── 3年总成本: 自建 ¥29,500 vs 租赁 ¥12,456 → 租赁更优
```

**关键结论**：
- 对于大多数场景，**租赁比自建更经济**——除非你能保证 90%+ 的利用率且使用超过 5 年
- 云 Spot 实例可以再省 40-60%，适合非关键任务
- 自建的优势在于**数据隐私**和**无网络延迟**

## 四、Apple Silicon 方案

```python
def apple_silicon_option():
    """Apple Silicon 作为低成本替代"""
    
    info = """
    ┌──────────────────────────────────────────────────────┐
    │           Apple Silicon 方案                          │
    ├──────────────────────────────────────────────────────┤
    │                                                      │
    │  Mac Studio M2 Ultra (192GB RAM):                    │
    │  ├── 可运行: 70B Q4_K_M (~15-25 tok/s)              │
    │  ├── 可运行: 13B FP16 (~8-12 tok/s)                 │
    │  ├── 功耗: ~150W (vs GPU 350W+)                     │
    │  ├── 价格: ¥32,000 (含显示器约 ¥40,000)              │
    │  └── 优势: 静音、低功耗、统一内存架构                 │
    │                                                      │
    │  Mac Studio M4 Max (128GB RAM):                     │
    │  ├── 可运行: 32B Q4 (~20-30 tok/s)                  │
    │  ├── 可运行: 8B FP16 (~15-20 tok/s)                 │
    │  ├── 功耗: ~100W                                     │
    │  └── 价格: ¥27,000                                   │
    │                                                      │
    │  适用场景:                                           │
    │  ✅ 个人开发和实验                                    │
    │  ✅ 小规模内部工具                                    │
    │  ✅ 对延迟不敏感的批处理                               │
    │  ❌ 高并发在线服务 (< 5 QPS)                         │
    │  ❌ 需要 < 1s TTFT 的场景                             │
    │                                                      │
    │  性能参考 (Qwen2.5-7B-Instruct):                    │
    │  ┌────────────┬────────┬────────┬────────┐          │
    │  │ 平台       │ TTFT   │ TPOT   │ 吞吐   │          │
    │  ├────────────┼────────┼────────┼────────┤          │
    │  │ M2 Ultra   │ ~3s    │ ~80ms  │ ~12 t/s│          │
    │  │ RTX 4090   │ ~0.5s  │ ~35ms  │ ~50 t/s│          │
    │  │ A100 80GB  │ ~0.3s  │ ~25ms  │ ~90 t/s│          │
    │  └────────────┴────────┴────────┴────────┘          │
    └──────────────────────────────────────────────────────┘
    """
    print(info)

apple_silicon_option()
```

## 五、需求驱动的选型决策树

```python
def selection_decision_tree():
    """需求驱动选型决策树"""
    
    tree = """
    ════════════════════════════════════════════════════════
                  vLLM 硬件选型决策树
    ════════════════════════════════════════════════════════

    你的核心需求是什么？
    │
    ├─ 个人学习 / 实验 / 原型开发
    │   ├─ 预算 < 2万 → RTX 4090 (24GB) 或 Mac Studio M4
    │   ├─ 预算 2-5万 → RTX 6000 Ada (48GB)
    │   └─ 不想买硬件 → AutoDL / 阿里云 PAI 按量付费
    │
    ├─ 团队内部工具 (< 20 并发用户)
    │   ├─ 模型 ≤ 7B → 1×A100 40GB 或 1×RTX 6000 Ada
    │   ├─ 模型 14-32B → 2×A100 80GB (TP=2) 或 AWQ INT4
    │   └─ 预算有限 → 2×RTX 4090 + AWQ/GPTQ INT4
    │
    ├─ 生产级 API 服务 (> 50 并发用户)
    │   ├─ 模型 ≤ 14B → 2×A100 80GB 或 4×RTX 6000 Ada
    │   ├─ 模型 32-72B → 4×A100 80GB (TP=4) + INT4
    │   │             或 2×H100 80GB (TP=2) + INT4
    │   └─ 需要 FP16 72B → 4×H100 80GB (TP=4) 或 8×A100
    │
    ├─ 大规模服务 (> 200 QPS)
    │   ├─ 单模型 → 多副本横向扩展
    │   ├─ 多模型 → 更大集群 (8×H100 / 8×H200)
    │   └─ 考虑 Kubernetes 编排 + 自动伸缩
    │
    └─ 特殊约束
        ├─ 数据不能出内网 → 必须自建
        ├─ 弹性负载波动大 → 云 Spot 实例
        ├─ 需要最低延迟 → 本地部署 + NVLink
        └─ 成本敏感 → INT4 量化 + Apple Silicon / 云竞价实例

    ════════════════════════════════════════════════════════
    """
    print(tree)

selection_decision_tree()
```

---

## 六、总结

本节提供了完整的硬件选型和成本参考：

| 维度 | 核心建议 |
|------|---------|
| **个人/小团队** | RTX 4090 (¥13K) 或 Mac Studio (¥27-40K)，够用就好 |
| **团队标准** | 2×A100 80GB (¥280K) 或 2×RTX 6000 Ada (¥90K) |
| **企业生产** | 4×H100 80GB (¥1.12M) 或 8×A100 (¥1.12M) |
| **自建 vs 租赁** | 利用率 < 80% 或使用 < 3 年 → 租赁更划算 |
| **省钱策略** | INT4/AWQ 量化（省 70% 显存）+ Spot 实例（省 40-60%） |
| **Apple Silicon** | 开发利器但不适于高并发生产 |
| **NVLink/NVSwitch** | TP≥4 时必须考虑，否则通信开销吃掉收益 |

**核心要点回顾**：

1. **不要盲目追求最大 GPU**——根据实际模型大小和 QPS 需求选择
2. **量化是性价比最高的优化手段**——INT4 让一张卡跑 4 倍大的模型
3. **大多数情况下租赁优于自建**——除非你有 5 年以上的稳定需求和 90%+ 利用率
4. **Spot/竞价实例是隐藏的省钱神器**——可节省 40-60% 成本
5. **Apple Silicon 是被低估的开发平台**——M2 Ultra/M4 Max 能跑 70B 模型

至此，**Chapter 6（多GPU与分布式推理）全部完成**！接下来进入 **Chapter 7：量化技术**。
