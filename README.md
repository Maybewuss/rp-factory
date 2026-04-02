# RP 数据合成工厂 v3.1

> 基于认知仿真与对偶动机的高质量 RP 训练数据生成系统

## 概述

本项目实现了 [inspire.md](inspire.md) 中描述的完整数据合成管线，用于生成具有极高训练梯度增益的 RP（角色扮演）对话数据。生成的数据格式为包含 `reasoning_content` 的 OpenAI 标准 `messages` 数组。

**核心理念：生成有灵魂的数据，不是有剧本的数据。**

## 架构

系统由六大模块组成，对应 `inspire.md` 的六个章节：

```
┌─────────────────────────────────────────────────────────────┐
│                    管线编排器 (pipeline.py)                    │
├──────────┬───────────┬───────────┬──────────┬───────────────┤
│ 模块1     │ 模块2      │ 模块3      │ 模块4     │ 模块5+6       │
│ User     │ 语境控制    │ RP Agent  │ 质检漏斗  │ 评估 + 落盘    │
│ Agent    │ 扰动注入    │ A/B 管线   │ 两级过滤  │ JSONL 输出    │
│ 冰山三步法 │            │           │          │               │
└──────────┴───────────┴───────────┴──────────┴───────────────┘
```

### 模块 1: User Agent 引擎

基于 **冰山理论 (The Iceberg Theory)** 生成用户发言：

1. **动机反推** — 基于角色设定反推用户的深层心理动机
2. **实体锚定** — 生成琐碎事件作为情绪伪装载体
3. **加密生成** — 融合动机与事件，输出高熵的用户台词

支持多样性风格注入（话少冷淡型、暴躁直接型、故作轻松型等）。

### 模块 2: 语境控制与异常处理

三种扰动机制确保数据的复杂度和鲁棒性：

- **风味任务注入** — 在情绪对话中突然插入工具性任务（翻译、写代码等），测试人设包裹能力
- **记忆投毒** — 篡改历史对话细节，训练模型的纠偏肌肉记忆（分级：critical/trivial/correct）
- **认知转译** — 对历史/古代角色注入现代知识问题，测试优雅降级策略

### 模块 3: RP Agent 推理生成引擎

并行运行两条生成管线：

- **Pipeline A (Mentor 指导)** — Mentor Agent 提供考点拆解 → Teacher Model 内化 Hint 生成回复 → 销毁脚手架
- **Pipeline B (Best-of-N)** — 高温采样 N 个候选 → Verifier 法官筛选钻石级数据

MVP 阶段建议 50/50 配比，通过人工盲测确定量产比例。

### 模块 4: Rejection Sampling 与质检漏斗

- **第一级：人设 + 任务校验** — AI 味黑名单正则 + 代码块人设穿透检测
- **第二级：目标模型增益过滤** — Target Model 先跑一遍，对比 Teacher 回复是否有训练增益

### 模块 5: 评估闭环

五维度评分体系：深层需求识别、人设一致性、风味任务完成、事实纠偏、认知转译。

### 模块 6: 数据落盘

输出纯净 JSONL：`messages` 数组 + `_meta` 溯源元数据（不进入训练）。

## 项目结构

```
├── inspire.md                    # 设计规范文档
├── config/
│   └── default.yaml              # 默认配置
├── seeds/                        # 种子数据库（防止模式塌缩）
│   ├── deep_intents.yaml         # 深层动机种子
│   ├── proxy_events.yaml         # 表面事件种子
│   └── flavored_tasks.yaml       # 风味任务种子
├── examples/                     # 示例角色 System Prompt
│   ├── surgeon.txt               # 冷酷外科医生
│   └── tang_poet.txt             # 唐代诗人李白
├── src/rp_factory/
│   ├── models.py                 # 数据模型（Message, Perturbation 等）
│   ├── config.py                 # 配置加载与管理
│   ├── llm_client.py             # 统一 LLM 调用层
│   ├── pipeline.py               # 管线编排器
│   ├── cli.py                    # CLI 入口
│   ├── user_agent/
│   │   └── iceberg.py            # 冰山三步法引擎
│   ├── context_control/
│   │   └── perturbation.py       # 扰动引擎
│   ├── rp_agent/
│   │   ├── pipeline_a.py         # Mentor 指导管线
│   │   └── pipeline_b.py         # Best-of-N 采样管线
│   ├── quality/
│   │   └── filters.py            # 两级质检漏斗
│   ├── evaluation/
│   │   └── evaluator.py          # 评估闭环
│   └── output/
│       └── serializer.py         # JSONL 序列化
└── tests/                        # 单元测试
```

## 安装

```bash
pip install -e ".[dev]"
```

## 配置

编辑 `config/default.yaml` 或创建自定义配置文件。需要设置的环境变量：

```bash
export OPENAI_API_KEY="your-openai-key"        # Generator / Mentor / Judge
export ANTHROPIC_API_KEY="your-anthropic-key"   # Teacher Model
export TARGET_MODEL_API_KEY="your-target-key"   # 目标基座模型
```

## 使用

### 单角色生成

```bash
rp-factory generate examples/surgeon.txt -t 6 -o surgeon_data.jsonl
```

### 批量生成

```bash
rp-factory batch examples/ -t 6 -o batch_output.jsonl -n 100
```

### 检视数据

```bash
rp-factory inspect output/surgeon_data.jsonl -s 5
```

### 查看配置

```bash
rp-factory show-config
```

### Python API

```python
import asyncio
from rp_factory import DataFactory, load_config

config = load_config()
factory = DataFactory(config)

system_prompt = "你是一个高冷的外科医生..."
record = asyncio.run(factory.generate_conversation(system_prompt, num_turns=6))

if record:
    for msg in record.messages:
        print(f"[{msg.role.value}] {msg.content[:100]}")
```

## 输出格式

每条落盘数据的结构：

```json
{
  "messages": [
    {"role": "system", "content": "角色设定..."},
    {"role": "user", "content": "用户发言..."},
    {
      "role": "assistant",
      "content": "角色回复...",
      "reasoning_content": "内部推理链..."
    }
  ],
  "_meta": {
    "id": "a1b2c3d4e5f6",
    "created_at": "2026-04-02T...",
    "pipeline": "pipeline_b",
    "perturbations": [...],
    "quality": {"verdict": "pass", "gain_delta": 0.75}
  }
}
```

## 测试

```bash
pytest tests/ -v
```

## 设计文档

完整的设计规范和工程 SOP 请参见 [inspire.md](inspire.md)。
