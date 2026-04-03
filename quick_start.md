# Quick Start

本文档会带你从零开始把系统跑起来，生成第一批 RP 训练数据。

---

## 前置条件

- Python >= 3.10
- 至少一个 OpenAI-compatible API 的 key（OpenAI / DeepSeek / 通义千问 / 任何兼容 `/v1/chat/completions` 接口的服务）

---

## 第一步：安装

```bash
git clone <repo-url>
cd rp-factory
pip install -e .
```

验证安装成功：

```bash
rp-factory --help
```

你应该看到这样的输出：

```
Usage: rp-factory [OPTIONS] COMMAND [ARGS]...

  RP 数据合成工厂 v3.2 ...

Options:
  -c, --config TEXT  配置文件路径 (默认 config/default.yaml)
  -v, --verbose      启用详细日志
  --help             Show this message and exit.

Commands:
  batch          批量生成数据
  gen-personas   LLM 自动生成多样化角色 System Prompt
  generate       从单个角色 System Prompt 文件生成对话数据
  inspect        检视已生成的 JSONL 数据文件
  show-config    显示当前生效的配置
```

---

## 第二步：配置 API Key

系统需要调用 LLM API。最简配置只需要一个环境变量：

```bash
export OPENAI_API_KEY="sk-..."
```

设好之后，**系统默认会用这一个 key 驱动所有模块**（User 发言生成、角色回复生成、Mentor 考点拆解、质检判断）。

### 想用不同模型？

编辑 `config/default.yaml`。系统有三个 LLM 角色，各司其职：

```yaml
llm:
  # generator: 负责生成 User 发言、种子扩充、角色人设
  # 需要有创意但不需要推理能力，用 gpt-4o 或同档模型即可
  generator:
    model: "gpt-4o"
    base_url: null                  # null = 默认 OpenAI 地址
    api_key_env: "OPENAI_API_KEY"   # 从这个环境变量读 key
    temperature: 0.9
    max_tokens: 2048

  # teacher: 负责以角色身份回复用户，是生成数据的核心
  # 强烈建议用带 reasoning_content 能力的模型（DeepSeek-R1 / o1 / o3）
  # 这样落盘数据会包含推理链，训练价值更高
  teacher:
    model: "deepseek-reasoner"
    base_url: "https://api.deepseek.com/v1"
    api_key_env: "DEEPSEEK_API_KEY"
    temperature: 0.7
    max_tokens: 4096

  # mentor: 负责 Pipeline A 的考点拆解和质检的 AI 味检测
  # 需要分析能力，用 gpt-4o 即可
  mentor:
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
    temperature: 0.3
    max_tokens: 2048
```

> **所有端点必须兼容 OpenAI Chat Completions API 格式**（`/v1/chat/completions`）。如果你用的模型不原生支持这个格式（比如 Anthropic Claude），需要通过 LiteLLM 或 One API 等代理转换。

---

## 第三步：跑第一条数据

项目自带两个示例角色文件。试一下：

```bash
rp-factory generate examples/surgeon.txt
```

这会：
1. 读取 `examples/surgeon.txt` 中的角色 System Prompt（一个高冷的外科医生）
2. 用冰山三步法生成 6 轮用户发言
3. 用 Teacher 模型以角色身份回复
4. 对每轮回复做质检（正则黑名单 + AI 味检测）
5. 输出到 `output/output.jsonl`

查看结果：

```bash
rp-factory inspect output/output.jsonl
```

你会看到类似这样的输出：

```
共 1 条记录

━━━ 样本 1 ━━━
  [SYSTEM] 你是一名顶尖的外科医生，性格极其冷酷理性...
  [USER] 我真是个纯废物，连杯咖啡都拿不稳！...
  [REASONING] 用户表面在抱怨咖啡洒了，但情绪强度明显不匹配...
  [ASSISTANT] 啧，一杯咖啡而已，至于把自己骂成废物？...
  pipeline=pipeline_a | quality=pass
```

### 常用参数

```bash
# 指定轮数（默认 6）
rp-factory generate examples/surgeon.txt -t 4

# 指定输出路径（支持绝对路径）
rp-factory generate examples/surgeon.txt -o /tmp/my_data.jsonl

# 看详细日志
rp-factory -v generate examples/surgeon.txt
```

---

## 第四步：批量生成

### 方式 A：提供角色文件

准备一个目录，每个 `.txt` 文件是一个角色的 System Prompt：

```bash
mkdir my_roles
echo "你是一个毒舌的花艺师..." > my_roles/florist.txt
echo "你是一个沉默的天文台看守..." > my_roles/astronomer.txt

rp-factory batch -d my_roles/ -n 10
```

`-d` 指定角色目录，`-n` 指定要生成多少条对话。目录里只有 2 个角色但要 10 条数据？系统会自动用 LLM 生成 8 个新角色来补齐。

### 方式 B：完全自动（零手写）

不提供任何角色文件，全部由 LLM 动态生成：

```bash
rp-factory batch -n 50
```

系统会：
1. 调用 `PersonaGenerator` 生成 50 个差异化角色（基于职业 × 时代 × 性格 × 说话风格的随机组合约束）
2. 检查种子池（深层动机、表面事件）容量，不够就调 LLM 扩充
3. 对每个角色跑完整管线
4. 输出两个文件到 `output/` 目录：
   - `batch_output.jsonl` — 完整版，含 `_meta` 溯源元数据
   - `batch_output_train.jsonl` — 训练版，只有纯净 `messages`，可直接用于 SFT

运行结束后会打印统计表：

```
       批量生成结果
┌─────────────────┬──────┐
│ 指标            │    值 │
├─────────────────┼──────┤
│ 目标数量        │    50 │
│ 通过质检        │    47 │
│ 废弃率          │  6.0% │
│ 种子池 (intents)│    38 │
│ 种子池 (events) │    72 │
│ 种子池 (styles) │    15 │
└─────────────────┴──────┘
```

---

## 第五步：单独生成角色

如果你想先审核角色质量再跑数据：

```bash
# 生成 20 个角色，保存到 my_personas/ 目录
rp-factory gen-personas -n 20 -o my_personas/

# 人工审核、筛选、编辑 .txt 文件...

# 用审核后的角色跑数据
rp-factory batch -d my_personas/ -n 20
```

不带 `-o` 则直接打印到终端预览：

```bash
rp-factory gen-personas -n 3
```

---

## 输出格式说明

### 完整版 JSONL（含溯源）

每行一个 JSON 对象：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个高冷的外科医生..."
    },
    {
      "role": "user",
      "content": "我真是个纯废物，连杯冰美式都拿不稳！..."
    },
    {
      "role": "assistant",
      "content": "啧，一杯咖啡而已...",
      "reasoning_content": "用户表面在抱怨咖啡，但情绪强度远超事件本身..."
    }
  ],
  "_meta": {
    "id": "a1b2c3d4e5f6",
    "created_at": "2026-04-03T...",
    "system_prompt_source": "你是一个高冷的外科医生...",
    "pipeline": "pipeline_a",
    "perturbations": [],
    "quality": {"verdict": "pass"}
  }
}
```

字段说明：
- `messages` — OpenAI 标准格式，可直接用于训练
- `reasoning_content` — Teacher 模型的推理链（只有用带 reasoning 能力的模型才会有值）
- `_meta` — 溯源信息，**不进入训练**，仅用于事后分析
- `_meta.pipeline` — 这条数据是 `pipeline_a`（Mentor 指导）还是 `pipeline_b`（Best-of-N 采样）生成的
- `_meta.perturbations` — 对话中注入了哪些扰动（风味任务 / 记忆投毒 / 认知转译）

### 训练版 JSONL（纯净）

只保留 `{"messages": [...]}` ，没有 `_meta`。可以直接喂给 SFT 训练框架。

---

## 配置参考

查看当前生效的完整配置：

```bash
rp-factory show-config
```

常用可调参数：

| 配置路径 | 含义 | 默认值 |
|---------|------|--------|
| `llm.teacher.model` | Teacher 模型名 | deepseek-reasoner |
| `pipeline.conversation_turns` | 每条对话轮数 | 6 |
| `pipeline.max_concurrent_conversations` | 批量时并发数 | 5 |
| `rp_agent.mix_ratio.pipeline_a` | Pipeline A 占比（0~1） | 0.5 |
| `rp_agent.pipeline_b.n_samples` | Pipeline B 每轮采样数 | 8 |
| `rp_agent.pipeline_b.temperature` | Pipeline B 采样温度 | 0.9 |
| `context_control.flavored_task.injection_probability` | 风味任务注入概率 | 0.3 |
| `context_control.flavored_task.injection_rounds` | 哪些轮次可能注入任务 | [3, 4, 5] |
| `context_control.memory_poisoning.injection_probability` | 记忆投毒概率 | 0.25 |
| `quality.level1.persona_blacklist_patterns` | AI 味正则黑名单 | 见 default.yaml |
| `quality.level1.payload_lint_enabled` | 代码块穿透检测开关 | true |
| `user_agent.iceberg.diversity.user_styles` | 用户风格标签池 | 7 种 |
| `user_agent.iceberg.diversity.ngram_collapse_threshold` | n-gram 塌缩警告阈值 | 0.7 |

使用自定义配置：

```bash
cp config/default.yaml config/my_config.yaml
# 编辑 my_config.yaml ...
rp-factory -c config/my_config.yaml batch -n 100
```

---

## Python API

如果你不用 CLI 而是在自己的代码里调用：

```python
import asyncio
from rp_factory import DataFactory, load_config

config = load_config()  # 或 load_config("path/to/config.yaml")
factory = DataFactory(config)

# 单条对话
record = asyncio.run(factory.generate_conversation(
    system_prompt="你是一个高冷的外科医生...",
    num_turns=4,
))

if record:
    for msg in record.messages:
        print(f"[{msg.role.value}] {msg.content[:80]}")

# 批量生成（自动生成角色 + 种子扩充）
records = asyncio.run(factory.run_batch(count=20, num_turns=4))
print(f"生成了 {len(records)} 条数据")
```

---

## 目录结构

```
├── config/default.yaml            默认配置
├── seeds/
│   ├── deep_intents.yaml          深层动机种子（32 条）
│   ├── proxy_events.yaml          表面事件种子（65 条）
│   └── flavored_tasks.yaml        风味任务种子（33 条）
├── examples/
│   ├── surgeon.txt                示例角色：冷酷外科医生
│   └── tang_poet.txt              示例角色：唐代诗人李白
├── src/rp_factory/
│   ├── pipeline.py                管线编排器
│   ├── prompts.py                 所有 Prompt 模板
│   ├── config.py                  配置定义
│   ├── models.py                  数据模型
│   ├── llm_client.py              LLM 调用封装
│   ├── generators.py              LLM 动态生成器（角色 + 种子）
│   ├── diversity.py               Pool + n-gram 监控
│   ├── cli.py                     CLI 入口
│   ├── user_agent/iceberg.py      冰山三步法
│   ├── context_control/perturbation.py  扰动注入
│   ├── rp_agent/pipeline_a.py     Mentor 指导管线
│   ├── rp_agent/pipeline_b.py     Best-of-N 采样管线
│   ├── quality/filters.py         质检漏斗
│   ├── evaluation/evaluator.py    评估闭环
│   └── output/serializer.py       JSONL 序列化
└── tests/                         52 个单元测试
```
