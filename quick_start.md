# Quick Start

## 1. 安装

```bash
git clone <repo-url> && cd rp-factory
pip install -e .
```

验证安装：

```bash
rp-factory --help
```

## 2. 配置 API Key

系统通过 OpenAI-compatible API 调用 LLM。**最少只需一个 key** 即可跑通：

```bash
export OPENAI_API_KEY="sk-..."
```

默认配置中，generator / mentor / teacher 三个角色都可以指向同一个 OpenAI-compatible 端点。如果你的不同模型在不同的服务上，修改 `config/default.yaml`：

```yaml
llm:
  generator:                           # User 发言生成 + 种子扩充
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
  teacher:                             # RP 角色回复生成（核心）
    model: "deepseek-reasoner"
    base_url: "https://api.deepseek.com/v1"
    api_key_env: "DEEPSEEK_API_KEY"
  mentor:                              # Pipeline A 考点拆解 + 质检判断
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
```

> **所有端点必须兼容 OpenAI Chat Completions API 格式。**
>
> **Teacher 模型**建议使用带 `reasoning_content` 能力的模型（如 DeepSeek-R1、OpenAI o1/o3），这样生成的训练数据会包含推理链。

## 3. 最快体验：单角色生成

项目自带示例角色。生成一条 6 轮对话：

```bash
rp-factory generate examples/surgeon.txt
```

指定轮数和输出路径：

```bash
rp-factory generate examples/surgeon.txt -t 4 -o /tmp/test.jsonl
```

查看产出：

```bash
rp-factory inspect output/output.jsonl
```

## 4. 批量生成：手动提供角色

准备一个目录，每个 `.txt` 文件是一个角色的 System Prompt：

```bash
rp-factory batch -d examples/ -n 10 -t 4
```

`-d` 指定角色目录，`-n` 指定目标数量。如果目录里的 `.txt` 文件不足 `-n` 个，系统会自动调用 LLM 生成新角色补齐。

## 5. 全自动模式：零手写输入

不提供任何角色文件，完全由 LLM 动态生成角色 + 种子 + 对话：

```bash
rp-factory batch -n 50
```

系统会：
1. 调用 `PersonaGenerator` 生成 50 个差异化角色（基于职业/时代/性格/说话风格维度约束）
2. 检查种子池容量，不足时调用 `SeedExpander` 扩充
3. 对每个角色运行完整管线（冰山法 → 扰动注入 → Pipeline A/B → 质检）
4. 输出两个文件：
   - `output/batch_output.jsonl` — 含 `_meta` 溯源元数据
   - `output/batch_output_train.jsonl` — 纯 `messages`，可直接用于 SFT

## 6. 单独生成角色

如果你只想生成角色设定文件，后续再跑数据：

```bash
rp-factory gen-personas -n 20 -o my_personas/
```

然后：

```bash
rp-factory batch -d my_personas/ -n 20
```

## 7. 自定义配置

```bash
cp config/default.yaml config/my_config.yaml
# 编辑 ...
rp-factory -c config/my_config.yaml batch -n 100
```

常用参数：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `pipeline.conversation_turns` | 每条对话轮数 | 6 |
| `pipeline.max_concurrent_conversations` | 并发数 | 5 |
| `rp_agent.mix_ratio.pipeline_a` | Pipeline A 占比 | 0.5 |
| `rp_agent.pipeline_b.n_samples` | Best-of-N 采样数 | 8 |
| `context_control.flavored_task.injection_probability` | 风味任务注入概率 | 0.3 |
| `context_control.memory_poisoning.injection_probability` | 记忆投毒概率 | 0.25 |
| `quality.level1.payload_lint_enabled` | 代码块穿透检测 | true |

## 8. 输出格式

### 完整版（含溯源元数据）

每行一个 JSON：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个高冷的外科医生..."},
    {"role": "user", "content": "我今天真的太倒霉了..."},
    {"role": "assistant", "content": "...", "reasoning_content": "..."}
  ],
  "_meta": {
    "id": "a1b2c3d4e5f6",
    "pipeline": "pipeline_a",
    "perturbations": [],
    "quality": {"verdict": "pass", "gain_delta": 0.75}
  }
}
```

### 训练版（纯净）

只保留 `messages` 数组，可直接用于 SFT 训练。

## 9. Python API

```python
import asyncio
from rp_factory import DataFactory, load_config

config = load_config()
factory = DataFactory(config)

# 单条对话
record = asyncio.run(factory.generate_conversation(
    system_prompt="你是一个高冷的外科医生...",
    num_turns=4,
))

# 批量（自动生成角色）
records = asyncio.run(factory.run_batch(count=20, num_turns=4))
```
