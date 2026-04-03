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

系统需要调用 LLM API。**最少只需一个 key** 即可跑通全流程（让所有角色共用同一个 OpenAI-compatible 端点）。

```bash
export OPENAI_API_KEY="sk-..."
```

如果你有多个模型分别充当不同角色（generator / teacher / mentor / target），在 `config/default.yaml` 中分别配置：

```yaml
llm:
  generator:
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
  teacher:
    model: "claude-3-5-sonnet-20241022"
    base_url: "https://api.anthropic.com/v1"   # 如果走 OpenAI-compatible 代理
    api_key_env: "ANTHROPIC_API_KEY"
  mentor:
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
  target:
    model: "qwen-max"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "TARGET_MODEL_API_KEY"
```

> **注意**：`LLMClient` 底层使用的是 `openai.AsyncOpenAI`，所以所有模型端点必须兼容 OpenAI Chat Completions API 格式。Anthropic Claude 需要通过 OpenAI-compatible 代理（如 LiteLLM / One API）接入，或者将 teacher 也配成一个 OpenAI 格式的端点。

## 3. 最快体验：单角色生成

项目自带示例角色文件。生成一条 6 轮对话数据：

```bash
rp-factory generate examples/surgeon.txt -t 6 -o test_output.jsonl
```

查看产出：

```bash
rp-factory inspect output/test_output.jsonl
```

## 4. 批量生成：手动提供角色

准备一个目录，每个 `.txt` 文件是一个角色的 System Prompt：

```bash
rp-factory batch examples/ -n 10 -t 4 -o batch_result.jsonl
```

这会加载 `examples/` 下的所有 `.txt` 角色。如果文件数量不足 `-n` 指定的数量，系统会自动调用 LLM 生成新角色补齐。

## 5. 全自动模式：零手写输入

不提供任何角色文件，完全由 LLM 动态生成角色 + 种子 + 对话：

```bash
rp-factory batch -n 50
```

系统会：
1. 调用 `PersonaGenerator` 生成 50 个差异化角色
2. 调用 `warmup_seeds` 预判种子池容量，不足时调用 `SeedExpander` 扩充
3. 对每个角色运行完整管线（冰山法 → 扰动注入 → Pipeline A/B → 质检）
4. 输出 `output/batch_output.jsonl`（含 `_meta`）和 `output/batch_output_train.jsonl`（纯训练数据）

## 6. 单独生成角色

如果你只想生成角色设定保存下来，后续再跑数据：

```bash
rp-factory gen-personas -n 20 -o my_personas/
```

然后：

```bash
rp-factory batch my_personas/ -n 20
```

## 7. 自定义配置

复制默认配置后修改：

```bash
cp config/default.yaml config/my_config.yaml
# 编辑 my_config.yaml
rp-factory -c config/my_config.yaml batch -n 100
```

常用的可调参数：

| 参数路径 | 含义 | 默认值 |
|---------|------|--------|
| `pipeline.conversation_turns` | 每条对话的轮数 | 6 |
| `pipeline.max_concurrent_conversations` | 并发对话数 | 5 |
| `rp_agent.mix_ratio.pipeline_a` | Pipeline A 占比 | 0.5 |
| `rp_agent.pipeline_b.n_samples` | Best-of-N 采样数 | 8 |
| `context_control.flavored_task.injection_probability` | 风味任务注入概率 | 0.3 |
| `context_control.memory_poisoning.injection_probability` | 记忆投毒概率 | 0.25 |
| `quality.level2.enabled` | 是否启用增益过滤 | true |

## 8. 输出格式

### 完整版（含溯源元数据）

`output/batch_output.jsonl` — 每行一个 JSON：

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

`output/batch_output_train.jsonl` — 只保留 `messages`，可直接用于 SFT 训练。

## 9. Python API

```python
import asyncio
from rp_factory import DataFactory, load_config

config = load_config("config/default.yaml")
factory = DataFactory(config)

# 生成单条对话
record = asyncio.run(factory.generate_conversation(
    system_prompt="你是一个高冷的外科医生...",
    num_turns=4,
))

# 批量生成（自动生成角色）
records = asyncio.run(factory.run_batch(count=20, num_turns=4))
```

---

# 我在写这个 Quick Start 过程中发现的问题

以下是我逐步走查代码时发现的不严谨或需要你确认的地方：

## 问题 1：Teacher Model (Claude) 的接入方式不明确

`llm_client.py` 底层写死了 `openai.AsyncOpenAI`。配置里 teacher 默认填的是 `claude-3-5-sonnet-20241022`，但 Anthropic 原生 API 不兼容 OpenAI 格式。

**现状**：如果用户直接填 Anthropic 的 key，调用会报错。必须走 OpenAI-compatible 代理。但这个前提条件在配置和文档中都没有说清楚。

**需要确认**：
- 你们的实际部署里 Claude 是怎么接的？是 LiteLLM / One API / 自建代理？
- 还是说实际上所有模型都会走同一个 OpenAI-compatible 的转发服务？
- 需不需要我在 `LLMClient` 里加一层对 Anthropic SDK 的原生适配？

## 问题 2：四个 LLM 角色的边界定义含糊

配置里定义了 4 个 LLM 端点（generator / teacher / mentor / target），但：
- **generator** 和 **mentor** 在默认配置里指向同一个模型（gpt-4o），代码里也确实让 mentor 复用了 generator 的 key。那为什么要分成两个端点？
- **target** 是用于增益过滤的"基座模型"——但如果用户还没开始微调，target 该填什么？填一个弱模型？填和 teacher 一样的模型？inspire.md 说的是"当前正在微调的基座模型"，但在 MVP 阶段这个模型可能还不存在。

**需要确认**：
- 最简配置（只有一个 API key）能跑通所有流程吗？现在代码会给所有 4 个端点都创建 client，如果 target 的 key 为空，level2 增益过滤会被跳过（有 `if not self.target_llm` 的 guard），但 `LLMClient.__init__` 里会用 `"placeholder"` 作为 key 创建 client，所以 `target_llm` 永远不是 None——这意味着会尝试调用 target 模型然后报错。
- 你希望 target 为空时是静默跳过 L2，还是报错提示？

## 问题 3：`generate` 命令输出路径的语义不一致

`generate` 命令的 `-o` 参数传给 `Serializer.write_jsonl` 时是一个 **文件名**（不是路径），最终文件会写到 `config.output.output_dir / filename`（默认 `output/output.jsonl`）。但 CLI help 说的是"输出文件名"，用户可能会传一个完整路径如 `/tmp/result.jsonl`，此时它会被当作文件名拼到 `output/` 下变成 `output//tmp/result.jsonl`，这会崩。

**需要确认**：`-o` 应该是纯文件名还是完整路径？

## 问题 4：`reasoning_content` 的获取依赖特殊 API 能力

代码里用 `getattr(msg, "reasoning_content", None)` 尝试拿推理链。这个字段只有特定模型（如 DeepSeek-R1、OpenAI o1/o3）才会在 response 里返回。普通 GPT-4o / Claude 不会有这个字段。

**现状**：如果 teacher 是 GPT-4o，所有落盘数据的 `reasoning_content` 都会是 `null`。但 inspire.md 明确要求生成带有 reasoning_content 的数据。

**需要确认**：
- 你们的 teacher 模型实际用的是什么？是 DeepSeek-R1 这类有原生 reasoning 的模型？
- 还是说需要在 prompt 里引导模型自己输出一段推理，然后代码层面把它拆成 reasoning_content + content？
- 如果是后者，这是一个需要实现的功能点。

## 问题 5：Pipeline B 的 API 成本问题没有在文档中提醒

Pipeline B 对每轮对话会并发 N 次请求（默认 N=8），再加一次 Verifier 调用。一条 6 轮对话经过 Pipeline B 会产生 6×8+6 = **54 次 API 调用**。再加上 Pipeline A 的 Mentor + Teacher（6×2=12），冰山三步法每轮 3 次调用（6×3=18），加上 L2 增益过滤（2 次）。

一条对话最多可能产生 **70+ 次 API 调用**。50 条对话就是 3500+ 次。用 GPT-4o 这个费用不低。

**需要确认**：是否需要在 Quick Start 中给出费用估算或建议？比如建议开发调试阶段用便宜的模型？

## 问题 6：`batch` 命令不传 prompts_dir 时 Click 的行为

`prompts_dir` 声明为 `type=click.Path(exists=True), required=False, default=None`。当用户执行 `rp-factory batch -n 100` 时，Click 会把 `-n` 后面的参数解析为 option 而不是 positional argument，所以应该没问题。但如果用户执行 `rp-factory batch -n 100 some_dir`，`some_dir` 会被当作 prompts_dir 且 Click 会校验它必须存在。这个行为对吗？还是说应该让 prompts_dir 变成 option 而不是 argument？

## 问题 7：`tiktoken` 和 `numpy` 声明了依赖但从未使用

`pyproject.toml` 里声明了 `tiktoken>=0.5.0` 和 `numpy>=1.24.0`，`aiohttp>=3.9.0`，但代码中没有任何文件 import 它们。这三个都是重量级依赖。

**需要确认**：是否有计划使用它们（比如 token 计数、向量化多样性检测）？如果目前没有就应该删掉。
