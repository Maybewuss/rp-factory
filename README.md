## rp-factory

一个基于 `inspire.md` 落地的最小可运行 MVP，用来演示：

- 用户冰山三步法生成
- RP 回复的 Pipeline A / Pipeline B 双路线采样
- 一级质检漏斗
- 目标模型闭环过滤与批次多样性统计
- OpenAI 风格 JSONL 落盘（含 `reasoning_content` 与 `_meta`）

## 快速开始

要求：

- Python 3.12+

生成示例数据：

```bash
python3 -m src.rp_factory.cli build \
  --input examples/scenarios.json \
  --output output/dataset.jsonl \
  --report output/report.json
```

运行测试：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

查看帮助：

```bash
python3 -m src.rp_factory.cli --help
```

## 当前实现范围

这个仓库当前提供的是一个离线可运行骨架，不依赖外部模型 API：

- 用规则式但可组合的组件模拟 `Deep_Intent -> Proxy_Event -> User_Message`
- 用两条可扩展管线模拟 Teacher / Mentor 协作与 Best-of-N 采样
- 用启发式规则执行黑名单、任务载荷、AI 味等检查
- 用模拟的 Target Model 做增益过滤，丢弃“目标模型本来就会”的样本
- 输出批次级多样性报告，观察用户表达、风格和事件模式是否塌缩

当前的用户生成不再只依赖少量整句模板，而是结合：

- 标签匹配的软加权种子采样
- 批次内最近使用惩罚
- 风格化片段组合（开头 / 事件 / 情绪泄漏 / 收尾）

来减少批量生产时的重复感。

后续如果要接真实模型，只需要替换 `src/rp_factory/backends.py` 中的后端实现。
