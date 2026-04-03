"""命令行入口 — 提供 rp-factory 的 CLI 接口。"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from rp_factory.config import FactoryConfig, load_config

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--config", "-c", default=None, help="配置文件路径 (默认 config/default.yaml)")
@click.option("--verbose", "-v", is_flag=True, help="启用详细日志")
@click.pass_context
def main(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """RP 数据合成工厂 v3.2 — 基于认知仿真与对偶动机的训练数据生成系统。"""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("system_prompt_file", type=click.Path(exists=True))
@click.option("--turns", "-t", default=None, type=int, help="对话轮数")
@click.option("--output", "-o", default="output.jsonl", help="输出文件路径")
@click.pass_context
def generate(
    ctx: click.Context,
    system_prompt_file: str,
    turns: int | None,
    output: str,
) -> None:
    """从单个角色 System Prompt 文件生成对话数据。"""
    cfg: FactoryConfig = ctx.obj["config"]

    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    console.print(f"[bold green]角色设定已加载[/] ({len(system_prompt)} 字符)")
    console.print(f"[dim]管线配比: A={cfg.rp_agent.mix_ratio.get('pipeline_a', 0.5)}"
                  f" / B={cfg.rp_agent.mix_ratio.get('pipeline_b', 0.5)}[/]")

    from rp_factory.pipeline import DataFactory

    factory = DataFactory(cfg)
    record = asyncio.run(factory.generate_conversation(system_prompt, turns))

    if record:
        from rp_factory.output.serializer import Serializer
        serializer = Serializer(cfg)
        path = serializer.write_jsonl([record], output)
        console.print(f"[bold green]✓ 数据已落盘:[/] {path}")
    else:
        console.print("[bold red]✗ 数据未通过质检，已废弃[/]")
        sys.exit(1)


@main.command()
@click.option("--prompts-dir", "-d", default=None, type=click.Path(exists=True),
              help="角色 System Prompt 目录（每个 .txt 文件一个角色）")
@click.option("--turns", "-t", default=None, type=int, help="对话轮数")
@click.option("--output", "-o", default="batch_output.jsonl", help="输出文件路径")
@click.option("--count", "-n", default=None, type=int,
              help="生成对话总数（超出角色数量时自动 LLM 生成新角色）")
@click.pass_context
def batch(
    ctx: click.Context,
    prompts_dir: str | None,
    turns: int | None,
    output: str,
    count: int | None,
) -> None:
    """批量生成数据。可提供角色目录，也可纯靠 LLM 自动生成角色。

    \b
    示例：
      rp-factory batch -d examples/ -n 50   # 从文件加载角色，不足 50 个时自动生成
      rp-factory batch -n 100               # 完全自动生成 100 个角色
    """
    cfg: FactoryConfig = ctx.obj["config"]

    system_prompts: list[str] = []
    if prompts_dir:
        prompt_files = sorted(Path(prompts_dir).glob("*.txt"))
        for pf in prompt_files:
            with open(pf, "r", encoding="utf-8") as f:
                system_prompts.append(f.read().strip())
        console.print(f"[bold green]已加载 {len(system_prompts)} 个角色设定[/]")

    target = count or len(system_prompts) or 10
    if not system_prompts:
        console.print(f"[bold yellow]未提供角色文件，将 LLM 自动生成 {target} 个角色[/]")
    elif target > len(system_prompts):
        console.print(
            f"[bold yellow]需要 {target} 条数据但只有 {len(system_prompts)} 个角色，"
            f"将自动生成 {target - len(system_prompts)} 个新角色[/]"
        )

    from rp_factory.pipeline import DataFactory

    factory = DataFactory(cfg)
    records = asyncio.run(factory.run_batch(
        system_prompts=system_prompts or None,
        count=target,
        num_turns=turns,
        output_file=output,
    ))

    table = Table(title="批量生成结果")
    table.add_column("指标", style="bold")
    table.add_column("值", justify="right")
    table.add_row("目标数量", str(target))
    table.add_row("通过质检", str(len(records)))
    table.add_row("废弃率", f"{(1 - len(records) / max(target, 1)) * 100:.1f}%")
    table.add_row("种子池 (intents)", str(len(factory.iceberg.intents)))
    table.add_row("种子池 (events)", str(len(factory.iceberg.events)))
    table.add_row("种子池 (styles)", str(len(factory.iceberg.styles)))

    console.print(table)


@main.command()
@click.option("--count", "-n", default=5, type=int, help="生成角色数量")
@click.option("--output", "-o", default=None, help="输出目录（每个角色一个 .txt 文件）")
@click.pass_context
def gen_personas(ctx: click.Context, count: int, output: str | None) -> None:
    """LLM 自动生成多样化角色 System Prompt。"""
    cfg: FactoryConfig = ctx.obj["config"]

    from rp_factory.pipeline import DataFactory

    factory = DataFactory(cfg)
    personas = asyncio.run(factory.generate_personas(count))

    console.print(f"[bold green]成功生成 {len(personas)} 个角色[/]")

    if output:
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(personas):
            path = out_dir / f"persona_{i:03d}.txt"
            with open(path, "w", encoding="utf-8") as f:
                f.write(p)
            console.print(f"  [dim]{path}[/]")
    else:
        for i, p in enumerate(personas):
            console.print(f"\n[bold cyan]━━━ 角色 {i + 1} ━━━[/]")
            console.print(p[:300] + ("..." if len(p) > 300 else ""))


@main.command()
@click.argument("jsonl_file", type=click.Path(exists=True))
@click.option("--sample", "-s", default=5, type=int, help="抽样数量")
@click.pass_context
def inspect(ctx: click.Context, jsonl_file: str, sample: int) -> None:
    """检视已生成的 JSONL 数据文件。"""
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    console.print(f"[bold]共 {len(records)} 条记录[/]")

    import random
    sampled = random.sample(records, min(sample, len(records)))

    for i, rec in enumerate(sampled):
        console.print(f"\n[bold cyan]━━━ 样本 {i + 1} ━━━[/]")
        messages = rec.get("messages", [])
        for msg in messages:
            role = msg["role"]
            content = msg["content"][:200]
            if role == "system":
                console.print(f"  [bold magenta][SYSTEM][/] {content}...")
            elif role == "user":
                console.print(f"  [bold blue][USER][/] {content}")
            else:
                reasoning = msg.get("reasoning_content", "")
                if reasoning:
                    console.print(f"  [dim][REASONING][/] {reasoning[:150]}...")
                console.print(f"  [bold green][ASSISTANT][/] {content}")

        meta = rec.get("_meta", {})
        if meta:
            console.print(f"  [dim]pipeline={meta.get('pipeline')} | "
                          f"quality={meta.get('quality', {}).get('verdict')}[/]")


@main.command()
@click.option("--category", "-c", multiple=True, default=None,
              help="要扩充的类别（可多选）：intents / events / styles / tasks。不指定则全部扩充。")
@click.option("--count", "-n", default=10, type=int, help="每个类别生成多少条")
@click.pass_context
def expand_seeds(ctx: click.Context, category: tuple[str, ...], count: int) -> None:
    """调用 LLM 扩充种子库并写回 YAML 文件持久化。

    \b
    示例：
      rp-factory expand-seeds -n 20                    # 全部类别各扩充 20 条
      rp-factory expand-seeds -c intents -c events -n 15  # 只扩充动机和事件
    """
    cfg: FactoryConfig = ctx.obj["config"]

    all_categories = ["intents", "events", "styles", "tasks"]
    targets = list(category) if category else all_categories

    for c in targets:
        if c not in all_categories:
            console.print(f"[bold red]未知类别: {c}[/]（可选：{', '.join(all_categories)}）")
            sys.exit(1)

    from rp_factory.generators import SeedExpander, expand_seeds_to_file
    from rp_factory.llm_client import LLMClient

    expander = SeedExpander(LLMClient(cfg.llm.generator))

    async def _run() -> None:
        table = Table(title="种子扩充结果")
        table.add_column("类别", style="bold")
        table.add_column("新增", justify="right")
        table.add_column("总数", justify="right")

        for c in targets:
            console.print(f"[dim]扩充 {c} (×{count})...[/]")
            added, total = await expand_seeds_to_file(expander, c, count)
            table.add_row(c, str(added), str(total))

        console.print(table)
        console.print("[bold green]种子已写回 seeds/ 目录[/]")

    asyncio.run(_run())


@main.command()
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """显示当前生效的配置。"""
    cfg: FactoryConfig = ctx.obj["config"]
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
