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
    """RP 数据合成工厂 v3.1 — 基于认知仿真与对偶动机的训练数据生成系统。"""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("system_prompt_file", type=click.Path(exists=True))
@click.option("--turns", "-t", default=None, type=int, help="对话轮数")
@click.option("--output", "-o", default="output.jsonl", help="输出文件名")
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
@click.argument("prompts_dir", type=click.Path(exists=True))
@click.option("--turns", "-t", default=None, type=int, help="对话轮数")
@click.option("--output", "-o", default="batch_output.jsonl", help="输出文件名")
@click.option("--limit", "-n", default=None, type=int, help="最大处理数量")
@click.pass_context
def batch(
    ctx: click.Context,
    prompts_dir: str,
    turns: int | None,
    output: str,
    limit: int | None,
) -> None:
    """从目录中的多个 System Prompt 文件批量生成数据。"""
    cfg: FactoryConfig = ctx.obj["config"]

    prompt_files = sorted(Path(prompts_dir).glob("*.txt"))
    if limit:
        prompt_files = prompt_files[:limit]

    if not prompt_files:
        console.print("[bold red]目录中未找到 .txt 文件[/]")
        sys.exit(1)

    system_prompts = []
    for pf in prompt_files:
        with open(pf, "r", encoding="utf-8") as f:
            system_prompts.append(f.read().strip())

    console.print(f"[bold green]已加载 {len(system_prompts)} 个角色设定[/]")

    from rp_factory.pipeline import DataFactory

    factory = DataFactory(cfg)
    records = asyncio.run(factory.run_batch(system_prompts, turns, output))

    table = Table(title="批量生成结果")
    table.add_column("指标", style="bold")
    table.add_column("值", justify="right")
    table.add_row("总角色数", str(len(system_prompts)))
    table.add_row("通过质检", str(len(records)))
    table.add_row("废弃率", f"{(1 - len(records) / max(len(system_prompts), 1)) * 100:.1f}%")

    console.print(table)


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
@click.pass_context
def show_config(ctx: click.Context) -> None:
    """显示当前生效的配置。"""
    cfg: FactoryConfig = ctx.obj["config"]
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
