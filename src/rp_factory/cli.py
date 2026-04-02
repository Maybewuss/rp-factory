from __future__ import annotations

import argparse
from pathlib import Path

import json

from .io_utils import write_jsonl
from .pipeline import build_dataset


def build_command(args: argparse.Namespace) -> int:
    result = build_dataset(
        input_path=Path(args.input),
        seed=args.seed,
        best_of_n=args.best_of_n,
        mix_ratio=args.mix_ratio,
    )
    write_jsonl(Path(args.output), result.records)
    if args.report:
        Path(args.report).write_text(
            json.dumps(result.batch_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"built {len(result.records)} records -> {args.output}")
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RP data factory MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build dataset from scenarios")
    build.add_argument("--input", required=True, help="scenario json path")
    build.add_argument("--output", required=True, help="jsonl output path")
    build.add_argument("--seed", type=int, default=7, help="random seed")
    build.add_argument("--best-of-n", type=int, default=4, help="pipeline B sample count")
    build.add_argument("--report", help="optional batch report json path")
    build.add_argument(
        "--mix-ratio",
        type=float,
        default=0.5,
        help="fraction of scenarios routed to pipeline A before alternating",
    )
    build.set_defaults(func=build_command)
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
