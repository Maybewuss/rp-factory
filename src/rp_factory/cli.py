from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import write_jsonl
from .models import GenerationPolicy, GenerationTargets
from .pipeline import build_dataset


def build_command(args: argparse.Namespace) -> int:
    result = build_dataset(
        input_path=Path(args.input),
        seed=args.seed,
        best_of_n=args.best_of_n,
        mix_ratio=args.mix_ratio,
        seed_pool_path=Path(args.seed_pool),
        allow_pool_expansion=not args.disable_pool_expansion,
        expansion_backend=args.expansion_backend,
        diversity_targets=GenerationTargets(
            min_unique_styles=args.min_unique_styles,
            min_unique_intents=args.min_unique_intents,
            min_unique_events=args.min_unique_events,
            min_unique_persona_overlays=args.min_unique_persona_overlays,
        ),
        generation_policy=GenerationPolicy(
            expansion_batch_size=args.expansion_batch_size,
            max_generation_attempts=args.max_generation_attempts,
        ),
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
        "--seed-pool",
        default="src/rp_factory/seed_pool.json",
        help="seed pool json path",
    )
    build.add_argument(
        "--expansion-backend",
        choices=("mock", "llm"),
        default="mock",
        help="dynamic pool expansion backend",
    )
    build.add_argument(
        "--expansion-batch-size",
        type=int,
        default=3,
        help="number of new items generated per pool expansion",
    )
    build.add_argument("--min-unique-styles", type=int, default=3, help="minimum style diversity target")
    build.add_argument("--min-unique-intents", type=int, default=3, help="minimum intent diversity target")
    build.add_argument("--min-unique-events", type=int, default=3, help="minimum event diversity target")
    build.add_argument(
        "--min-unique-persona-overlays",
        type=int,
        default=3,
        help="minimum persona overlay diversity target",
    )
    build.add_argument(
        "--max-generation-attempts",
        type=int,
        default=4,
        help="max retries for satisfying diversity constraints per scenario",
    )
    build.add_argument(
        "--disable-pool-expansion",
        action="store_true",
        help="disable dynamic pool expansion and only use existing pool",
    )
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
