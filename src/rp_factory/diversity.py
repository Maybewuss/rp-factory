from __future__ import annotations

from collections import Counter

from .models import DatasetRecord


def _ngrams(text: str, n: int) -> list[str]:
    tokens = text.split()
    if len(tokens) < n:
        return []
    return [" ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def summarize_diversity(records: list[DatasetRecord]) -> dict[str, object]:
    messages = [record.messages[1]["content"] for record in records if len(record.messages) > 1]
    style_counter = Counter(
        record._meta.get("user_bundle", {}).get("style_tag", "unknown")
        for record in records
    )
    bigrams = Counter()
    trigrams = Counter()
    for message in messages:
        bigrams.update(_ngrams(message, 2))
        trigrams.update(_ngrams(message, 3))
    return {
        "record_count": len(records),
        "unique_styles": len(style_counter),
        "style_distribution": dict(style_counter),
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "top_bigrams": bigrams.most_common(5),
        "top_trigrams": trigrams.most_common(5),
    }
