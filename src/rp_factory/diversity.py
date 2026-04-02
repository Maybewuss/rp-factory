from __future__ import annotations

import re
from collections import Counter

from .models import DatasetRecord


_PUNCT_RE = re.compile(r"[，。！？；：、“”‘’（）《》【】\[\]\(\),.!?;:\n\r\t]+")


def _tokenize(text: str) -> list[str]:
    normalized = _PUNCT_RE.sub(" ", text)
    tokens = [token.strip() for token in normalized.split(" ") if token.strip()]
    if tokens:
        return tokens
    return [char for char in text if not char.isspace()]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    if len(tokens) < n:
        return []
    return [" / ".join(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1)]


def summarize_diversity(records: list[DatasetRecord]) -> dict[str, object]:
    user_bundles = [record._meta.get("user_bundle", {}) for record in records]
    messages = [bundle.get("user_message", "") for bundle in user_bundles]

    style_counter = Counter(bundle.get("style_tag", "unknown") for bundle in user_bundles)
    intent_counter = Counter(bundle.get("deep_intent", "unknown") for bundle in user_bundles)
    event_counter = Counter(bundle.get("proxy_event", "unknown") for bundle in user_bundles)

    bigrams = Counter()
    trigrams = Counter()
    for message in messages:
        tokens = _tokenize(message)
        bigrams.update(_ngrams(tokens, 2))
        trigrams.update(_ngrams(tokens, 3))

    return {
        "record_count": len(records),
        "unique_styles": len(style_counter),
        "style_distribution": dict(style_counter),
        "unique_intents": len(intent_counter),
        "unique_events": len(event_counter),
        "top_intents": intent_counter.most_common(5),
        "top_events": event_counter.most_common(5),
        "unique_bigrams": len(bigrams),
        "unique_trigrams": len(trigrams),
        "top_bigrams": bigrams.most_common(5),
        "top_trigrams": trigrams.most_common(5),
    }
