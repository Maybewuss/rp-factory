"""测试数据落盘序列化。"""

import json
import tempfile

import pytest

from rp_factory.config import FactoryConfig
from rp_factory.models import (
    ConversationMeta,
    ConversationRecord,
    Message,
    PipelineTag,
    QualityResult,
    QualityVerdict,
    Role,
)
from rp_factory.output.serializer import Serializer


@pytest.fixture
def sample_records():
    messages = [
        Message(role=Role.SYSTEM, content="你是一个高冷的外科医生。"),
        Message(role=Role.USER, content="我今天真的太倒霉了！"),
        Message(
            role=Role.ASSISTANT,
            content="倒霉？定义一下什么叫倒霉。",
            reasoning_content="用户情绪崩溃中，表面抱怨但可能有更深的困扰...",
        ),
    ]
    meta = ConversationMeta(
        system_prompt_source="你是一个高冷的外科医生。",
        pipeline=PipelineTag.PIPELINE_A,
        quality=QualityResult(verdict=QualityVerdict.PASS),
    )
    return [ConversationRecord(messages=messages, meta=meta)]


def test_write_jsonl(sample_records: list[ConversationRecord]):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = FactoryConfig()
        cfg.output.output_dir = tmpdir
        serializer = Serializer(cfg)

        path = serializer.write_jsonl(sample_records, "test.jsonl")
        assert path.exists()

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert "messages" in data
        assert len(data["messages"]) == 3
        assert data["messages"][2]["reasoning_content"] is not None
        assert "_meta" in data
        assert data["_meta"]["pipeline"] == "pipeline_a"


def test_write_training_only(sample_records: list[ConversationRecord]):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = FactoryConfig()
        cfg.output.output_dir = tmpdir
        serializer = Serializer(cfg)

        path = serializer.write_training_only(sample_records, "train.jsonl")
        assert path.exists()

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = json.loads(lines[0])
        assert "messages" in data
        assert "_meta" not in data


def test_write_empty_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = FactoryConfig()
        cfg.output.output_dir = tmpdir
        serializer = Serializer(cfg)

        path = serializer.write_jsonl([], "empty.jsonl")
        assert path.exists()

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        assert content == ""
