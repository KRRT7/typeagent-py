# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmarks for add_messages_with_indexing — the core indexing pipeline.

Exercises: message storage, semantic ref creation, term index insertion,
property index insertion, and embedding computation.

Only the hot path (add_messages_with_indexing) is timed — DB creation,
storage provider init, VTT parsing, and teardown are excluded via
async_benchmark.pedantic().

Run:
    uv run python -m pytest tests/benchmarks/test_benchmark_indexing.py -v -s
"""

import itertools
import os
import shutil
import tempfile

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.universal_message import ConversationMessage
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)
from typeagent.transcripts.transcript_ingest import ingest_vtt_transcript

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "testdata")
CONFUSE_A_CAT_VTT = os.path.join(TESTDATA, "Confuse-A-Cat.vtt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_settings() -> ConversationSettings:
    """Create conversation settings with fake embedding model (no API keys)."""
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    return settings


async def extract_vtt_messages(vtt_path: str) -> list[ConversationMessage]:
    """Parse a VTT file via ingest_vtt_transcript and return the messages."""
    settings = make_settings()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "parse.db")
        transcript = await ingest_vtt_transcript(vtt_path, settings, dbname=db_path)
        n = await transcript.messages.size()
        messages = await transcript.messages.get_slice(0, n)
        await settings.storage_provider.close()
    return messages


def synthetic_messages(n: int) -> list[TranscriptMessage]:
    """Build n synthetic TranscriptMessages."""
    return [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_benchmark_vtt_ingest(async_benchmark):
    """Benchmark indexing of pre-parsed VTT messages (Confuse-A-Cat, 40 msgs)."""
    messages = await extract_vtt_messages(CONFUSE_A_CAT_VTT)
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    counter = itertools.count()

    async def setup():
        i = next(counter)
        db_path = os.path.join(tmpdir, f"bench_{i}.db")
        storage = SqliteStorageProvider(
            db_path,
            message_type=ConversationMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="bench")
        return transcript, storage, db_path

    async def teardown(setup_rv):
        _, storage, db_path = setup_rv
        await storage.close()
        os.remove(db_path)

    async def target(transcript, storage, db_path):
        await transcript.add_messages_with_indexing(messages)

    try:
        await async_benchmark.pedantic(
            target, setup=setup, teardown=teardown, rounds=20, warmup_rounds=3
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_add_messages_50(async_benchmark):
    """Benchmark add_messages_with_indexing with 50 synthetic messages."""
    messages = synthetic_messages(50)
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    counter = itertools.count()

    async def setup():
        i = next(counter)
        db_path = os.path.join(tmpdir, f"bench_{i}.db")
        storage = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="bench")
        return transcript, storage, db_path

    async def teardown(setup_rv):
        _, storage, db_path = setup_rv
        await storage.close()
        os.remove(db_path)

    async def target(transcript, storage, db_path):
        await transcript.add_messages_with_indexing(messages)

    try:
        await async_benchmark.pedantic(
            target, setup=setup, teardown=teardown, rounds=20, warmup_rounds=3
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_add_messages_200(async_benchmark):
    """Benchmark add_messages_with_indexing with 200 synthetic messages."""
    messages = synthetic_messages(200)
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    counter = itertools.count()

    async def setup():
        i = next(counter)
        db_path = os.path.join(tmpdir, f"bench_{i}.db")
        storage = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="bench")
        return transcript, storage, db_path

    async def teardown(setup_rv):
        _, storage, db_path = setup_rv
        await storage.close()
        os.remove(db_path)

    async def target(transcript, storage, db_path):
        await transcript.add_messages_with_indexing(messages)

    try:
        await async_benchmark.pedantic(
            target, setup=setup, teardown=teardown, rounds=20, warmup_rounds=3
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
