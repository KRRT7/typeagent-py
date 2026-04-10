# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmarks for add_messages_with_indexing — the core indexing pipeline.

Exercises: message storage, semantic ref creation, term index insertion,
property index insertion, and embedding computation.

Only the hot path (add_messages_with_indexing) is timed — DB creation,
storage provider init, VTT parsing, and teardown are excluded.

Run:
    uv run python -m pytest tests/benchmarks/test_benchmark_indexing.py -v -s
"""

import os
import shutil
import statistics
import tempfile
import time
from datetime import timedelta

import pytest
import webvtt

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.universal_message import (
    UNIX_EPOCH,
    ConversationMessage,
    ConversationMessageMeta,
    format_timestamp_utc,
)
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)
from typeagent.transcripts.transcript_ingest import (
    parse_voice_tags,
    webvtt_timestamp_to_seconds,
)

TESTDATA = os.path.join(os.path.dirname(__file__), "..", "testdata")
CONFUSE_A_CAT_VTT = os.path.join(TESTDATA, "Confuse-A-Cat.vtt")

WARMUP = 3
ROUNDS = 20


def _make_settings():
    """Create conversation settings with fake embedding model (no API keys)."""
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    return settings


def _parse_vtt(vtt_file_path: str) -> list[ConversationMessage]:
    """Parse a VTT file into messages (mirrors ingest_vtt_transcript parsing)."""
    vtt = webvtt.read(vtt_file_path)
    base_date = UNIX_EPOCH

    messages: list[ConversationMessage] = []
    current_speaker = None
    current_text_chunks: list[str] = []
    current_start_time = None

    for caption in vtt:
        if not caption.text.strip():
            continue
        raw_text = getattr(caption, "raw_text", caption.text)
        voice_segments = parse_voice_tags(raw_text)
        start_time = caption.start

        for speaker, text in voice_segments:
            if not text.strip():
                continue
            if speaker == current_speaker and current_text_chunks:
                current_text_chunks.append(text)
            else:
                if current_text_chunks and current_start_time is not None:
                    combined_text = " ".join(current_text_chunks).strip()
                    if combined_text:
                        offset_seconds = webvtt_timestamp_to_seconds(
                            current_start_time
                        )
                        timestamp = format_timestamp_utc(
                            base_date + timedelta(seconds=offset_seconds)
                        )
                        messages.append(
                            ConversationMessage(
                                text_chunks=[combined_text],
                                metadata=ConversationMessageMeta(
                                    speaker=current_speaker, recipients=[]
                                ),
                                timestamp=timestamp,
                            )
                        )
                current_speaker = speaker
                current_text_chunks = [text] if text.strip() else []
                current_start_time = start_time

    if current_text_chunks and current_start_time is not None:
        combined_text = " ".join(current_text_chunks).strip()
        if combined_text:
            offset_seconds = webvtt_timestamp_to_seconds(current_start_time)
            timestamp = format_timestamp_utc(
                base_date + timedelta(seconds=offset_seconds)
            )
            messages.append(
                ConversationMessage(
                    text_chunks=[combined_text],
                    metadata=ConversationMessageMeta(
                        speaker=current_speaker, recipients=[]
                    ),
                    timestamp=timestamp,
                )
            )

    return messages


def _report(name: str, times: list[float]) -> None:
    """Print benchmark stats to stdout."""
    mn = min(times)
    md = statistics.median(times)
    avg = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"\n{'=' * 60}")
    print(f"  {name}  ({ROUNDS} rounds, {WARMUP} warmup)")
    print(f"  min    = {mn * 1000:.3f} ms")
    print(f"  median = {md * 1000:.3f} ms")
    print(f"  mean   = {avg * 1000:.3f} ms")
    print(f"  stddev = {sd * 1000:.3f} ms")
    print(f"{'=' * 60}\n")


@pytest.mark.asyncio
async def test_benchmark_vtt_ingest():
    """Benchmark indexing of pre-parsed VTT messages."""
    vtt_messages = _parse_vtt(CONFUSE_A_CAT_VTT)
    settings = _make_settings()
    tmpdir = tempfile.mkdtemp()
    times: list[float] = []

    try:
        for i in range(WARMUP + ROUNDS):
            db_path = os.path.join(tmpdir, f"bench_{i}.db")
            storage = SqliteStorageProvider(
                db_path,
                message_type=ConversationMessage,
                message_text_index_settings=settings.message_text_index_settings,
                related_term_index_settings=settings.related_term_index_settings,
            )
            settings.storage_provider = storage
            transcript = await Transcript.create(
                settings,
                name="bench",
                tags=["bench", "vtt-transcript"],
            )

            start = time.perf_counter()
            await transcript.add_messages_with_indexing(vtt_messages)
            elapsed = time.perf_counter() - start

            await storage.close()
            os.remove(db_path)

            if i >= WARMUP:
                times.append(elapsed)

        _report(f"VTT ingest ({len(vtt_messages)} msgs)", times)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_add_messages_50():
    """Benchmark add_messages_with_indexing with 50 messages."""
    messages = [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(50)
    ]
    settings = _make_settings()
    tmpdir = tempfile.mkdtemp()
    times: list[float] = []

    try:
        for i in range(WARMUP + ROUNDS):
            db_path = os.path.join(tmpdir, f"bench_{i}.db")
            storage = SqliteStorageProvider(
                db_path,
                message_type=TranscriptMessage,
                message_text_index_settings=settings.message_text_index_settings,
                related_term_index_settings=settings.related_term_index_settings,
            )
            settings.storage_provider = storage
            transcript = await Transcript.create(settings, name="bench")

            start = time.perf_counter()
            await transcript.add_messages_with_indexing(messages)
            elapsed = time.perf_counter() - start

            await storage.close()
            os.remove(db_path)

            if i >= WARMUP:
                times.append(elapsed)

        _report("add_messages (50)", times)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_add_messages_200():
    """Benchmark add_messages_with_indexing with 200 messages."""
    messages = [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(200)
    ]
    settings = _make_settings()
    tmpdir = tempfile.mkdtemp()
    times: list[float] = []

    try:
        for i in range(WARMUP + ROUNDS):
            db_path = os.path.join(tmpdir, f"bench_{i}.db")
            storage = SqliteStorageProvider(
                db_path,
                message_type=TranscriptMessage,
                message_text_index_settings=settings.message_text_index_settings,
                related_term_index_settings=settings.related_term_index_settings,
            )
            settings.storage_provider = storage
            transcript = await Transcript.create(settings, name="bench")

            start = time.perf_counter()
            await transcript.add_messages_with_indexing(messages)
            elapsed = time.perf_counter() - start

            await storage.close()
            os.remove(db_path)

            if i >= WARMUP:
                times.append(elapsed)

        _report("add_messages (200)", times)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
