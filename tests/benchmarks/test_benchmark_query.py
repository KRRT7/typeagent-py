# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmarks for batch metadata query optimization.

After indexing 200 synthetic messages, exercises each function that was
converted from N+1 get_item() to batch get_metadata_multiple().

Run:
    uv run python -m pytest tests/benchmarks/test_benchmark_query.py -v -s
"""

import os
import shutil
import tempfile

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.answers import get_scored_semantic_refs_from_ordinals_iter
from typeagent.knowpro.collections import (
    SemanticRefAccumulator,
    TextRangeCollection,
    TextRangesInScope,
)
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces_core import Term, TextLocation, TextRange
from typeagent.knowpro.query import lookup_term_filtered
from typeagent.storage.memory.propindex import (
    PropertyNames,
    lookup_property_in_property_index,
)
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)


def make_settings() -> ConversationSettings:
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    return settings


def synthetic_messages(n: int) -> list[TranscriptMessage]:
    return [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(n)
    ]


async def create_indexed_transcript(
    db_path: str, settings: ConversationSettings, n_messages: int
) -> Transcript:
    """Create and index a transcript, returning it ready for queries."""
    storage = SqliteStorageProvider(
        db_path,
        message_type=TranscriptMessage,
        message_text_index_settings=settings.message_text_index_settings,
        related_term_index_settings=settings.related_term_index_settings,
    )
    settings.storage_provider = storage
    transcript = await Transcript.create(settings, name="bench")
    messages = synthetic_messages(n_messages)
    await transcript.add_messages_with_indexing(messages)
    return transcript


async def find_best_term(semref_index) -> tuple[str, int]:
    """Find the term with the most matches in the semantic ref index."""
    terms = await semref_index.get_terms()
    best_term = None
    best_count = 0
    for t in terms:
        refs = await semref_index.lookup_term(t)
        if refs and len(refs) > best_count:
            best_count = len(refs)
            best_term = t
    assert best_term is not None, "No terms found after indexing"
    return best_term, best_count


def make_scope_first_half(n_messages: int) -> TextRangesInScope:
    """Build a TextRangesInScope covering the first half of messages."""
    ranges = [
        TextRange(
            start=TextLocation(i, 0),
            end=TextLocation(i, 0),
        )
        for i in range(n_messages // 2)
    ]
    scope = TextRangesInScope()
    scope.add_text_ranges(TextRangeCollection(ranges))
    return scope


@pytest.mark.asyncio
async def test_benchmark_lookup_term_filtered(async_benchmark):
    """Benchmark lookup_term_filtered with batch get_metadata_multiple."""
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_ltf.db")

    transcript = await create_indexed_transcript(db_path, settings, 200)
    semref_index = transcript.semantic_ref_index
    best_term, best_count = await find_best_term(semref_index)
    print(f"\nBenchmarking term '{best_term}' with {best_count} matches")

    term = Term(text=best_term)
    semantic_refs = transcript.semantic_refs
    accept_all = lambda sr, scored: True

    async def target():
        await lookup_term_filtered(semref_index, term, semantic_refs, accept_all)

    try:
        await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
    finally:
        await settings.storage_provider.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_lookup_property_in_property_index(async_benchmark):
    """Benchmark property lookup with range filtering."""
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_prop.db")

    transcript = await create_indexed_transcript(db_path, settings, 200)
    assert transcript.secondary_indexes is not None
    property_index = transcript.secondary_indexes.property_to_semantic_ref_index
    assert property_index is not None, "Property index not built"

    # Verify there are matches for entity type "person"
    refs = await property_index.lookup_property(
        PropertyNames.EntityType.value, "person"
    )
    match_count = len(refs) if refs else 0
    print(f"\nBenchmarking property 'type=person' with {match_count} matches")
    assert match_count > 0

    scope = make_scope_first_half(200)

    async def target():
        await lookup_property_in_property_index(
            property_index,
            PropertyNames.EntityType.value,
            "person",
            transcript.semantic_refs,
            ranges_in_scope=scope,
        )

    try:
        await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
    finally:
        await settings.storage_provider.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_group_matches_by_type(async_benchmark):
    """Benchmark grouping accumulated matches by knowledge type."""
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_group.db")

    transcript = await create_indexed_transcript(db_path, settings, 200)
    semref_index = transcript.semantic_ref_index
    best_term, best_count = await find_best_term(semref_index)
    print(f"\nBenchmarking group_matches_by_type: term '{best_term}' ({best_count} matches)")

    scored_refs = await semref_index.lookup_term(best_term)
    accumulator = SemanticRefAccumulator()
    accumulator.add_term_matches(
        Term(text=best_term), scored_refs, is_exact_match=True
    )

    async def target():
        await accumulator.group_matches_by_type(transcript.semantic_refs)

    try:
        await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
    finally:
        await settings.storage_provider.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_get_matches_in_scope(async_benchmark):
    """Benchmark filtering accumulated matches by range scope."""
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_scope.db")

    transcript = await create_indexed_transcript(db_path, settings, 200)
    semref_index = transcript.semantic_ref_index
    best_term, best_count = await find_best_term(semref_index)
    print(f"\nBenchmarking get_matches_in_scope: term '{best_term}' ({best_count} matches)")

    scored_refs = await semref_index.lookup_term(best_term)
    accumulator = SemanticRefAccumulator()
    accumulator.add_term_matches(
        Term(text=best_term), scored_refs, is_exact_match=True
    )

    scope = make_scope_first_half(200)

    async def target():
        await accumulator.get_matches_in_scope(transcript.semantic_refs, scope)

    try:
        await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
    finally:
        await settings.storage_provider.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_benchmark_get_scored_semantic_refs_from_ordinals_iter(async_benchmark):
    """Benchmark two-phase metadata filter + batch fetch for scored refs."""
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_scored.db")

    transcript = await create_indexed_transcript(db_path, settings, 200)
    semref_index = transcript.semantic_ref_index
    best_term, best_count = await find_best_term(semref_index)
    print(
        f"\nBenchmarking get_scored_semantic_refs_from_ordinals_iter: "
        f"term '{best_term}' ({best_count} matches), filter=entity"
    )

    scored_refs = await semref_index.lookup_term(best_term)
    assert scored_refs is not None

    async def target():
        await get_scored_semantic_refs_from_ordinals_iter(
            transcript.semantic_refs, scored_refs, "entity"
        )

    try:
        await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
    finally:
        await settings.storage_provider.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
