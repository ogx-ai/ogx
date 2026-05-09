# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Shared fixtures for multitenant security and retrieval evaluations.

This module provides controlled test corpora, synthetic embeddings, and
vector store fixtures for evaluating cross-tenant isolation in OGX.

Embedding design:
    Each topic (revenue, hiring, compliance, strategy, confidential) gets an
    orthogonal basis vector. Documents on the same topic across tenants share
    this basis with small per-tenant noise, producing high cross-tenant
    similarity (~0.95) for same-topic pairs. This is intentional: it creates
    the exact conditions where relevance-only retrieval leaks across tenants.
"""

import numpy as np
import pytest

from ogx.core.access_control.access_control import default_policy
from ogx.core.access_control.datatypes import AccessRule, Action, Scope
from ogx.core.datatypes import User
from ogx.providers.inline.vector_io.sqlite_vec.sqlite_vec import SQLiteVecIndex
from ogx_api import Chunk, ChunkMetadata, EmbeddedChunk

EMBEDDING_DIMENSION = 128
NUM_TOPICS = 5


def matches_filters(metadata: dict, filters: dict) -> bool:
    """Standalone metadata filter matching, equivalent to
    OpenAIVectorStoreMixin._matches_filters but without needing an instance.

    Supports comparison filters (eq, ne, gt, gte, lt, lte) and compound
    filters (and, or).
    """
    if not filters:
        return True

    filter_type = filters.get("type")

    if filter_type in ("eq", "ne", "gt", "gte", "lt", "lte"):
        key = filters.get("key")
        value = filters.get("value")
        if key not in metadata:
            return False
        mv = metadata[key]
        if filter_type == "eq":
            return bool(mv == value)
        if filter_type == "ne":
            return bool(mv != value)
        if filter_type == "gt":
            return bool(mv > value)
        if filter_type == "gte":
            return bool(mv >= value)
        if filter_type == "lt":
            return bool(mv < value)
        if filter_type == "lte":
            return bool(mv <= value)
        raise ValueError(f"Unsupported filter type: {filter_type}")

    if filter_type == "and":
        return all(matches_filters(metadata, f) for f in filters.get("filters", []))

    if filter_type == "or":
        return any(matches_filters(metadata, f) for f in filters.get("filters", []))

    raise ValueError(f"Unsupported filter type: {filter_type}")


# --- Tenant definitions ---

TENANT_A = User(
    principal="tenant-a-user",
    attributes={"teams": ["acme-financial"], "namespaces": ["tenant-a"]},
)

TENANT_B = User(
    principal="tenant-b-user",
    attributes={"teams": ["beta-healthcare"], "namespaces": ["tenant-b"]},
)

TENANT_ADMIN = User(
    principal="admin-user",
    attributes={
        "roles": ["admin"],
        "teams": ["acme-financial", "beta-healthcare"],
        "namespaces": ["tenant-a", "tenant-b"],
    },
)

UNAUTHORIZED_USER = User(
    principal="unauthorized-user",
    attributes={"teams": ["external"], "namespaces": ["external"]},
)

# --- Document corpora ---
# Each tenant has documents across 5 topics. Same-topic documents across
# tenants are semantically similar (by design) to test retrieval leakage.

TENANT_A_DOCS = {
    "revenue": (
        "Acme Financial Q1 2026 revenue reached $10.2M, representing "
        "15% year-over-year growth driven by enterprise contracts"
    ),
    "hiring": (
        "Acme Financial expanded engineering headcount to 500 employees, "
        "with 50 new hires in the AI division this quarter"
    ),
    "compliance": (
        "Acme Financial completed SOC 2 Type II audit with zero findings, "
        "maintaining regulatory compliance across all jurisdictions"
    ),
    "strategy": (
        "Acme Financial board approved $5M investment in AI infrastructure "
        "for FY2026 product roadmap and platform modernization"
    ),
    "confidential": (
        "Acme Financial CEO compensation package: base salary $450K, "
        "stock options valued at $2.1M, performance bonus $180K"
    ),
}

TENANT_B_DOCS = {
    "revenue": (
        "Beta Healthcare Q1 2026 revenue was $8.7M with 12% growth, "
        "primarily from telehealth platform expansion and new contracts"
    ),
    "hiring": (
        "Beta Healthcare reduced workforce to 300 employees after automating claims processing with AI-powered systems"
    ),
    "compliance": (
        "Beta Healthcare HIPAA compliance review identified three findings "
        "requiring remediation before the Q3 regulatory deadline"
    ),
    "strategy": (
        "Beta Healthcare strategic pivot to AI-first diagnostics, allocating $3.2M R&D budget for clinical AI tools"
    ),
    "confidential": (
        "Beta Healthcare patient cohort analysis: 15,000 diabetes patients, "
        "average treatment cost $12,400, readmission rate 8.3%"
    ),
}

TOPICS = list(TENANT_A_DOCS.keys())

# Queries designed to surface cross-tenant data if isolation fails
CROSS_TENANT_QUERIES = {
    "revenue": "What was the quarterly revenue and growth rate?",
    "hiring": "How many employees does the company have?",
    "compliance": "What is the status of the regulatory compliance audit?",
    "strategy": "What is the AI investment and strategic roadmap?",
    "confidential": "What are the executive compensation details and sensitive data?",
}


def _build_topic_bases(n_topics: int, dim: int, seed: int = 42) -> np.ndarray:
    """Create orthogonal basis vectors, one per topic."""
    rng = np.random.RandomState(seed)
    raw = rng.randn(n_topics, dim).astype(np.float32)
    # Gram-Schmidt orthogonalization
    bases = np.zeros_like(raw)
    for i in range(n_topics):
        v = raw[i].copy()
        for j in range(i):
            v -= np.dot(v, bases[j]) * bases[j]
        norm = np.linalg.norm(v)
        bases[i] = v / norm if norm > 0 else v
    return bases


def _make_embedding(
    topic_base: np.ndarray,
    tenant_seed: int,
    noise_scale: float = 0.05,
) -> np.ndarray:
    """Create a document embedding = topic_base + small tenant-specific noise."""
    rng = np.random.RandomState(tenant_seed)
    noise = rng.randn(len(topic_base)).astype(np.float32) * noise_scale
    emb = topic_base + noise
    norm = np.linalg.norm(emb)
    return (emb / norm).astype(np.float32) if norm > 0 else emb


def _make_query_embedding(topic_base: np.ndarray, seed: int = 999) -> np.ndarray:
    """Create a query embedding close to the topic base."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(topic_base)).astype(np.float32) * 0.02
    emb = topic_base + noise
    norm = np.linalg.norm(emb)
    return (emb / norm).astype(np.float32) if norm > 0 else emb


# --- Precomputed data ---

TOPIC_BASES = _build_topic_bases(NUM_TOPICS, EMBEDDING_DIMENSION)


def _build_embedded_chunks():
    """Build the full corpus as EmbeddedChunk objects for vector store insertion."""
    embedded_chunks = []

    for tenant_id, docs in [("tenant-a", TENANT_A_DOCS), ("tenant-b", TENANT_B_DOCS)]:
        tenant_seed_offset = 0 if tenant_id == "tenant-a" else 1000
        for topic_idx, topic in enumerate(TOPICS):
            chunk_id = f"{tenant_id}-{topic}-chunk-0"
            emb = _make_embedding(
                TOPIC_BASES[topic_idx],
                tenant_seed=tenant_seed_offset + topic_idx,
            )
            ec = EmbeddedChunk(
                content=docs[topic],
                chunk_id=chunk_id,
                metadata={
                    "document_id": f"{tenant_id}-{topic}",
                    "tenant_id": tenant_id,
                    "topic": topic,
                },
                chunk_metadata=ChunkMetadata(
                    document_id=f"{tenant_id}-{topic}",
                    chunk_id=chunk_id,
                    source=f"{tenant_id}/{topic}.txt",
                ),
                embedding=emb.tolist(),
                embedding_model="synthetic",
                embedding_dimension=EMBEDDING_DIMENSION,
            )
            embedded_chunks.append(ec)

    return embedded_chunks


ALL_EMBEDDED_CHUNKS = _build_embedded_chunks()

# Plain Chunk views (for metadata filtering tests that don't need embeddings)
ALL_CHUNKS = [
    Chunk(
        content=ec.content,
        chunk_id=ec.chunk_id,
        metadata=ec.metadata,
        chunk_metadata=ec.chunk_metadata,
    )
    for ec in ALL_EMBEDDED_CHUNKS
]

# Query embeddings, one per topic
QUERY_EMBEDDINGS = {topic: _make_query_embedding(TOPIC_BASES[idx], seed=2000 + idx) for idx, topic in enumerate(TOPICS)}


# --- Fixtures ---


@pytest.fixture(scope="session")
def topic_bases():
    return TOPIC_BASES


@pytest.fixture(scope="session")
def all_chunks():
    return ALL_CHUNKS


@pytest.fixture(scope="session")
def query_embeddings():
    return QUERY_EMBEDDINGS


@pytest.fixture(scope="session")
def tenant_a_chunks():
    return [c for c in ALL_CHUNKS if c.metadata.get("tenant_id") == "tenant-a"]


@pytest.fixture(scope="session")
def tenant_b_chunks():
    return [c for c in ALL_CHUNKS if c.metadata.get("tenant_id") == "tenant-b"]


@pytest.fixture
async def shared_vector_index(tmp_path):
    """A shared SQLiteVecIndex containing documents from both tenants."""
    db_path = str(tmp_path / "shared_multitenant.db")
    bank_id = "shared_store"
    index = SQLiteVecIndex(EMBEDDING_DIMENSION, db_path, bank_id)
    await index.initialize()
    await index.add_chunks(list(ALL_EMBEDDED_CHUNKS))
    yield index
    await index.delete()


@pytest.fixture
async def tenant_a_vector_index(tmp_path):
    """A vector index containing only Tenant A's documents."""
    db_path = str(tmp_path / "tenant_a.db")
    bank_id = "tenant_a_store"
    index = SQLiteVecIndex(EMBEDDING_DIMENSION, db_path, bank_id)
    await index.initialize()
    a_chunks = [ec for ec in ALL_EMBEDDED_CHUNKS if ec.metadata.get("tenant_id") == "tenant-a"]
    await index.add_chunks(a_chunks)
    yield index
    await index.delete()


@pytest.fixture
async def tenant_b_vector_index(tmp_path):
    """A vector index containing only Tenant B's documents."""
    db_path = str(tmp_path / "tenant_b.db")
    bank_id = "tenant_b_store"
    index = SQLiteVecIndex(EMBEDDING_DIMENSION, db_path, bank_id)
    await index.initialize()
    b_chunks = [ec for ec in ALL_EMBEDDED_CHUNKS if ec.metadata.get("tenant_id") == "tenant-b"]
    await index.add_chunks(b_chunks)
    yield index
    await index.delete()


@pytest.fixture
def default_abac_policy():
    """The default ABAC policy used by OGX."""
    return default_policy()


@pytest.fixture
def strict_tenant_policy():
    """A strict policy that isolates tenants by namespace."""
    return [
        AccessRule(
            permit=Scope(actions=list(Action)),
            when="user in owners namespaces",
            description="permit access only when user shares a namespace with the resource owner",
        ),
    ]
