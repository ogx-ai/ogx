# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import BadRequestError as OGXBadRequestError
from llama_stack_client.types.alpha import InferenceRerankResponse
from llama_stack_client.types.shared.interleaved_content import (
    ImageContentItem,
    ImageContentItemImage,
    ImageContentItemImageURL,
    TextContentItem,
)

from ogx.core.library_client import OGXAsLibraryClient

# Test data
DUMMY_STRING = "string_1"
DUMMY_STRING2 = "string_2"
DUMMY_TEXT = TextContentItem(text=DUMMY_STRING, type="text")
DUMMY_TEXT2 = TextContentItem(text=DUMMY_STRING2, type="text")
DUMMY_IMAGE_URL = ImageContentItem(
    image=ImageContentItemImage(url=ImageContentItemImageURL(uri="https://example.com/image.jpg")), type="image"
)
DUMMY_IMAGE_BASE64 = ImageContentItem(image=ImageContentItemImage(data="base64string"), type="image")

PROVIDERS_SUPPORTING_MEDIA = {}  # Providers that support media input for rerank models


def skip_if_provider_doesnt_support_rerank(inference_provider_type):
    supported_providers = {"remote::nvidia", "remote::vllm", "inline::transformers"}
    if inference_provider_type not in supported_providers:
        pytest.skip(f"{inference_provider_type} doesn't support rerank models")


def _validate_rerank_response(response: InferenceRerankResponse, items: list) -> None:
    """
    Validate that a rerank response has the correct structure and ordering.

    Args:
        response: The InferenceRerankResponse to validate
        items: The original items list that was ranked

    Raises:
        AssertionError: If any validation fails
    """
    seen = set()
    last_score = float("inf")
    for d in response:
        assert 0 <= d.index < len(items), f"Index {d.index} out of bounds for {len(items)} items"
        assert d.index not in seen, f"Duplicate index {d.index} found"
        seen.add(d.index)
        assert isinstance(d.relevance_score, float), f"Score must be float, got {type(d.relevance_score)}"
        assert d.relevance_score <= last_score, f"Scores not in descending order: {d.relevance_score} > {last_score}"
        last_score = d.relevance_score


def _validate_semantic_ranking(response: InferenceRerankResponse, items: list, expected_first_item: str) -> None:
    """
    Validate that the expected most relevant item ranks first.

    Args:
        response: The InferenceRerankResponse to validate
        items: The original items list that was ranked
        expected_first_item: The expected first item in the ranking

    Raises:
        AssertionError: If any validation fails
    """
    if not response:
        raise AssertionError("No ranking data returned in response")

    actual_first_index = response[0].index
    actual_first_item = items[actual_first_index]
    assert actual_first_item == expected_first_item, (
        f"Expected '{expected_first_item}' to rank first, but '{actual_first_item}' ranked first instead."
    )


@pytest.mark.parametrize(
    "query,items",
    [
        (DUMMY_STRING, [DUMMY_STRING, DUMMY_STRING2]),
        (DUMMY_TEXT, [DUMMY_TEXT, DUMMY_TEXT2]),
        (DUMMY_STRING, [DUMMY_STRING2, DUMMY_TEXT]),
        (DUMMY_TEXT, [DUMMY_STRING, DUMMY_TEXT2]),
    ],
    ids=[
        "string-query-string-items",
        "text-query-text-items",
        "mixed-content-1",
        "mixed-content-2",
    ],
)
def test_rerank_text(client_with_models, rerank_model_id, query, items, inference_provider_type):
    skip_if_provider_doesnt_support_rerank(inference_provider_type)

    response = client_with_models.alpha.inference.rerank(model=rerank_model_id, query=query, items=items)
    assert isinstance(response, list)
    # TODO: Add type validation for response items once InferenceRerankResponseItem is exported from ogx client.
    assert len(response) <= len(items)
    _validate_rerank_response(response, items)


@pytest.mark.parametrize(
    "query,items",
    [
        (DUMMY_IMAGE_URL, [DUMMY_STRING]),
        (DUMMY_IMAGE_BASE64, [DUMMY_TEXT]),
        (DUMMY_TEXT, [DUMMY_IMAGE_URL]),
        (DUMMY_IMAGE_BASE64, [DUMMY_IMAGE_URL, DUMMY_STRING, DUMMY_IMAGE_BASE64, DUMMY_TEXT]),
        (DUMMY_TEXT, [DUMMY_IMAGE_URL, DUMMY_STRING, DUMMY_IMAGE_BASE64, DUMMY_TEXT]),
    ],
    ids=[
        "image-query-url",
        "image-query-base64",
        "text-query-image-item",
        "mixed-content-1",
        "mixed-content-2",
    ],
)
def test_rerank_image(client_with_models, rerank_model_id, query, items, inference_provider_type):
    skip_if_provider_doesnt_support_rerank(inference_provider_type)

    if rerank_model_id not in PROVIDERS_SUPPORTING_MEDIA:
        error_type = ValueError if isinstance(client_with_models, OGXAsLibraryClient) else OGXBadRequestError
        with pytest.raises(error_type):
            client_with_models.alpha.inference.rerank(model=rerank_model_id, query=query, items=items)
    else:
        response = client_with_models.alpha.inference.rerank(model=rerank_model_id, query=query, items=items)

        assert isinstance(response, list)
        assert len(response) <= len(items)
        _validate_rerank_response(response, items)


def test_rerank_max_results(client_with_models, rerank_model_id, inference_provider_type):
    skip_if_provider_doesnt_support_rerank(inference_provider_type)

    items = [DUMMY_STRING, DUMMY_STRING2, DUMMY_TEXT, DUMMY_TEXT2]
    max_num_results = 2

    response = client_with_models.alpha.inference.rerank(
        model=rerank_model_id,
        query=DUMMY_STRING,
        items=items,
        max_num_results=max_num_results,
    )

    assert isinstance(response, list)
    assert len(response) == max_num_results
    _validate_rerank_response(response, items)


def test_rerank_max_results_larger_than_items(client_with_models, rerank_model_id, inference_provider_type):
    skip_if_provider_doesnt_support_rerank(inference_provider_type)

    items = [DUMMY_STRING, DUMMY_STRING2]
    response = client_with_models.alpha.inference.rerank(
        model=rerank_model_id,
        query=DUMMY_STRING,
        items=items,
        max_num_results=10,  # Larger than items length
    )

    assert isinstance(response, list)
    assert len(response) <= len(items)  # Should return at most len(items)


@pytest.mark.parametrize(
    "query,items,expected_first_item",
    [
        (
            "What is a reranking model? ",
            [
                "A reranking model reranks a list of items based on the query. ",
                "Machine learning algorithms learn patterns from data. ",
                "Python is a programming language. ",
            ],
            "A reranking model reranks a list of items based on the query. ",
        ),
        (
            "What is C++?",
            [
                "Learning new things is interesting. ",
                "C++ is a programming language. ",
                "Books provide knowledge and entertainment. ",
            ],
            "C++ is a programming language. ",
        ),
        (
            "What are good learning habits? ",
            [
                "Cooking pasta is a fun activity. ",
                "Plants need water and sunlight. ",
                "Good learning habits include reading daily and taking notes. ",
            ],
            "Good learning habits include reading daily and taking notes. ",
        ),
    ],
)
def test_rerank_semantic_correctness(
    client_with_models, rerank_model_id, query, items, expected_first_item, inference_provider_type
):
    skip_if_provider_doesnt_support_rerank(inference_provider_type)

    response = client_with_models.alpha.inference.rerank(model=rerank_model_id, query=query, items=items)

    _validate_rerank_response(response, items)
    _validate_semantic_ranking(response, items, expected_first_item)
