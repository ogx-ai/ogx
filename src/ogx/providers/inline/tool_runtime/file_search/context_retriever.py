# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from jinja2.sandbox import SandboxedEnvironment

from ogx.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)
from ogx_api import (
    DefaultRAGQueryGeneratorConfig,
    InterleavedContent,
    LLMRAGQueryGeneratorConfig,
    OpenAIChatCompletionRequestWithExtraBody,
    OpenAIUserMessageParam,
    RAGQueryGenerator,
    RAGQueryGeneratorConfig,
)


async def generate_rag_query(
    config: RAGQueryGeneratorConfig,
    content: InterleavedContent,
    **kwargs,
):
    """
    Generates a query that will be used for
    retrieving relevant information from the memory bank.
    """
    if config.type == RAGQueryGenerator.default.value:
        query = await default_rag_query_generator(config, content, **kwargs)
    elif config.type == RAGQueryGenerator.llm.value:
        query = await llm_rag_query_generator(config, content, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported memory query generator {config.type}")
    return query


async def default_rag_query_generator(
    config: DefaultRAGQueryGeneratorConfig,
    content: InterleavedContent,
    **kwargs,
):
    """Generate a RAG query by converting content to a plain string.

    Args:
        config: query generator configuration with separator settings
        content: interleaved content to convert

    Returns:
        String representation of the content
    """
    return interleaved_content_as_str(content, sep=config.separator)


async def llm_rag_query_generator(
    config: LLMRAGQueryGeneratorConfig,
    content: InterleavedContent,
    **kwargs,
):
    """Generate a RAG query using an LLM to reformulate the content.

    Args:
        config: LLM query generator configuration with template and model
        content: interleaved content to reformulate

    Returns:
        LLM-generated query string for RAG retrieval
    """
    assert "inference_api" in kwargs, "LLMRAGQueryGenerator needs inference_api"
    inference_api = kwargs["inference_api"]

    messages = []
    if isinstance(content, list):
        messages = [interleaved_content_as_str(m) for m in content]
    else:
        messages = [interleaved_content_as_str(content)]

    sandbox_env = SandboxedEnvironment()
    template = sandbox_env.from_string(config.template)
    rendered_content: str = template.render({"messages": messages})

    model = config.model
    message = OpenAIUserMessageParam(content=rendered_content)
    params = OpenAIChatCompletionRequestWithExtraBody(
        model=model,
        messages=[message],
        stream=False,
    )
    response = await inference_api.openai_chat_completion(params)

    query = response.choices[0].message.content

    return query
