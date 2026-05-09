# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest

from ogx.core.utils.context import preserve_contexts_async_generator


async def test_preserve_contexts_with_exception():
    # Create context variable
    context_var = ContextVar("exception_var", default="initial")
    token = context_var.set("start_value")

    # Create an async generator that raises an exception
    async def exception_generator():
        yield context_var.get()
        context_var.set("modified")
        raise ValueError("Test exception")
        yield None  # This will never be reached

    # Wrap the generator
    wrapped_gen = preserve_contexts_async_generator(exception_generator(), [context_var])

    # First iteration should work
    value = await wrapped_gen.__anext__()
    assert value == "start_value"

    # Second iteration should raise the exception
    with pytest.raises(ValueError, match="Test exception"):
        await wrapped_gen.__anext__()

    # Clean up
    context_var.reset(token)


async def test_preserve_contexts_empty_generator():
    # Create context variable
    context_var = ContextVar("empty_var", default="initial")
    token = context_var.set("value")

    # Create an empty async generator
    async def empty_generator():
        if False:  # This condition ensures the generator yields nothing
            yield None

    # Wrap the generator
    wrapped_gen = preserve_contexts_async_generator(empty_generator(), [context_var])

    # The generator should raise StopAsyncIteration immediately
    with pytest.raises(StopAsyncIteration):
        await wrapped_gen.__anext__()

    # Context variable should remain unchanged
    assert context_var.get() == "value"

    # Clean up
    context_var.reset(token)


async def test_preserve_contexts_across_event_loops():
    """
    Test that context variables are preserved across event loop boundaries with nested generators.
    This simulates the real-world scenario where:
    1. A new event loop is created for each streaming request
    2. The async generator runs inside that loop
    3. There are multiple levels of nested generators
    4. Context needs to be preserved across these boundaries
    """
    # Create context variables
    request_id = ContextVar("request_id", default=None)
    user_id = ContextVar("user_id", default=None)

    # Set initial values

    # Results container to verify values across thread boundaries
    results = []

    # Inner-most generator (level 2)
    async def inner_generator():
        # Should have the context from the outer scope
        yield (1, request_id.get(), user_id.get())

        # Modify one context variable
        user_id.set("user-modified")

        # Should reflect the modification
        yield (2, request_id.get(), user_id.get())

    # Middle generator (level 1)
    async def middle_generator():
        inner_gen = inner_generator()

        # Forward the first yield from inner
        item = await inner_gen.__anext__()
        yield item

        # Forward the second yield from inner
        item = await inner_gen.__anext__()
        yield item

        request_id.set("req-modified")

        # Add our own yield with both modified variables
        yield (3, request_id.get(), user_id.get())

    # Function to run in a separate thread with a new event loop
    def run_in_new_loop():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Outer generator (runs in the new loop)
            async def outer_generator():
                request_id.set("req-12345")
                user_id.set("user-6789")
                # Wrap the middle generator
                wrapped_gen = preserve_contexts_async_generator(middle_generator(), [request_id, user_id])

                # Process all items from the middle generator
                async for item in wrapped_gen:
                    # Store results for verification
                    results.append(item)

            # Run the outer generator in the new loop
            loop.run_until_complete(outer_generator())
        finally:
            loop.close()

    # Run the generator chain in a separate thread with a new event loop
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_new_loop)
        future.result()  # Wait for completion

    # Verify the results
    assert len(results) == 3

    # First yield should have original values
    assert results[0] == (1, "req-12345", "user-6789")

    # Second yield should have modified user_id
    assert results[1] == (2, "req-12345", "user-modified")

    # Third yield should have both modified values
    assert results[2] == (3, "req-modified", "user-modified")
