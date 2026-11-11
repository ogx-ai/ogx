# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Regression test for https://github.com/ogx-ai/ogx/issues/6969.
#
# Each MCP session holds anyio cancel scopes that are entered in the calling
# task: the transport client's task group and the ClientSession's task group.
# anyio only lets a task leave a cancel scope when it is that task's current
# (innermost) scope, so a task must fully leave one session's scopes before it
# leaves an older session's. Closing the oldest session first fails with
# "Attempted to exit a cancel scope that isn't the current task's current
# cancel scope"; MCPSessionManager.__aexit__ swallows that error by design, so
# the task is left with scopes it never leaves and breaks later at its next
# scope exit.

from typing import Any

import anyio

from ogx.providers.utils.tools.mcp import MCPSessionManager


class _FakeMCPContext:
    """Stands in for the MCP SDK client/session contexts.

    Enters a real anyio task group in the calling task, like the task groups
    of ``streamable_http_client``/``sse_client`` and ``ClientSession``, so the
    cancel-scope rules the test relies on are the real ones. ``exited`` is set
    only after the scope is left successfully, so a swallowed close error
    leaves it False.
    """

    def __init__(self) -> None:
        self._task_group = anyio.create_task_group()
        self.exited = False

    async def __aenter__(self) -> "_FakeMCPContext":
        await self._task_group.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._task_group.__aexit__(None, None, None)
        self.exited = True


def _make_manager(contexts: list[_FakeMCPContext]) -> MCPSessionManager:
    manager = MCPSessionManager()

    async def fake_create_session(
        endpoint: str, headers: dict[str, str], endpoint_pins: dict[str, str] | None = None
    ) -> tuple[Any, Any, Any]:
        client_ctx = _FakeMCPContext()
        await client_ctx.__aenter__()
        session = _FakeMCPContext()
        await session.__aenter__()
        contexts.extend((client_ctx, session))
        return session, client_ctx, session

    manager._create_session = fake_create_session  # type: ignore[method-assign]
    return manager


def test_aexit_leaves_all_scopes_of_a_multi_server_request() -> None:
    """Two servers in one request: every session's scopes must actually be left.

    With the oldest-first close order the oldest session's scope exits fail
    while the newest session's scopes are still open, so those contexts are
    never marked exited and the task is left with unexited cancel scopes.
    """
    contexts: list[_FakeMCPContext] = []

    async def scenario() -> None:
        manager = _make_manager(contexts)
        async with manager:
            await manager.get_session("http://127.0.0.1:3101", {})
            await manager.get_session("http://127.0.0.1:3102", {})

    anyio.run(scenario)

    assert len(contexts) == 4
    assert all(ctx.exited for ctx in contexts), "some MCP session scopes were never left"


def test_aexit_leaves_the_scopes_of_a_single_session() -> None:
    contexts: list[_FakeMCPContext] = []

    async def scenario() -> None:
        manager = _make_manager(contexts)
        async with manager:
            await manager.get_session("http://127.0.0.1:3101", {})

    anyio.run(scenario)

    assert len(contexts) == 2
    assert all(ctx.exited for ctx in contexts)
