# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, MagicMock

import pytest

from ogx_api.containers import ExecInContainerResponse
from ogx_api.openai_responses import (
    OpenAIResponseOutputMessageShellCall,
)


class TestShellToolExecution:
    async def test_execute_shell_tool_creates_container_when_no_id(self):
        from ogx_api.containers import Container
        from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor

        mock_containers_api = AsyncMock()
        mock_containers_api.create_container.return_value = Container(
            id="cntr_auto123",
            name="shell-auto",
            created_at=1700000000,
            status="running",
        )
        mock_containers_api.exec_in_container.return_value = ExecInContainerResponse(
            stdout="hello world\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        executor = ToolExecutor(
            tool_groups_api=AsyncMock(),
            tool_runtime_api=AsyncMock(),
            vector_io_api=AsyncMock(),
            containers_api=mock_containers_api,
        )

        result = await executor._execute_shell_tool({"commands": ["echo hello world"]})

        mock_containers_api.create_container.assert_called_once()
        mock_containers_api.exec_in_container.assert_called_once()
        assert result.stdout == "hello world\n"
        assert result.exit_code == 0

    async def test_execute_shell_tool_uses_existing_container(self):
        mock_containers_api = AsyncMock()
        mock_containers_api.exec_in_container.return_value = ExecInContainerResponse(
            stdout="42\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor

        executor = ToolExecutor(
            tool_groups_api=AsyncMock(),
            tool_runtime_api=AsyncMock(),
            vector_io_api=AsyncMock(),
            containers_api=mock_containers_api,
        )

        result = await executor._execute_shell_tool({
            "container_id": "cntr_existing",
            "commands": ["echo 42"],
        })

        mock_containers_api.create_container.assert_not_called()
        mock_containers_api.exec_in_container.assert_called_once()
        assert result.stdout == "42\n"

    async def test_execute_shell_tool_no_containers_api_raises(self):
        from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor

        executor = ToolExecutor(
            tool_groups_api=AsyncMock(),
            tool_runtime_api=AsyncMock(),
            vector_io_api=AsyncMock(),
            containers_api=None,
        )

        with pytest.raises(ValueError, match="containers API is not configured"):
            await executor._execute_shell_tool({"commands": ["ls"]})

    async def test_execute_shell_tool_timeout(self):
        mock_containers_api = AsyncMock()
        mock_containers_api.exec_in_container.return_value = ExecInContainerResponse(
            stdout="",
            stderr="",
            exit_code=-1,
            timed_out=True,
        )

        from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor

        executor = ToolExecutor(
            tool_groups_api=AsyncMock(),
            tool_runtime_api=AsyncMock(),
            vector_io_api=AsyncMock(),
            containers_api=mock_containers_api,
        )

        result = await executor._execute_shell_tool({
            "container_id": "cntr_timeout",
            "commands": ["sleep 9999"],
            "timeout_ms": 1000,
        })

        assert result.timed_out is True

    async def test_build_shell_result_message(self):
        from ogx.providers.inline.responses.builtin.responses.tool_executor import ToolExecutor

        mock_containers_api = AsyncMock()
        executor = ToolExecutor(
            tool_groups_api=AsyncMock(),
            tool_runtime_api=AsyncMock(),
            vector_io_api=AsyncMock(),
            containers_api=mock_containers_api,
        )

        mock_function = MagicMock()
        mock_function.name = "shell"
        mock_function.arguments = '{"commands": ["ls"]}'

        result = ExecInContainerResponse(
            stdout="file1.txt\nfile2.txt\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

        ctx = MagicMock()
        ctx.response_tools = []
        output_msg, input_msg = await executor._build_result_messages(
            function=mock_function,
            tool_call_id="call_abc",
            item_id="item_1",
            tool_kwargs={"commands": ["ls"]},
            ctx=ctx,
            error_exc=None,
            result=result,
            has_error=False,
        )

        assert isinstance(output_msg, OpenAIResponseOutputMessageShellCall)
        assert output_msg.type == "shell_call"
        assert output_msg.action.commands == ["ls"]
        assert input_msg.content == "file1.txt\nfile2.txt\n"
