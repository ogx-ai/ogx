# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx_api.openai_responses import (
    OpenAIResponseInputToolShell,
    OpenAIResponseOutputMessageShellCall,
    OpenAIResponseOutputMessageShellCallOutput,
    ShellCallAction,
    ShellContainerAutoEnvironment,
    ShellContainerReferenceEnvironment,
    ShellExitOutcome,
    ShellOutputContent,
    ShellTimeoutOutcome,
)


class TestShellToolInputModels:
    def test_shell_tool_minimal(self):
        tool = OpenAIResponseInputToolShell()
        assert tool.type == "shell"
        assert tool.environment is None

    def test_shell_tool_container_auto(self):
        tool = OpenAIResponseInputToolShell(
            environment=ShellContainerAutoEnvironment(),
        )
        assert tool.type == "shell"
        assert tool.environment.type == "container_auto"

    def test_shell_tool_container_auto_with_options(self):
        tool = OpenAIResponseInputToolShell(
            environment=ShellContainerAutoEnvironment(
                memory_limit="4g",
                file_ids=["file-abc", "file-def"],
            ),
        )
        assert tool.environment.memory_limit == "4g"
        assert tool.environment.file_ids == ["file-abc", "file-def"]

    def test_shell_tool_container_reference(self):
        tool = OpenAIResponseInputToolShell(
            environment=ShellContainerReferenceEnvironment(container_id="cntr_abc123"),
        )
        assert tool.type == "shell"
        assert tool.environment.type == "container_reference"
        assert tool.environment.container_id == "cntr_abc123"

    def test_shell_tool_discriminated_union_from_dict(self):
        tool = OpenAIResponseInputToolShell.model_validate(
            {"type": "shell", "environment": {"type": "container_auto"}}
        )
        assert isinstance(tool.environment, ShellContainerAutoEnvironment)

        tool2 = OpenAIResponseInputToolShell.model_validate(
            {"type": "shell", "environment": {"type": "container_reference", "container_id": "cntr_x"}}
        )
        assert isinstance(tool2.environment, ShellContainerReferenceEnvironment)


class TestShellCallOutputModels:
    def test_shell_call_action(self):
        action = ShellCallAction(commands=["ls -la", "echo hello"])
        assert action.commands == ["ls -la", "echo hello"]
        assert action.timeout_ms is None
        assert action.max_output_length is None

    def test_shell_call_action_with_limits(self):
        action = ShellCallAction(commands=["python3 script.py"], timeout_ms=120000, max_output_length=4096)
        assert action.timeout_ms == 120000
        assert action.max_output_length == 4096

    def test_shell_call_output_item(self):
        call = OpenAIResponseOutputMessageShellCall(
            id="item_1",
            call_id="call_abc",
            action=ShellCallAction(commands=["ls -la"]),
            status="completed",
        )
        assert call.type == "shell_call"
        assert call.action.commands == ["ls -la"]
        assert call.status == "completed"

    def test_shell_exit_outcome(self):
        outcome = ShellExitOutcome(exit_code=0)
        assert outcome.type == "exit"
        assert outcome.exit_code == 0

    def test_shell_exit_outcome_nonzero(self):
        outcome = ShellExitOutcome(exit_code=127)
        assert outcome.type == "exit"
        assert outcome.exit_code == 127

    def test_shell_timeout_outcome(self):
        outcome = ShellTimeoutOutcome()
        assert outcome.type == "timeout"

    def test_shell_output_content_exit(self):
        content = ShellOutputContent(
            stdout="hello\n",
            stderr="",
            outcome=ShellExitOutcome(exit_code=0),
        )
        assert content.stdout == "hello\n"
        assert content.stderr == ""
        assert content.outcome.type == "exit"
        assert content.outcome.exit_code == 0

    def test_shell_output_content_timeout(self):
        content = ShellOutputContent(
            stdout="partial...",
            stderr="",
            outcome=ShellTimeoutOutcome(),
        )
        assert content.outcome.type == "timeout"

    def test_shell_call_output_message(self):
        output = OpenAIResponseOutputMessageShellCallOutput(
            id="item_2",
            call_id="call_abc",
            output=[
                ShellOutputContent(
                    stdout="42\n",
                    stderr="",
                    outcome=ShellExitOutcome(exit_code=0),
                )
            ],
            status="completed",
        )
        assert output.type == "shell_call_output"
        assert len(output.output) == 1
        assert output.output[0].stdout == "42\n"

    def test_shell_call_output_multiple_chunks(self):
        output = OpenAIResponseOutputMessageShellCallOutput(
            id="item_3",
            call_id="call_def",
            output=[
                ShellOutputContent(stdout="line1\n", stderr="", outcome=ShellExitOutcome(exit_code=0)),
                ShellOutputContent(stdout="", stderr="error\n", outcome=ShellExitOutcome(exit_code=1)),
            ],
            status="completed",
        )
        assert len(output.output) == 2
        assert output.output[1].stderr == "error\n"
        assert output.output[1].outcome.exit_code == 1
