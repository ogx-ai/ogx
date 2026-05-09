# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import traceback

import click
from rich.console import Console

from ...inference.event_logger import EventLogger
from ..common.utils import handle_client_errors


@click.group()
@click.help_option("-h", "--help")
def inference():
    """Inference (chat)."""


@click.command("chat-completion")
@click.help_option("-h", "--help")
@click.option("--message", help="Message")
@click.option("--stream", is_flag=True, help="Streaming", default=False)
@click.option("--session", is_flag=True, help="Start a Chat Session", default=False)
@click.option("--model-id", required=False, help="Model ID")
@click.pass_context
@handle_client_errors("inference chat-completion")
def chat_completion(ctx, message: str, stream: bool, session: bool, model_id: str | None):
    """Show available inference chat completion endpoints on distribution endpoint"""
    if not message and not session:
        click.secho(
            "you must specify either --message or --session",
            fg="red",
        )
        raise click.exceptions.Exit(1)
    client = ctx.obj["client"]
    console = Console()

    if not model_id:
        available_models = [model.identifier for model in client.models.list() if model.model_type == "llm"]
        model_id = available_models[0]

    messages = []
    if message:
        messages.append({"role": "user", "content": message})
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=stream,
        )
        if not stream:
            console.print(response)
        else:
            for event in EventLogger().log(response):
                event.print()
    if session:
        chat_session(client=client, model_id=model_id, messages=messages, console=console)


def chat_session(client, model_id: str | None, messages: list[dict[str, str]], console: Console):
    """Run an interactive chat session with the served model"""
    while True:
        try:
            message = input(">>> ")
            if message in ["\\q", "quit"]:
                console.print("Exiting")
                break
            messages.append({"role": "user", "content": message})
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
            )
            for event in EventLogger().log(response):
                event.print()
        except Exception as exc:
            traceback.print_exc()
            console.print(f"Error in chat session {exc}")
            break
        except KeyboardInterrupt:
            console.print("\nDetected user interrupt, exiting")
            break


# Register subcommands
inference.add_command(chat_completion)
