# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from functools import wraps

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def create_bar_chart(data, labels, title=""):
    """Create a bar chart using Rich Table."""

    console = Console()
    table = Table(title=title)
    table.add_column("Score")
    table.add_column("Count")

    max_value = max(data)
    total_count = sum(data)

    # Define a list of colors to cycle through
    colors = ["green", "blue", "red", "yellow", "magenta", "cyan"]

    for i, (label, value) in enumerate(zip(labels, data, strict=False)):
        bar_length = int((value / max_value) * 20)  # Adjust bar length as needed
        bar = "█" * bar_length + " " * (20 - bar_length)
        color = colors[i % len(colors)]
        table.add_row(label, f"[{color}]{bar}[/] {value}/{total_count}")

    console.print(table)


def handle_client_errors(operation_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                console = Console()
                console.print(
                    Panel.fit(
                        f"[bold red]Failed to {operation_name}[/bold red]\n\n"
                        f"[yellow]Error Type:[/yellow] {e.__class__.__name__}\n"
                        f"[yellow]Details:[/yellow] {str(e)}"
                    )
                )

        return wrapper

    return decorator
