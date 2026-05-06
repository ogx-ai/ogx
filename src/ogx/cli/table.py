# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import Iterable, Sequence

from rich.console import Console
from rich.table import Table


def print_table(
    rows: Sequence[Sequence[str | None]],
    headers: Sequence[str] | None = None,
    separate_rows: bool = False,
    sort_by: Iterable[int] = tuple(),
) -> None:
    """Print a formatted table to the console using Rich.

    Args:
        rows: list of row data, where each row is a list of strings.
        headers: optional list of column header strings.
        separate_rows: whether to draw lines between rows.
        sort_by: column indices to sort rows by.
    """
    # Convert rows and handle None values
    rows = [[x or "" for x in row] for row in rows]

    # Sort rows if sort_by is specified
    if sort_by:
        rows.sort(key=lambda x: tuple(x[i] for i in sort_by))

    # Create Rich table
    table = Table(show_lines=separate_rows)

    # Add headers if provided
    if headers:
        for header in headers:
            table.add_column(header, style="bold white")
    else:
        # Add unnamed columns based on first row
        for _ in range(len(rows[0]) if rows else 0):
            table.add_column()

    # Add rows
    for row in rows:
        table.add_row(*row)

    # Print table
    console = Console()
    console.print(table)
