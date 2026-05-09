# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from ogx_api import PaginatedResponse


def paginate_records(
    records: list[dict[str, Any]],
    start_index: int | None = None,
    limit: int | None = None,
) -> PaginatedResponse:
    """Helper function to handle pagination of records consistently across implementations.
    Inspired by stripe's pagination: https://docs.stripe.com/api/pagination

    :param records: List of records to paginate
    :param start_index: The starting index (0-based). If None, starts from beginning.
    :param limit: Number of items to return. If None or -1, returns all items.
    :return: PaginatedResponse with the paginated data
    """
    # Handle special case for fetching all rows
    if limit is None or limit == -1:
        return PaginatedResponse(
            data=records,
            has_more=False,
        )

    # Use offset-based pagination
    start_index = start_index or 0
    end_index = min(start_index + limit, len(records))
    page_data = records[start_index:end_index]

    # Calculate if there are more records
    has_more = end_index < len(records)

    return PaginatedResponse(
        data=page_data,
        has_more=has_more,
    )
