# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from typing import Any

from ogx.core.datatypes import AccessRule
from ogx.core.storage.datatypes import SqlStoreReference
from ogx.core.storage.sqlstore.authorized_sqlstore import AuthorizedSqlStore
from ogx.core.storage.sqlstore.sqlstore import sqlstore_impl
from ogx.log import get_logger
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType

logger = get_logger(name=__name__, category="interactions")


class InteractionsStore:
    """Persistent store for Google Interactions API state with SQL-backed storage.

    Stores completed interaction data (messages + output) so that subsequent
    requests can chain via ``previous_interaction_id``.
    """

    def __init__(self, reference: SqlStoreReference, policy: list[AccessRule]):
        self.reference = reference
        self.policy = policy

    async def initialize(self) -> None:
        """Create the interactions table if it does not exist."""
        base_store = sqlstore_impl(self.reference)
        self.sql_store = AuthorizedSqlStore(base_store, self.policy)
        await self.sql_store.create_table(
            self.reference.table_name,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "created_at": ColumnType.INTEGER,
                "model": ColumnType.STRING,
                "interaction_data": ColumnType.JSON,
            },
        )

    async def shutdown(self) -> None:
        return

    async def store_interaction(
        self,
        interaction_id: str,
        created_at: int,
        model: str,
        messages: list[dict[str, Any]],
        output_text: str,
    ) -> None:
        """Persist a completed interaction for future chaining."""
        await self.sql_store.insert(
            self.reference.table_name,
            {
                "id": interaction_id,
                "created_at": created_at,
                "model": model,
                "interaction_data": {
                    "messages": messages,
                    "output_text": output_text,
                },
            },
        )

    async def get_interaction(self, interaction_id: str) -> dict[str, Any] | None:
        """Retrieve a stored interaction by ID.

        Returns the ``interaction_data`` dict (messages + output_text), or
        ``None`` if not found.
        """
        row = await self.sql_store.fetch_one(
            self.reference.table_name,
            where={"id": interaction_id},
        )
        if not row:
            return None
        data: dict[str, Any] = row["interaction_data"]
        return data
