# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol

from ogx_api.conversations import AddItemsRequest, Conversations


class ConversationItemSync(Protocol):
    """Server-internal item sync used by the Responses provider.

    It is not an HTTP route and deliberately not part of the public ``Conversations``
    protocol in ``ogx_api``, so external implementations are not required to provide it.
    """

    async def sync_items(self, conversation_id: str, request: AddItemsRequest) -> None:
        """Append items keeping their ids and skipping ids already in the conversation."""
        ...


class SyncableConversations(Conversations, ConversationItemSync, Protocol):
    """The conversations service as the builtin Responses provider depends on it."""
