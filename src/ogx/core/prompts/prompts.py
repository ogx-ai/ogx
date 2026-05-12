# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from typing import Any

from pydantic import BaseModel

from ogx.core.access_control.datatypes import AccessRule
from ogx.core.datatypes import StackConfig
from ogx.core.storage.sqlstore.authorized_sqlstore import authorized_sqlstore
from ogx_api import (
    Api,
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    ListPromptsResponse,
    ListPromptVersionsRequest,
    Prompt,
    Prompts,
    ServiceNotEnabledError,
    SetDefaultVersionRequest,
    UpdatePromptRequest,
)
from ogx_api.internal.sqlstore import ColumnDefinition, ColumnType


class PromptServiceConfig(BaseModel):
    """Configuration for the built-in prompt service.

    :param run_config: Stack run configuration containing distribution info
    :param policy: Access control rules for prompt resources
    """

    config: StackConfig
    policy: list[AccessRule] = []


async def get_provider_impl(config: PromptServiceConfig, deps: dict[Api, Any]) -> "PromptServiceImpl":
    """Get the prompt service implementation."""
    impl = PromptServiceImpl(config, deps)
    await impl.initialize()
    return impl


TABLE_PROMPTS = "prompts"


class PromptServiceImpl(Prompts):
    """Built-in prompt service implementation using AuthorizedSqlStore."""

    def __init__(self, config: PromptServiceConfig, deps: dict[Api, Any]):
        self.config = config
        self.deps = deps
        self.policy = config.policy

        prompts_ref = config.config.storage.stores.prompts
        if not prompts_ref:
            raise ServiceNotEnabledError("storage.stores.prompts")

        self.sql_store = authorized_sqlstore(prompts_ref, self.policy)

    async def initialize(self) -> None:
        await self.sql_store.create_table(
            TABLE_PROMPTS,
            {
                "id": ColumnDefinition(type=ColumnType.STRING, primary_key=True),
                "prompt_id": ColumnType.STRING,
                "version": ColumnType.INTEGER,
                "is_default": ColumnType.BOOLEAN,
                "created_at": ColumnType.INTEGER,
                "prompt_data": ColumnType.JSON,
            },
        )

    async def list_prompts(self) -> ListPromptsResponse:
        """List all prompts (default versions only)."""
        results = await self.sql_store.fetch_all(
            table=TABLE_PROMPTS,
            where={"is_default": True},
            order_by=[("prompt_id", "desc")],
        )

        prompts = [self._row_to_prompt(row) for row in results.data]
        return ListPromptsResponse(data=prompts)

    async def get_prompt(self, request: GetPromptRequest) -> Prompt:
        """Get a prompt by its identifier and optional version."""
        if request.version is not None:
            record = await self.sql_store.fetch_one(
                table=TABLE_PROMPTS,
                where={"prompt_id": request.prompt_id, "version": request.version},
            )
        else:
            record = await self.sql_store.fetch_one(
                table=TABLE_PROMPTS,
                where={"prompt_id": request.prompt_id, "is_default": True},
            )

        if record is None:
            version_label = request.version if request.version else "default"
            raise ValueError(f"Prompt {request.prompt_id}:{version_label} not found")

        return self._row_to_prompt(record)

    async def create_prompt(self, request: CreatePromptRequest) -> Prompt:
        """Create a new prompt."""
        variables = request.variables if request.variables is not None else []
        prompt_id = Prompt.generate_prompt_id()
        created_at = int(time.time())

        prompt_obj = Prompt(
            prompt_id=prompt_id,
            prompt=request.prompt,
            version=1,
            variables=variables,
        )

        await self.sql_store.insert(
            table=TABLE_PROMPTS,
            data={
                "id": f"{prompt_id}:1",
                "prompt_id": prompt_id,
                "version": 1,
                "is_default": True,
                "created_at": created_at,
                "prompt_data": {
                    "prompt": request.prompt,
                    "variables": variables,
                },
            },
        )

        return prompt_obj

    async def update_prompt(self, request: UpdatePromptRequest) -> Prompt:
        """Update an existing prompt (increments version)."""
        if request.version < 1:
            raise ValueError("Version must be >= 1")
        variables = request.variables if request.variables is not None else []

        prompt_versions = await self.list_prompt_versions(ListPromptVersionsRequest(prompt_id=request.prompt_id))
        latest_prompt = max(prompt_versions.data, key=lambda x: int(x.version))

        if request.version and latest_prompt.version != request.version:
            raise ValueError(
                f"'{request.version}' is not the latest prompt version for prompt_id='{request.prompt_id}'. Use the latest version '{latest_prompt.version}' in request."
            )

        current_version = latest_prompt.version if request.version is None else request.version
        new_version = current_version + 1
        created_at = int(time.time())

        updated_prompt = Prompt(
            prompt_id=request.prompt_id, prompt=request.prompt, version=new_version, variables=variables
        )

        await self.sql_store.insert(
            table=TABLE_PROMPTS,
            data={
                "id": f"{request.prompt_id}:{new_version}",
                "prompt_id": request.prompt_id,
                "version": new_version,
                "is_default": False,
                "created_at": created_at,
                "prompt_data": {
                    "prompt": request.prompt,
                    "variables": variables,
                },
            },
        )

        if request.set_as_default:
            await self.set_default_version(SetDefaultVersionRequest(prompt_id=request.prompt_id, version=new_version))

        return updated_prompt

    async def delete_prompt(self, request: DeletePromptRequest) -> None:
        """Delete a prompt and all its versions."""
        await self.get_prompt(GetPromptRequest(prompt_id=request.prompt_id))

        await self.sql_store.delete(table=TABLE_PROMPTS, where={"prompt_id": request.prompt_id})

    async def list_prompt_versions(self, request: ListPromptVersionsRequest) -> ListPromptsResponse:
        """List all versions of a specific prompt."""
        results = await self.sql_store.fetch_all(
            table=TABLE_PROMPTS,
            where={"prompt_id": request.prompt_id},
            order_by=[("version", "asc")],
        )

        if not results.data:
            raise ValueError(f"Prompt {request.prompt_id} not found")

        prompts = [self._row_to_prompt(row) for row in results.data]
        return ListPromptsResponse(data=prompts)

    async def set_default_version(self, request: SetDefaultVersionRequest) -> Prompt:
        """Set which version of a prompt should be the default."""
        record = await self.sql_store.fetch_one(
            table=TABLE_PROMPTS,
            where={"prompt_id": request.prompt_id, "version": request.version},
        )
        if record is None:
            raise ValueError(f"Prompt {request.prompt_id} version {request.version} not found")

        # Clear all defaults first, then set the target. This avoids the race where
        # two concurrent calls interleave "set" and "clear" and remove each other's
        # target, leaving zero defaults.
        all_versions = await self.sql_store.fetch_all(
            table=TABLE_PROMPTS,
            where={"prompt_id": request.prompt_id, "is_default": True},
        )
        for row in all_versions.data:
            await self.sql_store.update(
                table=TABLE_PROMPTS,
                data={"is_default": False},
                where={"prompt_id": request.prompt_id, "version": row["version"]},
            )

        await self.sql_store.update(
            table=TABLE_PROMPTS,
            data={"is_default": True},
            where={"prompt_id": request.prompt_id, "version": request.version},
        )

        return self._row_to_prompt(record, is_default=True)

    def _row_to_prompt(self, row: dict[str, Any], is_default: bool | None = None) -> Prompt:
        prompt_data = row.get("prompt_data", {})
        return Prompt(
            prompt_id=row["prompt_id"],
            prompt=prompt_data.get("prompt", ""),
            version=row["version"],
            variables=prompt_data.get("variables", []),
            is_default=is_default if is_default is not None else row.get("is_default", False),
        )

    async def shutdown(self) -> None:
        pass
