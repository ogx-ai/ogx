# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

from pydantic import BaseModel

from ogx.core.datatypes import StackConfig
from ogx.core.storage.kvstore import KVStore, kvstore_impl
from ogx_api import (
    Api,
    CreatePromptRequest,
    DeletePromptRequest,
    GetPromptRequest,
    ListPromptsResponse,
    ListPromptVersionsRequest,
    Prompt,
    Prompts,
    SetDefaultVersionRequest,
    UpdatePromptRequest,
)


class PromptServiceConfig(BaseModel):
    """Configuration for the built-in prompt service.

    :param run_config: Stack run configuration containing distribution info
    """

    config: StackConfig


async def get_provider_impl(config: PromptServiceConfig, deps: dict[Api, Any]) -> "PromptServiceImpl":
    """Get the prompt service implementation."""
    impl = PromptServiceImpl(config, deps)
    await impl.initialize()
    return impl


class PromptServiceImpl(Prompts):
    """Built-in prompt service implementation using KVStore."""

    def __init__(self, config: PromptServiceConfig, deps: dict[Api, Any]):
        self.stack_config = config.config
        self.deps = deps
        self.kvstore: KVStore

    async def initialize(self) -> None:
        # Use prompts store reference from run config
        prompts_ref = self.stack_config.storage.stores.prompts
        if not prompts_ref:
            raise ValueError("storage.stores.prompts must be configured in run config")
        self.kvstore = await kvstore_impl(prompts_ref)

    def _get_default_key(self, prompt_id: str) -> str:
        """Get the KVStore key that stores the default version number."""
        return f"prompts:v1:{prompt_id}:default"

    async def _get_prompt_key(self, prompt_id: str, version: int | None = None) -> str:
        """Get the KVStore key for prompt data, returning default version if applicable."""
        if version:
            return self._get_version_key(prompt_id, str(version))

        default_key = self._get_default_key(prompt_id)
        resolved_version = await self.kvstore.get(default_key)
        if resolved_version is None:
            raise ValueError(f"Prompt {prompt_id}:default not found")
        return self._get_version_key(prompt_id, resolved_version)

    def _get_version_key(self, prompt_id: str, version: str) -> str:
        """Get the KVStore key for a specific prompt version."""
        return f"prompts:v1:{prompt_id}:{version}"

    def _get_list_key_prefix(self) -> str:
        """Get the key prefix for listing prompts."""
        return "prompts:v1:"

    def _serialize_prompt(self, prompt: Prompt) -> str:
        """Serialize a prompt to JSON string for storage."""
        return json.dumps(
            {
                "prompt_id": prompt.prompt_id,
                "prompt": prompt.prompt,
                "version": prompt.version,
                "variables": prompt.variables or [],
                "is_default": prompt.is_default,
            }
        )

    def _deserialize_prompt(self, data: str) -> Prompt:
        """Deserialize a prompt from JSON string."""
        obj = json.loads(data)
        return Prompt(
            prompt_id=obj["prompt_id"],
            prompt=obj["prompt"],
            version=obj["version"],
            variables=obj.get("variables", []),
            is_default=obj.get("is_default", False),
        )

    async def list_prompts(self) -> ListPromptsResponse:
        """List all prompts (default versions only)."""
        prefix = self._get_list_key_prefix()
        keys = await self.kvstore.keys_in_range(prefix, prefix + "\xff")

        prompts = []
        for key in keys:
            if key.endswith(":default"):
                try:
                    default_version = await self.kvstore.get(key)
                    if default_version:
                        prompt_id = key.replace(prefix, "").replace(":default", "")
                        version_key = self._get_version_key(prompt_id, default_version)
                        data = await self.kvstore.get(version_key)
                        if data:
                            prompt = self._deserialize_prompt(data)
                            prompts.append(prompt)
                except (json.JSONDecodeError, KeyError):
                    continue

        prompts.sort(key=lambda p: p.prompt_id or "", reverse=True)
        return ListPromptsResponse(data=prompts)

    async def get_prompt(self, request: GetPromptRequest) -> Prompt:
        """Get a prompt by its identifier and optional version."""
        key = await self._get_prompt_key(request.prompt_id, request.version)
        data = await self.kvstore.get(key)
        if data is None:
            raise ValueError(
                f"Prompt {request.prompt_id}:{request.version if request.version else 'default'} not found"
            )
        return self._deserialize_prompt(data)

    async def create_prompt(self, request: CreatePromptRequest) -> Prompt:
        """Create a new prompt."""
        variables = request.variables if request.variables is not None else []

        prompt_obj = Prompt(
            prompt_id=Prompt.generate_prompt_id(),
            prompt=request.prompt,
            version=1,
            variables=variables,
        )

        version_key = self._get_version_key(prompt_obj.prompt_id, str(prompt_obj.version))
        data = self._serialize_prompt(prompt_obj)
        await self.kvstore.set(version_key, data)

        default_key = self._get_default_key(prompt_obj.prompt_id)
        await self.kvstore.set(default_key, str(prompt_obj.version))

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

        updated_prompt = Prompt(
            prompt_id=request.prompt_id, prompt=request.prompt, version=new_version, variables=variables
        )

        version_key = self._get_version_key(request.prompt_id, str(new_version))
        data = self._serialize_prompt(updated_prompt)
        await self.kvstore.set(version_key, data)

        if request.set_as_default:
            await self.set_default_version(SetDefaultVersionRequest(prompt_id=request.prompt_id, version=new_version))

        return updated_prompt

    async def delete_prompt(self, request: DeletePromptRequest) -> None:
        """Delete a prompt and all its versions."""
        await self.get_prompt(GetPromptRequest(prompt_id=request.prompt_id))

        prefix = f"prompts:v1:{request.prompt_id}:"
        keys = await self.kvstore.keys_in_range(prefix, prefix + "\xff")

        for key in keys:
            await self.kvstore.delete(key)

    async def list_prompt_versions(self, request: ListPromptVersionsRequest) -> ListPromptsResponse:
        """List all versions of a specific prompt."""
        prefix = f"prompts:v1:{request.prompt_id}:"
        keys = await self.kvstore.keys_in_range(prefix, prefix + "\xff")

        default_version = None
        prompts = []

        for key in keys:
            data = await self.kvstore.get(key)
            if key.endswith(":default"):
                default_version = data
            else:
                if data:
                    prompt_obj = self._deserialize_prompt(data)
                    prompts.append(prompt_obj)

        if not prompts:
            raise ValueError(f"Prompt {request.prompt_id} not found")

        for prompt in prompts:
            prompt.is_default = str(prompt.version) == default_version

        prompts.sort(key=lambda x: x.version)
        return ListPromptsResponse(data=prompts)

    async def set_default_version(self, request: SetDefaultVersionRequest) -> Prompt:
        """Set which version of a prompt should be the default, If not set. the default is the latest."""
        version_key = self._get_version_key(request.prompt_id, str(request.version))
        data = await self.kvstore.get(version_key)
        if data is None:
            raise ValueError(f"Prompt {request.prompt_id} version {request.version} not found")

        default_key = self._get_default_key(request.prompt_id)
        await self.kvstore.set(default_key, str(request.version))

        return self._deserialize_prompt(data)

    async def shutdown(self) -> None:
        pass
