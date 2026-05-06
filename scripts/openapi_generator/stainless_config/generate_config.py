# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

HEADER = "# yaml-language-server: $schema=https://app.stainlessapi.com/config-internal.schema.json\n\n"

SECTION_ORDER = [
    "organization",
    "security",
    "security_schemes",
    "targets",
    "client_settings",
    "environments",
    "pagination",
    "streaming",
    "settings",
    "openapi",
    "readme",
    "resources",
]

ORGANIZATION = {
    "name": "ogx-client",
    "docs": "https://ogx.readthedocs.io/en/latest/",
    "contact": "contributors@ogx.dev",
}

SECURITY = [{}, {"BearerAuth": []}]

SECURITY_SCHEMES = {"BearerAuth": {"type": "http", "scheme": "bearer"}}

TARGETS = {
    "node": {
        "package_name": "ogx-client",
        "production_repo": "ogx-ai/ogx-client-typescript",
        "publish": {"npm": False},
    },
    "python": {
        "package_name": "ogx_client",
        "production_repo": "ogx-ai/ogx-client-python",
        "options": {"use_uv": True},
        "publish": {"pypi": True},
        "project_name": "ogx_client",
    },
    "go": {
        "package_name": "ogx-client",
        "production_repo": "ogx-ai/ogx-client-go",
        "options": {"enable_v2": True, "back_compat_use_shared_package": False},
    },
}

CLIENT_SETTINGS = {
    "default_env_prefix": "OGX_CLIENT",
    "opts": {
        "api_key": {
            "type": "string",
            "read_env": "OGX_CLIENT_API_KEY",
            "auth": {"security_scheme": "BearerAuth"},
            "nullable": True,
        }
    },
}

ENVIRONMENTS = {"production": "http://any-hosted-ogx.com"}

PAGINATION = [
    {
        "name": "openai_cursor_page",
        "type": "cursor",
        "request": {
            "limit": {"type": "integer"},
            "after": {
                "type": "string",
                "x-stainless-pagination-property": {"purpose": "next_cursor_param"},
            },
        },
        "response": {
            "data": {"type": "array", "items": {}},
            "has_more": {"type": "boolean"},
            "last_id": {
                "type": "string",
                "x-stainless-pagination-property": {"purpose": "next_cursor_field"},
            },
        },
    },
]

STREAMING = {
    "on_event": [
        {"data_starts_with": "[DONE]", "handle": "done"},
        {"kind": "fallthrough", "handle": "yield", "error_property": "error"},
    ]
}

SETTINGS = {
    "license": "MIT",
    "unwrap_response_fields": ["data"],
    "file_header": "Copyright (c) The OGX Contributors.\n"
    "All rights reserved.\n"
    "\n"
    "This source code is licensed under the terms described in the "
    "LICENSE file in\n"
    "the root directory of this source tree.\n",
}

OPENAPI = {
    "transformations": [
        {
            "command": "oneOfToAnyOf",
            "reason": "Prism (mock server) doesn't like one of our "
            "requests as it technically matches multiple "
            "variants",
        },
    ]
}

README = {
    "example_requests": {
        "default": {
            "type": "request",
            "endpoint": "post /v1/chat/completions",
            "params": {},
        },
        "headline": {"type": "request", "endpoint": "get /v1/models", "params": {}},
        "pagination": {
            "type": "request",
            "endpoint": "post /v1/chat/completions",
            "params": {},
        },
    }
}

ALL_RESOURCES = {
    "$shared": {
        "models": {
            "interleaved_content_item": "InterleavedContentItem",
            "interleaved_content": "InterleavedContent",
            "param_type": "ParamType",
            "safety_violation": "SafetyViolation",
            "sampling_params": "SamplingParams",
            "system_message": "SystemMessage",
            "health_info": "HealthInfo",
            "provider_info": "ProviderInfo",
            "list_providers_response": "ListProvidersResponse",
            "route_info": "RouteInfo",
            "list_routes_response": "ListRoutesResponse",
            "version_info": "VersionInfo",
        }
    },
    "responses": {
        "models": {
            "response_object_stream": "OpenAIResponseObjectStream",
            "response_object": "OpenAIResponseObject",
            "compacted_response": "OpenAICompactedResponse",
            "response_input": "OpenAIResponseInput",
            "response_message": "OpenAIResponseMessage",
            "response_output": "OpenAIResponseOutput",
        },
        "methods": {
            "create": {
                "type": "http",
                "streaming": {
                    "stream_event_model": "responses.response_object_stream",
                    "param_discriminator": "stream",
                },
                "endpoint": "post /v1/responses",
            },
            "retrieve": "get /v1/responses/{response_id}",
            "list": {"type": "http", "endpoint": "get /v1/responses"},
            "delete": {
                "type": "http",
                "endpoint": "delete /v1/responses/{response_id}",
            },
            "compact": {
                "type": "http",
                "endpoint": "post /v1/responses/compact",
            },
        },
        "subresources": {
            "input_items": {
                "methods": {
                    "list": {
                        "type": "http",
                        "paginated": False,
                        "endpoint": "get /v1/responses/{response_id}/input_items",
                    }
                }
            }
        },
    },
    "prompts": {
        "models": {"prompt": "Prompt", "list_prompts_response": "ListPromptsResponse"},
        "methods": {
            "create": "post /v1/prompts",
            "list": {"paginated": False, "endpoint": "get /v1/prompts"},
            "retrieve": "get /v1/prompts/{prompt_id}",
            "update": "put /v1/prompts/{prompt_id}",
            "delete": "delete /v1/prompts/{prompt_id}",
            "set_default_version": "put /v1/prompts/{prompt_id}/set-default-version",
        },
        "subresources": {
            "versions": {
                "methods": {
                    "list": {
                        "paginated": False,
                        "endpoint": "get /v1/prompts/{prompt_id}/versions",
                    }
                }
            }
        },
    },
    "conversations": {
        "models": {"conversation_object": "Conversation"},
        "methods": {
            "create": {"type": "http", "endpoint": "post /v1/conversations"},
            "retrieve": "get /v1/conversations/{conversation_id}",
            "update": {
                "type": "http",
                "endpoint": "post /v1/conversations/{conversation_id}",
            },
            "delete": {
                "type": "http",
                "endpoint": "delete /v1/conversations/{conversation_id}",
            },
        },
        "subresources": {
            "items": {
                "methods": {
                    "get": {
                        "type": "http",
                        "endpoint": "get /v1/conversations/{conversation_id}/items/{item_id}",
                    },
                    "list": {
                        "type": "http",
                        "endpoint": "get /v1/conversations/{conversation_id}/items",
                    },
                    "create": {
                        "type": "http",
                        "endpoint": "post /v1/conversations/{conversation_id}/items",
                    },
                    "delete": {
                        "type": "http",
                        "endpoint": "delete /v1/conversations/{conversation_id}/items/{item_id}",
                    },
                }
            }
        },
    },
    "inspect": {
        "methods": {"health": "get /v1/health", "version": "get /v1/version"},
    },
    "embeddings": {
        "models": {"create_embeddings_response": "OpenAIEmbeddingsResponse"},
        "methods": {"create": "post /v1/embeddings"},
    },
    "chat": {
        "models": {"chat_completion_chunk": "OpenAIChatCompletionChunk"},
        "subresources": {
            "completions": {
                "methods": {
                    "create": {
                        "type": "http",
                        "streaming": {
                            "stream_event_model": "chat.chat_completion_chunk",
                            "param_discriminator": "stream",
                        },
                        "endpoint": "post /v1/chat/completions",
                    },
                    "list": {
                        "type": "http",
                        "paginated": False,
                        "endpoint": "get /v1/chat/completions",
                    },
                    "retrieve": {
                        "type": "http",
                        "endpoint": "get /v1/chat/completions/{completion_id}",
                    },
                }
            }
        },
    },
    "completions": {
        "methods": {
            "create": {
                "type": "http",
                "streaming": {"param_discriminator": "stream"},
                "endpoint": "post /v1/completions",
            }
        }
    },
    "vector_io": {
        "models": {"queryChunksResponse": "QueryChunksResponse"},
        "methods": {
            "insert": "post /v1/vector-io/insert",
            "query": "post /v1/vector-io/query",
        },
    },
    "vector_stores": {
        "models": {
            "vector_store": "VectorStoreObject",
            "list_vector_stores_response": "VectorStoreListResponse",
            "vector_store_delete_response": "VectorStoreDeleteResponse",
            "vector_store_search_response": "VectorStoreSearchResponsePage",
        },
        "methods": {
            "create": "post /v1/vector_stores",
            "list": "get /v1/vector_stores",
            "retrieve": "get /v1/vector_stores/{vector_store_id}",
            "update": "post /v1/vector_stores/{vector_store_id}",
            "delete": "delete /v1/vector_stores/{vector_store_id}",
            "search": "post /v1/vector_stores/{vector_store_id}/search",
        },
        "subresources": {
            "files": {
                "models": {"vector_store_file": "VectorStoreFileObject"},
                "methods": {
                    "list": "get /v1/vector_stores/{vector_store_id}/files",
                    "retrieve": "get /v1/vector_stores/{vector_store_id}/files/{file_id}",
                    "update": "post /v1/vector_stores/{vector_store_id}/files/{file_id}",
                    "delete": "delete /v1/vector_stores/{vector_store_id}/files/{file_id}",
                    "create": "post /v1/vector_stores/{vector_store_id}/files",
                    "content": "get /v1/vector_stores/{vector_store_id}/files/{file_id}/content",
                },
            },
            "file_batches": {
                "models": {
                    "vector_store_file_batches": "VectorStoreFileBatchObject",
                    "list_vector_store_files_in_batch_response": "VectorStoreFilesListInBatchResponse",
                },
                "methods": {
                    "create": "post /v1/vector_stores/{vector_store_id}/file_batches",
                    "retrieve": "get /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}",
                    "list_files": "get /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files",
                    "cancel": "post /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel",
                },
            },
        },
    },
    "models": {
        "models": {
            "model": "OpenAIModel",
            "list_models_response": "OpenAIListModelsResponse",
        },
        "methods": {
            "list": {"paginated": False, "endpoint": "get /v1/models"},
            "retrieve": "get /v1/models/{model_id}",
        },
        "subresources": {"openai": {"methods": {"list": {"paginated": False, "endpoint": "get /v1/models"}}}},
    },
    "providers": {
        "methods": {
            "list": {"paginated": False, "endpoint": "get /v1/providers"},
            "retrieve": "get /v1/providers/{provider_id}",
        },
    },
    "routes": {
        "methods": {"list": {"paginated": False, "endpoint": "get /v1/inspect/routes"}},
    },
    "moderations": {
        "models": {"create_response": "ModerationObject"},
        "methods": {"create": "post /v1/moderations"},
    },
    "safety": {
        "models": {"run_shield_response": "RunShieldResponse"},
        "methods": {"run_shield": "post /v1/safety/run-shield"},
    },
    "shields": {
        "models": {"shield": "Shield", "list_shields_response": "ListShieldsResponse"},
        "methods": {
            "retrieve": "get /v1/shields/{identifier}",
            "list": {"paginated": False, "endpoint": "get /v1/shields"},
            "register": "post /v1/shields",
            "delete": "delete /v1/shields/{identifier}",
        },
    },
    "files": {
        "models": {
            "file": "OpenAIFileObject",
            "list_files_response": "ListOpenAIFileResponse",
            "delete_file_response": "OpenAIFileDeleteResponse",
        },
        "methods": {
            "create": "post /v1/files",
            "list": "get /v1/files",
            "retrieve": "get /v1/files/{file_id}",
            "delete": "delete /v1/files/{file_id}",
            "content": "get /v1/files/{file_id}/content",
        },
    },
    "batches": {
        "methods": {
            "create": "post /v1/batches",
            "list": "get /v1/batches",
            "retrieve": "get /v1/batches/{batch_id}",
            "cancel": "post /v1/batches/{batch_id}/cancel",
        }
    },
    "alpha": {
        "subresources": {
            "admin": {
                "methods": {
                    "list_providers": "get /v1alpha/admin/providers",
                    "inspect_provider": "get /v1alpha/admin/providers/{provider_id}",
                    "list_routes": "get /v1alpha/admin/inspect/routes",
                    "health": "get /v1alpha/admin/health",
                    "version": "get /v1alpha/admin/version",
                },
            },
            "inference": {
                "methods": {
                    "rerank": "post /v1alpha/inference/rerank",
                },
            },
        }
    },
}


HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


@dataclass
class Endpoint:
    method: str
    path: str
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, value: Any) -> Endpoint:
        if isinstance(value, str):
            method, _, path = value.partition(" ")
            return cls._from_parts(method, path)
        if isinstance(value, dict) and "endpoint" in value:
            method, _, path = value["endpoint"].partition(" ")
            extra = {k: v for k, v in value.items() if k != "endpoint"}
            endpoint = cls._from_parts(method, path)
            endpoint.extra.update(extra)
            return endpoint
        raise ValueError(f"Unsupported endpoint value: {value!r}")

    @classmethod
    def _from_parts(cls, method: str, path: str) -> Endpoint:
        method = method.strip().lower()
        path = path.strip()
        if method not in HTTP_METHODS:
            raise ValueError(f"Unsupported HTTP method for Stainless config: {method!r}")
        if not path.startswith("/"):
            raise ValueError(f"Endpoint path must start with '/': {path!r}")
        return cls(method=method, path=path)

    def to_config(self) -> Any:
        if not self.extra:
            return f"{self.method} {self.path}"
        data = dict(self.extra)
        data["endpoint"] = f"{self.method} {self.path}"
        return data

    def route_key(self) -> str:
        return f"{self.method} {self.path}"


@dataclass
class Resource:
    models: dict[str, str] | None = None
    methods: dict[str, Endpoint] = field(default_factory=dict)
    subresources: dict[str, Resource] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Resource:
        models = data.get("models")
        methods = {name: Endpoint.from_config(value) for name, value in data.get("methods", {}).items()}
        subresources = {name: cls.from_dict(value) for name, value in data.get("subresources", {}).items()}
        return cls(models=models, methods=methods, subresources=subresources)

    def to_config(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.models:
            result["models"] = self.models
        if self.methods:
            result["methods"] = {name: endpoint.to_config() for name, endpoint in self.methods.items()}
        if self.subresources:
            result["subresources"] = {name: resource.to_config() for name, resource in self.subresources.items()}
        return result

    def collect_endpoint_paths(self) -> set[str]:
        paths = {endpoint.route_key() for endpoint in self.methods.values()}
        for subresource in self.subresources.values():
            paths.update(subresource.collect_endpoint_paths())
        return paths

    def iter_endpoints(self, prefix: str) -> Iterator[tuple[str, str]]:
        for method_name, endpoint in self.methods.items():
            label = f"{prefix}.{method_name}" if prefix else method_name
            yield endpoint.route_key(), label
        for sub_name, subresource in self.subresources.items():
            sub_prefix = f"{prefix}.{sub_name}" if prefix else sub_name
            yield from subresource.iter_endpoints(sub_prefix)


_RESOURCES = {name: Resource.from_dict(data) for name, data in ALL_RESOURCES.items()}


def _load_openapi_paths(openapi_path: Path) -> set[str]:
    spec = yaml.safe_load(openapi_path.read_text()) or {}
    paths: set[str] = set()
    for path, path_item in (spec.get("paths") or {}).items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            paths.add(f"{str(method).lower()} {path}")
    return paths


@dataclass(frozen=True)
class StainlessConfig:
    organization: dict[str, Any]
    security: list[Any]
    security_schemes: dict[str, Any]
    targets: dict[str, Any]
    client_settings: dict[str, Any]
    environments: dict[str, Any]
    pagination: list[dict[str, Any]]
    streaming: dict[str, Any]
    settings: dict[str, Any]
    openapi: dict[str, Any]
    readme: dict[str, Any]
    resources: dict[str, Resource]

    @classmethod
    def make(cls) -> StainlessConfig:
        return cls(
            organization=ORGANIZATION,
            security=SECURITY,
            security_schemes=SECURITY_SCHEMES,
            targets=TARGETS,
            client_settings=CLIENT_SETTINGS,
            environments=ENVIRONMENTS,
            pagination=PAGINATION,
            streaming=STREAMING,
            settings=SETTINGS,
            openapi=OPENAPI,
            readme=README,
            resources=dict(_RESOURCES),
        )

    def referenced_paths(self) -> set[str]:
        paths: set[str] = set()
        for resource in self.resources.values():
            paths.update(resource.collect_endpoint_paths())
        paths.update(self.readme_endpoint_paths())
        return paths

    def readme_endpoint_paths(self) -> set[str]:
        example_requests = self.readme.get("example_requests", {}) if self.readme else {}
        paths: set[str] = set()
        for entry in example_requests.values():
            endpoint = entry.get("endpoint") if isinstance(entry, dict) else None
            if isinstance(endpoint, str):
                method, _, route = endpoint.partition(" ")
                method = method.strip().lower()
                route = route.strip()
                if method and route:
                    paths.add(f"{method} {route}")
        return paths

    def endpoint_map(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for resource_name, resource in self.resources.items():
            for route, label in resource.iter_endpoints(resource_name):
                mapping.setdefault(route, []).append(label)
        return mapping

    def validate_unique_endpoints(self) -> None:
        duplicates: dict[str, list[str]] = {}
        for route, labels in self.endpoint_map().items():
            top_levels = {label.split(".", 1)[0] for label in labels}
            if len(top_levels) > 1:
                duplicates[route] = labels
        if duplicates:
            formatted = "\n".join(
                f"  - {route} defined in: {', '.join(sorted(labels))}" for route, labels in sorted(duplicates.items())
            )
            raise ValueError("Duplicate endpoints found across resources:\n" + formatted)

    def validate_readme_endpoints(self) -> None:
        resource_paths: set[str] = set()
        for resource in self.resources.values():
            resource_paths.update(resource.collect_endpoint_paths())
        missing = sorted(path for path in self.readme_endpoint_paths() if path not in resource_paths)
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise ValueError("README example endpoints are not present in Stainless resources:\n" + formatted)

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for section in SECTION_ORDER:
            if section == "resources":
                cfg[section] = {name: resource.to_config() for name, resource in self.resources.items()}
                continue
            cfg[section] = getattr(self, section)
        return cfg

    def validate_against_openapi(self, openapi_path: Path) -> None:
        if not openapi_path.exists():
            raise FileNotFoundError(f"OpenAPI spec not found at {openapi_path}")
        spec_paths = _load_openapi_paths(openapi_path)
        config_paths = self.referenced_paths()
        missing = sorted(path for path in config_paths if path not in spec_paths)
        if missing:
            formatted = "\n".join(f"  - {path}" for path in missing)
            raise ValueError("Stainless config references missing endpoints:\n" + formatted)

    def validate(self, openapi_path: Path | None = None) -> None:
        self.validate_unique_endpoints()
        self.validate_readme_endpoints()
        if openapi_path is not None:
            self.validate_against_openapi(openapi_path)


def build_config() -> dict[str, Any]:
    return StainlessConfig.make().to_dict()


def write_config(repo_root: Path, openapi_path: Path | None = None) -> Path:
    stainless_config = StainlessConfig.make()
    spec_path = (openapi_path or (repo_root / "client-sdks" / "stainless" / "openapi.yml")).resolve()
    stainless_config.validate(spec_path)
    yaml_text = yaml.safe_dump(stainless_config.to_dict(), sort_keys=False)
    output = repo_root / "client-sdks" / "stainless" / "config.yml"
    output.write_text(HEADER + yaml_text)
    return output


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    output = write_config(repo_root)
    print(f"Wrote Stainless config: {output}")


if __name__ == "__main__":
    main()
