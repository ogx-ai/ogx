#!/usr/bin/env python
# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import subprocess
import sys
from pathlib import Path
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from rich.progress import Progress, SpinnerColumn, TextColumn

from ogx.core.distribution import get_provider_registry

REPO_ROOT = Path(__file__).parent.parent


def get_api_docstring(api_name: str) -> str | None:
    """Extract docstring from the API protocol class."""
    try:
        # Import the API module dynamically
        api_module = __import__(f"ogx_api.{api_name}", fromlist=[api_name.title()])

        # Get the main protocol class (usually capitalized API name)
        protocol_class_name = api_name.title()
        if hasattr(api_module, protocol_class_name):
            protocol_class = getattr(api_module, protocol_class_name)
            return inspect.cleandoc(protocol_class.__doc__) if protocol_class.__doc__ else None
    except (ImportError, AttributeError):
        pass

    return None


class ChangedPathTracker:
    """Track a list of paths we may have changed."""

    def __init__(self):
        self._changed_paths = []

    def add_paths(self, *paths):
        for path in paths:
            path = str(path)
            if path not in self._changed_paths:
                self._changed_paths.append(path)

    def changed_paths(self):
        return self._changed_paths


def is_pydantic_model(annotation: Any) -> bool:
    """Check if an annotation is a Pydantic BaseModel."""
    if not hasattr(annotation, "__bases__"):
        return False
    return BaseModel in annotation.__mro__


def get_nested_type_fields(type_class: Any, visited: set[str] | None = None) -> dict[str, dict[str, Any]] | None:
    """
    Get fields from a Pydantic BaseModel type for expansion in documentation.

    Returns None if the type is not a Pydantic model or if we've already visited it (to avoid cycles).
    """
    if visited is None:
        visited = set()

    if not is_pydantic_model(type_class):
        return None

    type_name = getattr(type_class, "__name__", str(type_class))
    if type_name in visited:
        return None  # Avoid infinite recursion

    visited.add(type_name)

    if not hasattr(type_class, "model_fields"):
        return None

    fields_info = {}
    for field_name, field in type_class.model_fields.items():
        if getattr(field, "exclude", False):
            continue

        # Extract the actual type class for potential expansion
        field_annotation = field.annotation
        origin = get_origin(field_annotation)
        args = get_args(field_annotation)

        # Handle Annotated types first
        if origin is Annotated and args:
            field_annotation = args[0]
            origin = get_origin(field_annotation)
            args = get_args(field_annotation)

        # Handle Union types to find the non-None type
        # Prefer Pydantic models over other types for expansion
        type_class_for_expansion = None
        if origin in [Union, UnionType]:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                # Prefer Pydantic models for expansion
                for arg in non_none_args:
                    if is_pydantic_model(arg):
                        type_class_for_expansion = arg
                        break
                if not type_class_for_expansion:
                    # Fall back to first non-None type
                    type_class_for_expansion = non_none_args[0]
        else:
            type_class_for_expansion = field_annotation

        field_type = extract_type_annotation(field.annotation)
        default_value = field.default
        if field.default_factory is not None:
            try:
                default_value = field.default_factory()
            except Exception:
                default_value = None
        elif field.default is None or field.default is PydanticUndefined:
            default_value = None

        field_info = {
            "type": field_type,
            "type_class": type_class_for_expansion,  # Store the type class for expansion (preferring Pydantic models)
            "description": field.description or "",
            "default": default_value,
            "required": field.default is None and not field.is_required,
        }

        display_name = field.alias if field.alias else field_name
        fields_info[display_name] = field_info

    return fields_info


def extract_type_annotation(annotation: Any, expand_models: bool = False) -> str:
    """extract a type annotation into a clean string representation."""
    if annotation is None:
        return "Any"

    if annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # recursive workaround for Annotated types to ignore FieldInfo part
    if origin is Annotated and args:
        return extract_type_annotation(args[0], expand_models=expand_models)

    if origin in [Union, UnionType]:
        non_none_args = [arg for arg in args if arg is not type(None)]
        has_none = len(non_none_args) < len(args)

        if len(non_none_args) == 1:
            formatted = extract_type_annotation(non_none_args[0], expand_models=expand_models)
            return f"{formatted} | None" if has_none else formatted
        else:
            formatted_args = [extract_type_annotation(arg, expand_models=expand_models) for arg in non_none_args]
            result = " | ".join(formatted_args)
            return f"{result} | None" if has_none else result

    if origin is not None and args:
        origin_name = getattr(origin, "__name__", str(origin))
        formatted_args = [extract_type_annotation(arg, expand_models=expand_models) for arg in args]
        return f"{origin_name}[{', '.join(formatted_args)}]"

    # Check if this is a Pydantic model that should be expanded
    if expand_models and is_pydantic_model(annotation):
        return annotation.__name__

    return annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)


def get_config_class_info(config_class_path: str) -> dict[str, Any]:
    """Extract configuration information from a config class."""
    try:
        module_path, class_name = config_class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        config_class = getattr(module, class_name)

        docstring = config_class.__doc__ or ""

        accepts_extra_config = False
        try:
            schema = config_class.model_json_schema()
            if schema.get("additionalProperties") is True:
                accepts_extra_config = True
        except Exception:
            if hasattr(config_class, "model_config"):
                model_config = config_class.model_config
                if hasattr(model_config, "extra") and model_config.extra == "allow":
                    accepts_extra_config = True
                elif isinstance(model_config, dict) and model_config.get("extra") == "allow":
                    accepts_extra_config = True

        fields_info = {}
        if hasattr(config_class, "model_fields"):
            for field_name, field in config_class.model_fields.items():
                if getattr(field, "exclude", False):
                    continue

                # Extract the actual type class for potential expansion
                field_annotation = field.annotation
                origin = get_origin(field_annotation)
                args = get_args(field_annotation)

                # Handle Annotated types first
                if origin is Annotated and args:
                    field_annotation = args[0]
                    origin = get_origin(field_annotation)
                    args = get_args(field_annotation)

                # Handle Union types to find the non-None type
                # Prefer Pydantic models over other types for expansion
                type_class_for_expansion = None
                if origin in [Union, UnionType]:
                    non_none_args = [arg for arg in args if arg is not type(None)]
                    if non_none_args:
                        # Prefer Pydantic models for expansion
                        for arg in non_none_args:
                            if is_pydantic_model(arg):
                                type_class_for_expansion = arg
                                break
                        if not type_class_for_expansion:
                            # Fall back to first non-None type
                            type_class_for_expansion = non_none_args[0]
                else:
                    type_class_for_expansion = field_annotation

                field_type = extract_type_annotation(field.annotation)

                default_value = field.default
                if field.default_factory is not None:
                    try:
                        default_value = field.default_factory()
                        # HACK ALERT:
                        # If the default value contains a path that looks like it came from RUNTIME_BASE_DIR,
                        # replace it with a generic ~/.llama/ path for documentation
                        if isinstance(default_value, str) and "/.llama/" in default_value:
                            if ".llama/" in default_value:
                                path_part = default_value.split(".llama/")[-1]
                                default_value = f"~/.llama/{path_part}"
                    except Exception:
                        default_value = ""
                elif field.default is None or field.default is PydanticUndefined:
                    default_value = ""

                field_info = {
                    "type": field_type,
                    "type_class": type_class_for_expansion,  # Store the type class for expansion (preferring Pydantic models)
                    "description": field.description or "",
                    "default": default_value,
                    "required": field.default is None and not field.is_required,
                }

                # Use alias if available, otherwise use the field name
                display_name = field.alias if field.alias else field_name
                fields_info[display_name] = field_info

        if accepts_extra_config:
            config_description = "Additional configuration options that will be forwarded to the underlying provider"
            try:
                import inspect

                source = inspect.getsource(config_class)
                lines = source.split("\n")

                for i, line in enumerate(lines):
                    if "model_config" in line and "ConfigDict" in line and 'extra="allow"' in line:
                        comments = []
                        for j in range(i - 1, -1, -1):
                            stripped = lines[j].strip()
                            if stripped.startswith("#"):
                                comments.append(stripped[1:].strip())
                            elif stripped == "":
                                continue
                            else:
                                break

                        if comments:
                            config_description = " ".join(reversed(comments))
                        break
            except Exception:
                pass

            fields_info["config"] = {
                "type": "dict",
                "description": config_description,
                "default": "{}",
                "required": False,
            }

        return {
            "docstring": docstring,
            "fields": fields_info,
            "sample_config": getattr(config_class, "sample_run_config", None),
            "accepts_extra_config": accepts_extra_config,
        }
    except Exception as e:
        return {
            "error": f"Failed to load config class {config_class_path}: {str(e)}",
            "docstring": "",
            "fields": {},
            "sample_config": None,
            "accepts_extra_config": False,
        }


def generate_provider_docs(progress, provider_spec: Any, api_name: str) -> str:
    """Generate MDX documentation for a provider."""
    provider_type = provider_spec.provider_type
    config_class = provider_spec.config_class

    config_info = get_config_class_info(config_class)
    if "error" in config_info:
        progress.print(config_info["error"])

    # Extract description for frontmatter
    description = ""
    if hasattr(provider_spec, "description") and provider_spec.description:
        description = provider_spec.description
    elif (
        hasattr(provider_spec, "adapter")
        and hasattr(provider_spec.adapter, "description")
        and provider_spec.adapter.description
    ):
        description = provider_spec.adapter.description
    elif config_info.get("docstring"):
        description = config_info["docstring"]

    # Create sidebar label (clean up provider_type for display)
    sidebar_label = provider_type.replace("::", " - ").replace("_", " ")
    if sidebar_label.startswith("inline - "):
        sidebar_label = sidebar_label[9:].title()  # Remove "inline - " prefix and title case
    else:
        sidebar_label = sidebar_label.title()

    md_lines = []

    # Add YAML frontmatter
    md_lines.append("---")
    if description:
        # Handle multi-line descriptions in YAML - keep it simple for single line
        if "\n" in description.strip():
            md_lines.append("description: |")
            for line in description.strip().split("\n"):
                # Avoid trailing whitespace by only adding spaces to non-empty lines
                md_lines.append(f"  {line}" if line.strip() else "")
        else:
            # For single line descriptions, format properly for YAML
            clean_desc = description.strip().replace('"', '\\"')
            md_lines.append(f'description: "{clean_desc}"')
    md_lines.append(f"sidebar_label: {sidebar_label}")
    md_lines.append(f"title: {provider_type}")
    md_lines.append("---")
    md_lines.append("")

    # Add main title
    md_lines.append(f"# {provider_type}")
    md_lines.append("")

    if description:
        md_lines.append("## Description")
        md_lines.append("")
        md_lines.append(description)
        md_lines.append("")

    if config_info.get("fields"):
        md_lines.append("## Configuration")
        md_lines.append("")
        md_lines.append("| Field | Type | Required | Default | Description |")
        md_lines.append("|-------|------|----------|---------|-------------|")

        for field_name, field_info in config_info["fields"].items():
            field_type = field_info["type"].replace("|", "\\|")
            required = "Yes" if field_info["required"] else "No"
            default = str(field_info["default"]) if field_info["default"] is not None else ""

            # Handle multiline default values and escape problematic characters for MDX
            if "\n" in default:
                # For multiline defaults, escape angle brackets, curly braces, and use <br/> for line breaks
                lines = default.split("\n")
                escaped_lines = []
                for line in lines:
                    if line.strip():
                        # Escape all special characters that MDX might interpret
                        escaped_line = (
                            line.strip()
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                            .replace("{", "&#123;")
                            .replace("}", "&#125;")
                        )
                        escaped_lines.append(escaped_line)
                    else:
                        escaped_lines.append("")
                default = "<br/>".join(escaped_lines)
            else:
                # For single line defaults, escape all special characters
                default = (
                    default.replace("<", "&lt;").replace(">", "&gt;").replace("{", "&#123;").replace("}", "&#125;")
                )

            description_text = field_info["description"] or ""
            # Escape curly braces in description text for MDX compatibility
            description_text = description_text.replace("{", "&#123;").replace("}", "&#125;")

            md_lines.append(f"| `{field_name}` | `{field_type}` | {required} | {default} | {description_text} |")

            # Expand nested Pydantic models
            type_class = field_info.get("type_class")
            original_type_str = field_info["type"]
            # Check if the original type is a Union that includes non-model types
            # We check the type string for primitive types that might be in a Union with a Pydantic model
            is_union_with_other_types = False
            model_name = None

            # Check if the type string contains both a Pydantic model and primitive types
            if type_class and is_pydantic_model(type_class):
                model_name = type_class.__name__
                # Check if the type string contains both the model name and primitive types
                # This indicates a Union like "float | TimeoutConfig | None"
                primitive_types = ["float", "int", "str", "bool"]
                has_primitive = any(primitive in original_type_str for primitive in primitive_types)
                has_model = model_name in original_type_str
                has_union = "|" in original_type_str

                if has_union and has_model and has_primitive:
                    is_union_with_other_types = True

            # Handle Union types - extract Pydantic model if present
            if type_class:
                origin = get_origin(type_class)
                args = get_args(type_class) if origin else []
                if origin in [Union, UnionType]:
                    # Prefer Pydantic models for expansion
                    for arg in args:
                        if is_pydantic_model(arg):
                            type_class = arg
                            if not model_name:
                                model_name = type_class.__name__
                            break
                    else:
                        type_class = None  # No Pydantic model found in Union

            if type_class and is_pydantic_model(type_class):
                nested_fields = get_nested_type_fields(type_class, visited=set())
                if nested_fields:
                    # Add a note if this is a Union with other types (e.g., float | TimeoutConfig)
                    first_nested = is_union_with_other_types

                    for nested_field_name, nested_field_info in nested_fields.items():
                        nested_field_type = nested_field_info["type"].replace("|", "\\|")
                        nested_required = "Yes" if nested_field_info["required"] else "No"
                        nested_default = (
                            str(nested_field_info["default"]) if nested_field_info["default"] is not None else ""
                        )
                        # Escape special characters in nested default values
                        if nested_default:
                            nested_default = (
                                nested_default.replace("<", "&lt;")
                                .replace(">", "&gt;")
                                .replace("{", "&#123;")
                                .replace("}", "&#125;")
                            )
                        nested_description = nested_field_info["description"] or ""

                        # Add note for Union types that the expanded fields only apply when using the model
                        if is_union_with_other_types and first_nested and model_name:
                            nested_description = (
                                f"Only available when `{field_name}` is a `{model_name}` object. {nested_description}"
                            )
                            first_nested = False
                        elif is_union_with_other_types and first_nested:
                            # Fallback if model_name wasn't set
                            nested_description = f"Only available when `{field_name}` is an object (not a primitive type). {nested_description}"
                            first_nested = False

                        nested_description = nested_description.replace("{", "&#123;").replace("}", "&#125;")

                        # Format nested field name with dot notation
                        nested_field_display = f"`{field_name}.{nested_field_name}`"
                        md_lines.append(
                            f"| {nested_field_display} | `{nested_field_type}` | {nested_required} | {nested_default} | {nested_description} |"
                        )

                        # Recursively expand deeper nested models (e.g., network.tls, network.proxy, network.timeout)
                        nested_type_class = nested_field_info.get("type_class")
                        if nested_type_class:
                            # Handle Union types in nested fields - prefer Pydantic models
                            nested_origin = get_origin(nested_type_class)
                            nested_args = get_args(nested_type_class) if nested_origin else []
                            if nested_origin in [Union, UnionType]:
                                for arg in nested_args:
                                    if is_pydantic_model(arg):
                                        nested_type_class = arg
                                        break
                                else:
                                    nested_type_class = None

                            if nested_type_class and is_pydantic_model(nested_type_class):
                                deeper_fields = get_nested_type_fields(nested_type_class, visited=set())
                                if deeper_fields:
                                    for deeper_field_name, deeper_field_info in deeper_fields.items():
                                        deeper_field_type = deeper_field_info["type"].replace("|", "\\|")
                                        deeper_required = "Yes" if deeper_field_info["required"] else "No"
                                        deeper_default = (
                                            str(deeper_field_info["default"])
                                            if deeper_field_info["default"] is not None
                                            else ""
                                        )
                                        # Escape special characters in deeper default values
                                        if deeper_default:
                                            deeper_default = (
                                                deeper_default.replace("<", "&lt;")
                                                .replace(">", "&gt;")
                                                .replace("{", "&#123;")
                                                .replace("}", "&#125;")
                                            )
                                        deeper_description = deeper_field_info["description"] or ""
                                        deeper_description = deeper_description.replace("{", "&#123;").replace(
                                            "}", "&#125;"
                                        )

                                        deeper_field_display = f"`{field_name}.{nested_field_name}.{deeper_field_name}`"
                                        md_lines.append(
                                            f"| {deeper_field_display} | `{deeper_field_type}` | {deeper_required} | {deeper_default} | {deeper_description} |"
                                        )

        md_lines.append("")

        if config_info.get("accepts_extra_config"):
            md_lines.append(":::note")
            md_lines.append(
                "This configuration class accepts additional fields beyond those listed above. You can pass any additional configuration options that will be forwarded to the underlying provider."
            )
            md_lines.append(":::")
            md_lines.append("")

    if config_info.get("sample_config"):
        md_lines.append("## Sample Configuration")
        md_lines.append("")
        md_lines.append("```yaml")
        try:
            sample_config_func = config_info["sample_config"]
            import inspect

            import yaml

            if sample_config_func is not None:
                sig = inspect.signature(sample_config_func)
                if "__distro_dir__" in sig.parameters:
                    sample_config = sample_config_func(__distro_dir__="~/.llama/dummy")
                else:
                    sample_config = sample_config_func()

                def convert_pydantic_to_dict(obj):
                    if hasattr(obj, "model_dump"):
                        return obj.model_dump()
                    elif hasattr(obj, "dict"):
                        return obj.dict()
                    elif isinstance(obj, dict):
                        return {k: convert_pydantic_to_dict(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_pydantic_to_dict(item) for item in obj]
                    else:
                        return obj

                sample_config_dict = convert_pydantic_to_dict(sample_config)
                # Strip trailing newlines from yaml.dump to prevent extra blank lines
                yaml_output = yaml.dump(sample_config_dict, default_flow_style=False, sort_keys=False).rstrip()
                md_lines.append(yaml_output)
            else:
                md_lines.append("# No sample configuration available.")
        except Exception as e:
            md_lines.append(f"# Error generating sample config: {str(e)}")
        md_lines.append("```")

    if hasattr(provider_spec, "deprecation_warning") and provider_spec.deprecation_warning:
        md_lines.append("## Deprecation Notice")
        md_lines.append("")
        md_lines.append(":::warning")
        md_lines.append(provider_spec.deprecation_warning)
        md_lines.append(":::")

    if hasattr(provider_spec, "deprecation_error") and provider_spec.deprecation_error:
        md_lines.append("## Deprecation Error")
        md_lines.append("")
        md_lines.append(":::danger")
        md_lines.append(f"**Error**: {provider_spec.deprecation_error}")
        md_lines.append(":::")

    return "\n".join(md_lines) + "\n"


def generate_index_docs(api_name: str, api_docstring: str | None, provider_entries: list) -> str:
    """Generate MDX documentation for the index file."""
    # Create sidebar label for the API
    sidebar_label = api_name.replace("_", " ").title()

    md_lines = []

    # Add YAML frontmatter for index
    md_lines.append("---")
    if api_docstring:
        # Handle multi-line descriptions in YAML
        if "\n" in api_docstring.strip():
            md_lines.append("description: |")
            for line in api_docstring.strip().split("\n"):
                # Avoid trailing whitespace by only adding spaces to non-empty lines
                md_lines.append(f"  {line}" if line.strip() else "")
        else:
            # For single line descriptions, format properly for YAML
            clean_desc = api_docstring.strip().replace('"', '\\"')
            md_lines.append(f'description: "{clean_desc}"')
    md_lines.append(f"sidebar_label: {sidebar_label}")
    md_lines.append(f"title: {sidebar_label}")
    md_lines.append("---")
    md_lines.append("")

    # Add main content
    md_lines.append(f"# {sidebar_label}")
    md_lines.append("")
    md_lines.append("## Overview")
    md_lines.append("")

    if api_docstring:
        cleaned_docstring = api_docstring.strip()
        md_lines.append(f"{cleaned_docstring}")
        md_lines.append("")

    md_lines.append(f"This section contains documentation for all available providers for the **{api_name}** API.")

    return "\n".join(md_lines) + "\n"


def process_provider_registry(progress, change_tracker: ChangedPathTracker) -> None:
    """Process the complete provider registry."""
    progress.print("Processing provider registry")

    try:
        provider_registry = get_provider_registry()

        for api, providers in provider_registry.items():
            api_name = api.value

            doc_output_dir = REPO_ROOT / "docs" / "docs" / "providers" / api_name
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            change_tracker.add_paths(doc_output_dir)

            api_docstring = get_api_docstring(api_name)
            provider_entries = []

            for provider_type, provider in sorted(providers.items()):
                filename = provider_type.replace("::", "_").replace(":", "_")
                provider_doc_file = doc_output_dir / f"{filename}.mdx"

                provider_docs = generate_provider_docs(progress, provider, api_name)

                provider_doc_file.write_text(provider_docs)
                change_tracker.add_paths(provider_doc_file)

                # Create display name for the index
                display_name = provider_type.replace("::", " - ").replace("_", " ")
                if display_name.startswith("inline - "):
                    display_name = display_name[9:].title()
                else:
                    display_name = display_name.title()

                provider_entries.append({"filename": filename, "display_name": display_name})

            # Generate index file with frontmatter
            index_content = generate_index_docs(api_name, api_docstring, provider_entries)
            index_file = doc_output_dir / "index.mdx"
            index_file.write_text(index_content)
            change_tracker.add_paths(index_file)

    except Exception as e:
        progress.print(f"[red]Error processing provider registry: {str(e)}")
        raise e


def check_for_changes(change_tracker: ChangedPathTracker) -> bool:
    """Check if there are any uncommitted changes, including new files."""
    has_changes = False
    for path in change_tracker.changed_paths():
        result = subprocess.run(
            ["git", "diff", "--exit-code", path],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"Change detected in '{path}'.", file=sys.stderr)
            has_changes = True
        status_result = subprocess.run(
            ["git", "status", "--porcelain", path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        for line in status_result.stdout.splitlines():
            if line.startswith("??"):
                print(f"New file detected: '{path}'.", file=sys.stderr)
                has_changes = True
    return has_changes


def main():
    change_tracker = ChangedPathTracker()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task("Processing provider registry...", total=1)

        process_provider_registry(progress, change_tracker)
        progress.update(task, advance=1)

    if check_for_changes(change_tracker):
        print(
            "Provider documentation changes detected. Please commit the changes.",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
