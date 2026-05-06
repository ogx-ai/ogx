# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import json
from enum import Enum
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType

from ogx.log import get_logger

log = get_logger(name=__name__, category="core")


def is_list_of_primitives(field_type: Any) -> bool:
    """Check if a field type is a List of primitive types."""
    origin = get_origin(field_type)
    if origin is list or origin is list:
        args = get_args(field_type)
        if len(args) == 1 and args[0] in (int, float, str, bool):
            return True
    return False


def is_basemodel_without_fields(typ: Any) -> bool:
    """Check if a type is a Pydantic BaseModel subclass with no defined fields.

    Args:
        typ: The type to check.

    Returns:
        True if typ is a BaseModel subclass with zero fields.
    """
    return inspect.isclass(typ) and issubclass(typ, BaseModel) and len(typ.__fields__) == 0  # type: ignore[arg-type]


def can_recurse(typ: Any) -> bool:
    """Check if a type is a Pydantic BaseModel subclass with fields that can be recursively prompted.

    Args:
        typ: The type to check.

    Returns:
        True if typ is a BaseModel subclass with one or more fields.
    """
    return inspect.isclass(typ) and issubclass(typ, BaseModel) and len(typ.__fields__) > 0  # type: ignore[arg-type]


def get_literal_values(field: FieldInfo) -> tuple[Any, ...] | None:
    """Extract literal values from a field if it's a Literal type."""
    if get_origin(field.annotation) is Literal:
        return get_args(field.annotation)
    return None


def is_optional(field_type: Any) -> bool:
    """Check if a field type is Optional."""
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def get_non_none_type(field_type: Any) -> Any:
    """Get the non-None type from an Optional type."""
    return next(arg for arg in get_args(field_type) if arg is not type(None))


def manually_validate_field(model: type[BaseModel], field_name: str, value: Any) -> Any:
    """Run Pydantic field validators manually on a single field value.

    Args:
        model: The Pydantic model class containing the validators.
        field_name: The name of the field to validate.
        value: The value to validate.

    Returns:
        The validated value.
    """
    validators = model.__pydantic_decorators__.field_validators
    for _name, validator in validators.items():
        if field_name in validator.info.fields:
            validator.func(value)

    return value


def is_discriminated_union(typ: Any) -> bool:
    """Check if a type or FieldInfo represents a discriminated union.

    Args:
        typ: A type hint or Pydantic FieldInfo to check.

    Returns:
        True if the type is a discriminated union.
    """
    if isinstance(typ, FieldInfo):
        return bool(typ.discriminator)
    else:
        if get_origin(typ) is not Annotated:
            return False
        args = get_args(typ)
        return len(args) >= 2 and bool(args[1].discriminator)


def prompt_for_discriminated_union(
    field_name: str,
    typ: Any,
    existing_value: BaseModel | None,
) -> BaseModel:
    """Interactively prompt the user to select and configure a discriminated union variant.

    Args:
        field_name: The name of the field being configured.
        typ: The type hint or FieldInfo for the discriminated union.
        existing_value: An existing value to use as default, or None.

    Returns:
        A configured instance of the selected union variant.
    """
    if isinstance(typ, FieldInfo):
        inner_type = typ.annotation
        discriminator = typ.discriminator
        default_value = typ.default
    else:
        args = get_args(typ)
        inner_type = args[0]
        discriminator = args[1].discriminator
        default_value = args[1].default

    union_types = get_args(inner_type)
    # Find the discriminator field in each union type
    type_map = {}
    for t in union_types:
        disc_field = t.__fields__[discriminator]
        literal_values = get_literal_values(disc_field)
        if literal_values:
            for value in literal_values:
                type_map[value] = t

    while True:
        prompt = f"Enter `{discriminator}` for {field_name} (options: {', '.join(type_map.keys())})"
        if default_value is not None:
            prompt += f" (default: {default_value})"

        discriminator_value = input(f"{prompt}: ")
        if discriminator_value == "" and default_value is not None:
            discriminator_value = default_value

        if discriminator_value in type_map:
            chosen_type = type_map[discriminator_value]
            log.info(f"\nConfiguring {chosen_type.__name__}:")

            if existing_value and (getattr(existing_value, str(discriminator)) != discriminator_value):
                existing_value = None

            sub_config = prompt_for_config(chosen_type, existing_value)
            # Set the discriminator field in the sub-config
            setattr(sub_config, str(discriminator), discriminator_value)
            return sub_config
        else:
            log.error(f"Invalid {discriminator}. Please try again.")


# This is somewhat elaborate, but does not purport to be comprehensive in any way.
# We should add handling for the most common cases to tide us over.
#
# doesn't support List[nested_class] yet or Dicts of any kind. needs a bunch of
# unit tests for coverage.
def prompt_for_config(config_type: type[BaseModel], existing_config: BaseModel | None = None) -> BaseModel:
    """
    Recursively prompt the user for configuration values based on a Pydantic BaseModel.

    Args:
        config_type: A Pydantic BaseModel class representing the configuration structure.

    Returns:
        An instance of the config_type with user-provided values.
    """
    config_data = {}

    for field_name, field in config_type.__fields__.items():  # type: ignore[attr-defined]
        field_type = field.annotation
        existing_value = getattr(existing_config, field_name) if existing_config else None
        if existing_value:
            default_value = existing_value
        else:
            default_value = field.default if not isinstance(field.default, PydanticUndefinedType) else None
        is_required = field.is_required

        # Skip fields with Literal type
        if get_origin(field_type) is Literal:
            continue

        # Skip fields with no type annotations
        if is_basemodel_without_fields(field_type):
            config_data[field_name] = field_type()
            continue

        if inspect.isclass(field_type) and issubclass(field_type, Enum):
            prompt = f"Choose {field_name} (options: {', '.join(e.name for e in field_type)}):"
            while True:
                # this branch does not handle existing and default values yet
                user_input = input(prompt + " ")
                try:
                    enum_value: Any = field_type[user_input]
                    validated_value = manually_validate_field(config_type, field, enum_value)
                    config_data[field_name] = validated_value
                    break
                except KeyError:
                    log.error(f"Invalid choice. Please choose from: {', '.join(e.name for e in field_type)}")
            continue

        if is_discriminated_union(field):
            config_data[field_name] = prompt_for_discriminated_union(field_name, field, existing_value)
            continue

        if is_optional(field_type) and can_recurse(get_non_none_type(field_type)):
            prompt = f"Do you want to configure {field_name}? (y/n): "
            if input(prompt).lower() == "n":
                config_data[field_name] = None
                continue
            nested_type = get_non_none_type(field_type)
            log.info(f"Entering sub-configuration for {field_name}:")
            config_data[field_name] = prompt_for_config(nested_type, existing_value)
        elif is_optional(field_type) and is_discriminated_union(get_non_none_type(field_type)):
            prompt = f"Do you want to configure {field_name}? (y/n): "
            if input(prompt).lower() == "n":
                config_data[field_name] = None
                continue
            nested_type = get_non_none_type(field_type)
            config_data[field_name] = prompt_for_discriminated_union(
                field_name,
                nested_type,
                existing_value,
            )
        elif can_recurse(field_type):
            log.info(f"\nEntering sub-configuration for {field_name}:")
            config_data[field_name] = prompt_for_config(
                field_type,
                existing_value,
            )
        else:
            prompt = f"Enter value for {field_name}"
            if existing_value is not None:
                prompt += f" (existing: {existing_value})"
            elif default_value is not None:
                prompt += f" (default: {default_value})"
            if is_optional(field_type):
                prompt += " (optional)"
            elif is_required:
                prompt += " (required)"
            prompt += ": "

            while True:
                user_input = input(prompt)
                if user_input == "":
                    if default_value is not None:
                        config_data[field_name] = default_value
                        break
                    elif is_optional(field_type) or not is_required:
                        config_data[field_name] = None
                        break
                    else:
                        log.error("This field is required. Please provide a value.")
                        continue
                else:
                    try:
                        # Handle Optional types
                        if is_optional(field_type):
                            if user_input.lower() == "none":
                                value: Any = None
                            else:
                                field_type = get_non_none_type(field_type)
                                value = user_input

                        # Handle List of primitives
                        elif is_list_of_primitives(field_type):
                            try:
                                value = json.loads(user_input)
                                if not isinstance(value, list):
                                    raise ValueError("Input must be a JSON-encoded list")
                                element_type = get_args(field_type)[0]
                                value = [element_type(item) for item in value]

                            except json.JSONDecodeError:
                                log.error('Invalid JSON. Please enter a valid JSON-encoded list e.g., ["foo","bar"]')
                                continue
                            except ValueError as e:
                                log.error(f"{str(e)}")
                                continue

                        elif get_origin(field_type) is dict:
                            try:
                                value = json.loads(user_input)
                                if not isinstance(value, dict):
                                    raise ValueError("Input must be a JSON-encoded dictionary")

                            except json.JSONDecodeError:
                                log.error("Invalid JSON. Please enter a valid JSON-encoded dict.")
                                continue

                        # Convert the input to the correct type
                        elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                            # For nested BaseModels, we assume a dictionary-like string input
                            import ast

                            value = field_type(**ast.literal_eval(user_input))
                        else:
                            value = field_type(user_input)

                    except ValueError:
                        log.error(f"Invalid input. Expected type: {getattr(field_type, '__name__', str(field_type))}")
                        continue

                try:
                    # Validate the field using our manual validation function
                    validated_value = manually_validate_field(config_type, field_name, value)
                    config_data[field_name] = validated_value
                    break
                except ValueError as e:
                    log.error(f"Validation error: {str(e)}")

    return config_type(**config_data)
