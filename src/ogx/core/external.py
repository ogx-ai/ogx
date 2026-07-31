# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import yaml

from ogx.core.datatypes import StackConfig
from ogx.log import get_logger
from ogx_api import Api, ExternalApiSpec

logger = get_logger(name=__name__, category="core")


def load_external_apis(config: StackConfig | None) -> dict[Api, ExternalApiSpec]:
    """Load external API specifications from the configured directory.

    Args:
        config: StackRunConfig or BuildConfig containing the external APIs directory path

    Returns:
        A dictionary mapping API names to their specifications
    """
    if not config or not config.external_apis_dir:
        return {}

    external_apis_dir = config.external_apis_dir.expanduser().resolve()
    if not external_apis_dir.is_dir():
        logger.error("External APIs directory is not a directory", path=str(external_apis_dir))
        return {}

    logger.info("Loading external APIs", path=str(external_apis_dir))
    external_apis: dict[Api, ExternalApiSpec] = {}

    # Look for YAML files in the external APIs directory
    for yaml_path in external_apis_dir.glob("*.yaml"):
        try:
            with open(yaml_path) as f:
                spec_data = yaml.safe_load(f)

            spec = ExternalApiSpec(**spec_data)
            api = Api.add(spec.name)
            logger.info("Loaded external API spec", api_name=spec.name, path=str(yaml_path))
            external_apis[api] = spec
        except yaml.YAMLError as yaml_err:
            logger.error("Failed to parse YAML file", path=str(yaml_path), error=str(yaml_err))
            raise
        except Exception:
            logger.exception("Failed to load external API spec", path=str(yaml_path))
            raise

    return external_apis
