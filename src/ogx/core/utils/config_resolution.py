# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR
from ogx.log import get_logger

logger = get_logger(name=__name__, category="core")


DISTRO_DIR = Path(__file__).parent.parent.parent.parent / "ogx" / "distributions"


def resolve_config_or_distro(
    config_or_distro: str,
) -> Path:
    """
    Resolve a config/distro argument to a concrete config file path.

    Args:
        config_or_distro: User input (file path, distribution name, or built distribution)

    Returns:
        Path to the resolved config file

    Raises:
        ValueError: If resolution fails
    """

    # Strategy 1: Try as file path first
    config_path = Path(config_or_distro)
    if config_path.exists() and config_path.is_file():
        logger.debug("Using file path", config_path=config_path)
        return config_path.resolve()

    # Strategy 2: Try as distribution name (if no .yaml extension)
    if not config_or_distro.endswith(".yaml"):
        distro_config = _get_distro_config_path(config_or_distro)
        if distro_config.exists():
            logger.debug("Using distribution", distro_config=distro_config)
            return distro_config

    # Strategy 3: Try as distro config path (if no .yaml extension and contains a slash)
    # eg: starter::run-with-postgres-store.yaml
    # Use :: to avoid slash and confusion with a filesystem path
    if "::" in config_or_distro:
        distro_name, config_name = config_or_distro.split("::")
        distro_config = _get_distro_config_path(distro_name, config_name)
        if distro_config.exists():
            logger.info("Using distribution", distro_config=distro_config)
            return distro_config

    # Strategy 4: Try as built distribution name
    distrib_config = DISTRIBS_BASE_DIR / f"ogx-{config_or_distro}" / f"{config_or_distro}-config.yaml"
    if distrib_config.exists():
        logger.debug("Using built distribution", distrib_config=distrib_config)
        return distrib_config

    distrib_config = DISTRIBS_BASE_DIR / f"{config_or_distro}" / "config.yaml"
    if distrib_config.exists():
        logger.debug("Using built distribution", distrib_config=distrib_config)
        return distrib_config

    # Strategy 5: Failed - provide helpful error
    raise ValueError(_format_resolution_error(config_or_distro))


def _get_distro_config_path(distro_name: str, path: str | None = None) -> Path:
    """Get the config file path for a distro."""
    if not path or not path.endswith(".yaml"):
        path = "config.yaml"
    return DISTRO_DIR / distro_name / path


def _format_resolution_error(config_or_distro: str) -> str:
    """Format a helpful error message for resolution failures."""
    from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR

    distro_path = _get_distro_config_path(config_or_distro)
    distrib_path = DISTRIBS_BASE_DIR / f"ogx-{config_or_distro}" / f"{config_or_distro}-config.yaml"
    distrib_path2 = DISTRIBS_BASE_DIR / f"{config_or_distro}" / f"{config_or_distro}-config.yaml"

    available_distros = _get_available_distros()
    distros_str = ", ".join(available_distros) if available_distros else "none found"

    return f"""Could not resolve config or distribution '{config_or_distro}'.

Tried the following locations:
  1. As file path: {Path(config_or_distro).resolve()}
  2. As distribution: {distro_path}
  3. As built distribution: ({distrib_path}, {distrib_path2})

Available distributions: {distros_str}

Did you mean one of these distributions?
{_format_distro_suggestions(available_distros, config_or_distro)}
"""


def _get_available_distros() -> list[str]:
    """Get list of available distro names."""

    distros = []
    if DISTRO_DIR.exists():
        distros.extend([d.name for d in DISTRO_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if DISTRIBS_BASE_DIR.exists():
        distros.extend([d.name for d in DISTRIBS_BASE_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])

    return list(set(distros))


def _format_distro_suggestions(distros: list[str], user_input: str) -> str:
    """Format distro suggestions for error messages, showing closest matches first."""
    if not distros:
        return "  (no distros found)"

    import difflib

    # Get up to 3 closest matches with similarity threshold of 0.3 (lower = more permissive)
    close_matches = difflib.get_close_matches(user_input, distros, n=3, cutoff=0.3)
    display_distros = close_matches if close_matches else distros[:3]

    suggestions = [f"  - {d}" for d in display_distros]
    return "\n".join(suggestions)
