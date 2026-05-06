# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import enum
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin

import httpx
import yaml
from termcolor import cprint

from ogx.cli.stack.run import StackRun
from ogx.cli.subcommand import Subcommand
from ogx.core.build import get_provider_dependencies
from ogx.core.stack import run_config_from_dynamic_config_spec
from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR
from ogx.log import get_logger

logger = get_logger(name=__name__, category="cli")


class _ProbeStatus(enum.Enum):
    OK = "ok"
    NO_KEY = "no_key"
    AUTH = "auth"
    UNREACHABLE = "unreachable"


class StackLetsGo(Subcommand):
    """Auto-detect providers, generate runtime config, and start the stack.

    Providers fall into three categories:

    - Inline providers (files=inline::localfs, vector_io=inline::faiss,
      tool_runtime=inline::file-search, file_processors=inline::pypdf,
      responses=inline::builtin): require no external service and are always
      included.
    - Key-free providers (Ollama, vLLM, llama-cpp-server): included when a
      lightweight HTTP probe to their endpoint succeeds.
    - API-key providers (OpenAI, Anthropic, ...): included only when the
      required API key environment variable is set *and* the probe succeeds.
      A missing key is reported without making a network request.
    """

    def __init__(self, subparsers: Any) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "letsgo",
            prog="ogx letsgo",
            description="Auto-detect providers and start the stack",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_lets_go_cmd)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "--port",
            type=int,
            help="Port to run the server on. It can also be passed via the env var OGX_PORT.",
            default=int(os.getenv("OGX_PORT", 8321)),
        )
        self.parser.add_argument(
            "--enable-ui",
            action="store_true",
            help="Start the UI server",
        )
        self.parser.add_argument(
            "--persist-config",
            action="store_true",
            help="Persist generated runtime config to the distro directory",
        )
        self.parser.add_argument(
            "--providers-override",
            type=str,
            default=None,
            help="Explicit providers spec to use instead of auto-detection (e.g. inference=remote::ollama)",
        )
        self.parser.add_argument(
            "--skip-install-deps",
            action="store_true",
            help="Skip automatic installation of provider pip dependencies before starting the server.",
        )

    def _run_stack_lets_go_cmd(self, args: argparse.Namespace) -> None:
        # If user asked to start the UI, attempt to start it (best-effort)
        if args.enable_ui:
            try:
                stack_run = StackRun(argparse.ArgumentParser().add_subparsers())
                stack_run._start_ui_development_server(args.port)
            except Exception:
                # UI is best-effort; do not fail the whole command
                logger.warning("Failed to start UI development server", exc_info=True)

        # Determine providers spec (either overridden or auto-detected)
        if args.providers_override:
            providers_spec = args.providers_override
        else:
            providers_spec = self._autodetect_providers()

        has_inference = any(p.startswith("inference=") for p in (providers_spec or "").split(","))
        if not has_inference:
            self.parser.error("No inference providers detected. Nothing to run.")

        distro_dir = DISTRIBS_BASE_DIR / "letsgo-run" if args.persist_config else Path(tempfile.mkdtemp())
        os.makedirs(distro_dir, exist_ok=True)

        try:
            run_config = run_config_from_dynamic_config_spec(
                dynamic_config_spec=providers_spec,
                distro_dir=distro_dir,
                distro_name="letsgo-run",
            )
        except ValueError as e:
            cprint(str(e), color="red", file=sys.stderr)
            sys.exit(1)

        if not args.skip_install_deps:
            normal_deps, special_deps, _ = get_provider_dependencies(run_config)
            self._install_provider_deps(normal_deps, special_deps)

        config_dict = run_config.model_dump(mode="json")

        config_file = distro_dir / "config.yaml"
        logger.info("Writing generated config to", config_file=config_file)
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        # Reuse StackRun's uvicorn startup
        try:
            stack_run = StackRun(argparse.ArgumentParser().add_subparsers())
            # Build args similar to stack run
            stack_args = argparse.Namespace()
            stack_args.port = args.port
            stack_args.enable_ui = args.enable_ui
            stack_args.providers = None
            stack_run._uvicorn_run(config_file, stack_args)
        except Exception:
            logger.exception("Failed to start the stack server")
            raise

    def _install_provider_deps(self, normal_deps: list[str], special_deps: list[str]) -> None:
        """Install provider pip dependencies into the current environment.

        Uses `uv pip install` when uv is available, falling back to `pip install`.
        A non-zero exit is logged as a warning rather than aborting startup,
        since packages may already satisfy the declared constraints.
        """
        if shutil.which("uv"):
            installer = ["uv", "pip", "install"]
        else:
            installer = [sys.executable, "-m", "pip", "install"]

        if normal_deps:
            cprint("Installing provider dependencies...", color="cyan")
            result = subprocess.run([*installer, *normal_deps])
            if result.returncode != 0:
                logger.warning("Failed to install provider dependencies", returncode=result.returncode)

        for special_dep in special_deps:
            result = subprocess.run([*installer, *special_dep.split()])
            if result.returncode != 0:
                logger.warning(
                    "Failed to install special provider dependency", dep=special_dep, returncode=result.returncode
                )

    def _autodetect_providers(self) -> str:
        """Probe all candidate providers and return a comma-separated providers spec string.

        Each provider is probed independently; all that pass are included in the
        result. Providers that require an API key skip the network probe entirely
        when the key environment variable is not set.
        """
        candidates = [
            # provider_type, env_for_base_url, default_base_url, probe_path, requires_api_key, api_key_env, extra_headers
            ("remote::ollama", "OLLAMA_URL", "http://localhost:11434/v1", "models", False, None, {}),
            ("remote::vllm", "VLLM_URL", "http://localhost:8000/v1", "health", False, None, {}),
            ("remote::llama-cpp-server", "LLAMA_CPP_SERVER_URL", "http://localhost:8080/v1", "models", False, None, {}),
            ("remote::openai", "OPENAI_BASE_URL", "https://api.openai.com/v1", "models", True, "OPENAI_API_KEY", {}),
            (
                "remote::llama-openai-compat",
                "LLAMA_API_BASE_URL",
                "https://api.llama.com/compat/v1/",
                "models",
                True,
                "LLAMA_API_KEY",
                {},
            ),
            (
                "remote::anthropic",
                None,
                "https://api.anthropic.com/v1",
                "models",
                True,
                "ANTHROPIC_API_KEY",
                {"anthropic-version": "2023-06-01"},
            ),
        ]

        passed: list[str] = []
        cprint("Scanning for available providers...", color="cyan")
        for provider_type, base_env, default_base, probe_path, requires_key, key_env, extra_headers in candidates:
            env_val: str | None = os.getenv(base_env) if base_env else None
            if env_val:
                base = env_val
                base_source = f"from {base_env}"
            else:
                base = default_base
                base_source = "default"

            status = self._probe_endpoint(base, probe_path, requires_key, key_env, extra_headers)

            # Build annotation parts
            parts = [f"{base}, {base_source}"]
            if requires_key and key_env:
                parts.append(f"{key_env} {'set' if os.getenv(key_env) else 'not set'}")

            annotation = ", ".join(parts)

            if status == _ProbeStatus.OK:
                passed.append(f"inference={provider_type}")
                cprint(f"  ✓ {provider_type} ({annotation})", color="green")
            elif status == _ProbeStatus.NO_KEY:
                cprint(f"  ✗ {provider_type} ({annotation})", color="yellow")
            elif status == _ProbeStatus.AUTH:
                cprint(f"  ✗ {provider_type} ({annotation}) — auth error", color="yellow")
            else:
                cprint(f"  ✗ {provider_type} ({annotation}) — unreachable", color="yellow")

        # Inline providers require no external service — always include them.
        inline_providers = [
            "files=inline::localfs",
            "vector_io=inline::faiss",
            "tool_runtime=inline::file-search",
            "file_processors=inline::pypdf",
            "responses=inline::builtin",
        ]
        cprint("  ✓ inline::localfs (built-in)", color="green")
        cprint("  ✓ inline::faiss (built-in)", color="green")
        cprint("  ✓ inline::file-search (built-in)", color="green")
        cprint("  ✓ inline::pypdf (built-in)", color="green")
        cprint("  ✓ inline::builtin responses (built-in)", color="green")

        if passed:
            cprint(f"\nDetected {len(passed)} inference provider(s). Starting stack...", color="cyan")
        else:
            cprint("\nDetected no inference providers, not starting stack.", color="red")
        return ",".join(passed + inline_providers)

    def _probe_endpoint(
        self,
        base_url: str,
        probe_path: str,
        requires_key: bool,
        key_env: str | None,
        extra_headers: dict[str, str] | None = None,
    ) -> _ProbeStatus:
        """Perform a lightweight HTTP probe for a provider."""
        if not base_url:
            return _ProbeStatus.UNREACHABLE

        url = urljoin(base_url.rstrip("/") + "/", probe_path)

        headers: dict[str, str] = dict(extra_headers or {})
        if requires_key:
            if not key_env or not os.getenv(key_env):
                return _ProbeStatus.NO_KEY
            key: str = os.getenv(key_env, "")
            headers["Authorization"] = f"Bearer {key}"
            headers["x-api-key"] = key

        try:
            resp = cast(httpx.Response, httpx.get(url, headers=headers, timeout=2.0))
            if resp.status_code in (401, 403):
                return _ProbeStatus.AUTH
            if resp.status_code < 400:
                return _ProbeStatus.OK
            return _ProbeStatus.UNREACHABLE
        except Exception:
            return _ProbeStatus.UNREACHABLE
