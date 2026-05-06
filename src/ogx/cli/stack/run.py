# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import ssl
import subprocess
import sys
from pathlib import Path

import uvicorn
import yaml
from termcolor import cprint

from ogx.cli.subcommand import Subcommand
from ogx.core.datatypes import StackConfig
from ogx.core.stack import cast_distro_name_to_string, replace_env_vars, run_config_from_dynamic_config_spec
from ogx.core.utils.config_dirs import DISTRIBS_BASE_DIR
from ogx.core.utils.config_resolution import resolve_config_or_distro
from ogx.log import get_logger

REPO_ROOT = Path(__file__).parent.parent.parent.parent

logger = get_logger(name=__name__, category="cli")


class StackRun(Subcommand):
    """CLI subcommand to start a OGX distribution server."""

    def __init__(self, subparsers: argparse._SubParsersAction) -> None:
        super().__init__()
        self.parser = subparsers.add_parser(
            "run",
            prog="ogx run",
            description="""Start the server for a OGX Distribution. You should have already built (or downloaded) and configured the distribution.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_run_cmd)

    def _add_arguments(self) -> None:
        self.parser.add_argument(
            "config",
            type=str,
            nargs="?",  # Make it optional
            metavar="config | distro",
            help="Path to config file to use for the run or name of known distro (`ogx list` for a list).",
        )
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
            "--providers",
            type=str,
            default=None,
            help="Run a stack with only a list of providers. This list is formatted like: api1=provider1,api1=provider2,api2=provider3. Where there can be multiple providers per API.",
        )

    def _run_stack_run_cmd(self, args: argparse.Namespace) -> None:
        import yaml

        from ogx.core.configure import parse_and_maybe_upgrade_config

        if args.enable_ui:
            self._start_ui_development_server(args.port)

        if args.config:
            try:
                from ogx.core.utils.config_resolution import resolve_config_or_distro

                config_file = resolve_config_or_distro(args.config)
            except ValueError as e:
                self.parser.error(str(e))
        elif args.providers:
            distro_dir = DISTRIBS_BASE_DIR / "providers-run"
            os.makedirs(distro_dir, exist_ok=True)
            try:
                run_config = run_config_from_dynamic_config_spec(
                    dynamic_config_spec=args.providers,
                    distro_dir=distro_dir,
                    distro_name="providers-run",
                )
            except ValueError as e:
                cprint(str(e), color="red", file=sys.stderr)
                sys.exit(1)
            config_dict = run_config.model_dump(mode="json")

            config_file = distro_dir / "config.yaml"
            logger.info("Writing generated config to", config_file=config_file)
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        else:
            config_file = None

        if config_file:
            logger.info("Using stack configuration", config_file=config_file)

            try:
                config_dict = yaml.safe_load(config_file.read_text())
            except yaml.parser.ParserError as e:
                self.parser.error(f"failed to load config file '{config_file}':\n {e}")

            try:
                config = parse_and_maybe_upgrade_config(config_dict)
                # Create external_providers_dir if it's specified and doesn't exist
                if config.external_providers_dir and not os.path.exists(str(config.external_providers_dir)):
                    os.makedirs(str(config.external_providers_dir), exist_ok=True)
            except AttributeError as e:
                self.parser.error(f"failed to parse config file '{config_file}':\n {e}")

        self._uvicorn_run(config_file, args)

    def _uvicorn_run(self, config_file: Path | None, args: argparse.Namespace) -> None:
        if not config_file:
            self.parser.error("Config file is required")

        config_file = resolve_config_or_distro(str(config_file))
        with open(config_file) as fp:
            config_contents = yaml.safe_load(fp)
            config = StackConfig(**cast_distro_name_to_string(replace_env_vars(config_contents)))

        port = args.port or config.server.port
        workers = config.server.workers

        # Dual-stack binding (IPv4+IPv6) depends on worker count:
        # - workers>1: use "::" (creates single dual-stack socket)
        # - workers=1: use "" (creates separate IPv4 and IPv6 sockets)
        host = ""
        if config.server.host:
            host = config.server.host
        elif workers and workers > 1:
            host = "::"

        # Set the config file in environment so create_app can find it
        os.environ["OGX_CONFIG"] = str(config_file)

        # Let create_app() handle logging setup instead of passing config to uvicorn
        uvicorn_config = {
            "factory": True,
            "host": host,
            "port": port,
            "lifespan": "on",
            "log_level": logger.getEffectiveLevel(),
            "workers": workers,
        }

        keyfile = config.server.tls_keyfile
        certfile = config.server.tls_certfile
        if keyfile and certfile:
            uvicorn_config["ssl_keyfile"] = config.server.tls_keyfile
            uvicorn_config["ssl_certfile"] = config.server.tls_certfile
            if config.server.tls_cafile:
                uvicorn_config["ssl_ca_certs"] = config.server.tls_cafile
                uvicorn_config["ssl_cert_reqs"] = ssl.CERT_REQUIRED

            logger.info(
                "HTTPS enabled with certificates", keyfile=keyfile, certfile=certfile, cafile=config.server.tls_cafile
            )
        else:
            logger.info("HTTPS enabled with certificates", keyfile=keyfile, certfile=certfile)

        logger.info("Listening on", host=host, port=port)

        # We need to catch KeyboardInterrupt because uvicorn's signal handling
        # re-raises SIGINT signals using signal.raise_signal(), which Python
        # converts to KeyboardInterrupt. Without this catch, we'd get a confusing
        # stack trace when using Ctrl+C or kill -2 (SIGINT).
        # SIGTERM (kill -15) works fine without this because Python doesn't
        # have a default handler for it.
        #
        # Another approach would be to ignore SIGINT entirely - let uvicorn handle it through its own
        # signal handling but this is quite intrusive and not worth the effort.
        try:
            uvicorn.run("ogx.core.server.server:create_app", **uvicorn_config)  # type: ignore[arg-type]
        except OSError as e:
            if e.errno in (97, 99):  # Address family not supported / Cannot assign requested address
                logger.error(
                    f"Failed to bind to {host}:{port}. "
                    "If you're on an IPv4-only system, set 'server.host: \"0.0.0.0\"' in your config."
                )
                raise
            raise
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received interrupt signal, shutting down gracefully...")

    def _start_ui_development_server(self, stack_server_port: int) -> None:
        logger.info("Attempting to start UI development server...")
        # Check if npm is available
        npm_check = subprocess.run(["npm", "--version"], capture_output=True, text=True, check=False)
        if npm_check.returncode != 0:
            logger.warning(
                "npm command not found or not executable, UI development server will not be started",
                error=npm_check.stderr,
            )
            return

        ui_dir = REPO_ROOT / "ogx_ui"
        logs_dir = Path("~/.llama/ui/logs").expanduser()
        try:
            # Create logs directory if it doesn't exist
            logs_dir.mkdir(parents=True, exist_ok=True)

            ui_stdout_log_path = logs_dir / "stdout.log"
            ui_stderr_log_path = logs_dir / "stderr.log"

            # Open log files in append mode
            stdout_log_file = open(ui_stdout_log_path, "a")
            stderr_log_file = open(ui_stderr_log_path, "a")

            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_dir),
                stdout=stdout_log_file,
                stderr=stderr_log_file,
                env={**os.environ, "NEXT_PUBLIC_OGX_BASE_URL": f"http://localhost:{stack_server_port}"},
            )
            logger.info("UI development server process started", ui_dir=ui_dir, pid=process.pid)
            logger.info("UI server logs", stdout=ui_stdout_log_path, stderr=ui_stderr_log_path)
            logger.info("UI will be available", port=os.getenv("OGX_UI_PORT", 8322))

        except FileNotFoundError:
            logger.error(
                "Failed to start UI development server: 'npm' command not found. Make sure npm is installed and in your PATH."
            )
        except Exception as e:
            logger.error("Failed to start UI development server", ui_dir=ui_dir, error=str(e))
