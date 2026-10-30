# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the `ogx connect` command group."""

import argparse

import pytest

from ogx.cli.connect.connect import ConnectParser, validate_base_url


@pytest.fixture
def connect_group() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="ogx")
    subparsers = top.add_subparsers()
    ConnectParser(subparsers)
    return top


class TestGroupUrl:
    def test_group_url_is_parsed(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(["connect", "--url", "https://ogx.example.com/v1", "claude"])
        assert args.connect_url == "https://ogx.example.com/v1"

    def test_group_url_beats_env(self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OGX_BASE_URL", "https://env.example.com/v1")
        args = connect_group.parse_args(["connect", "--url", "https://flag.example.com/v1", "claude"])
        assert args.connect_url == "https://flag.example.com/v1"

    def test_agent_level_url_flags_removed(self, connect_group: argparse.ArgumentParser) -> None:
        with pytest.raises(SystemExit):
            connect_group.parse_args(["connect", "claude", "--url", "http://agent.example.com/v1"])


class TestInsecureFlag:
    def test_defaults_to_false(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(["connect", "claude"])
        assert args.insecure is False

    def test_dash_k(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(["connect", "-k", "claude"])
        assert args.insecure is True

    def test_long_form(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(["connect", "--insecure", "claude"])
        assert args.insecure is True

    def test_wget_style_long_form(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(["connect", "--no-check-certificate", "claude"])
        assert args.insecure is True


class TestUrlResolution:
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OGX_BASE_URL", raising=False)
        monkeypatch.delenv("OGX_PORT", raising=False)

    def test_default(self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        args = connect_group.parse_args(["connect", "claude"])
        assert args.connect_url == "http://localhost:8321/v1"

    def test_port_from_env(self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("OGX_PORT", "9999")
        args = connect_group.parse_args(["connect", "claude"])
        assert args.connect_url == "http://localhost:9999/v1"

    def test_base_url_from_env(self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("OGX_BASE_URL", "https://ogx.example.com/custom/v1")
        args = connect_group.parse_args(["connect", "claude"])
        assert args.connect_url == "https://ogx.example.com/custom/v1"

    def test_base_url_env_beats_port_env(
        self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("OGX_PORT", "9999")
        monkeypatch.setenv("OGX_BASE_URL", "https://ogx.example.com/custom/v1")
        args = connect_group.parse_args(["connect", "claude"])
        assert args.connect_url == "https://ogx.example.com/custom/v1"

    def test_resolved_for_every_agent(
        self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        for agent in ("claude", "codex", "opencode"):
            args = connect_group.parse_args(["connect", agent])
            assert args.connect_url == "http://localhost:8321/v1"

    def test_not_resolved_without_subcommand(
        self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        args = connect_group.parse_args(["connect"])
        assert args.connect_url is None

    def test_invalid_url_exits(self, connect_group: argparse.ArgumentParser) -> None:
        with pytest.raises(SystemExit):
            connect_group.parse_args(["connect", "--url", "https://ogx.example.com", "claude"])

    def test_invalid_base_url_env_exits(
        self, connect_group: argparse.ArgumentParser, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clean_env(monkeypatch)
        monkeypatch.setenv("OGX_BASE_URL", "nope")
        with pytest.raises(SystemExit):
            connect_group.parse_args(["connect", "claude"])


class TestValidateBaseUrl:
    def test_preserves_valid_url(self) -> None:
        assert validate_base_url("https://ogx.example.com/v1") == "https://ogx.example.com/v1"

    def test_preserves_explicit_path(self) -> None:
        assert validate_base_url("https://ogx.example.com/prefix/v1") == "https://ogx.example.com/prefix/v1"

    def test_strips_whitespace_and_trailing_slash(self) -> None:
        assert validate_base_url("  https://ogx.example.com/v1/  ") == "https://ogx.example.com/v1"

    def test_exits_when_path_missing(self) -> None:
        with pytest.raises(SystemExit):
            validate_base_url("https://ogx.example.com")

    def test_exits_when_path_does_not_include_v1(self) -> None:
        with pytest.raises(SystemExit):
            validate_base_url("https://ogx.example.com/api")

    def test_exits_when_base_url_invalid(self) -> None:
        with pytest.raises(SystemExit):
            validate_base_url("localhost:8321")

    def test_exits_when_port_is_not_a_number(self) -> None:
        with pytest.raises(SystemExit):
            validate_base_url("http://localhost:abc/v1")
