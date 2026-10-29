# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for the `ogx connect` command group."""

import argparse

import pytest

from ogx.cli.connect.connect import ConnectParser


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

    def test_group_url_defaults_to_none(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(["connect", "claude"])
        assert args.connect_url is None

    def test_group_url_is_accepted_by_every_agent(self, connect_group: argparse.ArgumentParser) -> None:
        for agent in ("claude", "codex", "opencode"):
            args = connect_group.parse_args(["connect", "--url", "https://ogx.example.com/v1", agent])
            assert args.connect_url == "https://ogx.example.com/v1"

    def test_group_url_coexists_with_agent_url(self, connect_group: argparse.ArgumentParser) -> None:
        args = connect_group.parse_args(
            ["connect", "--url", "https://group.example.com/v1", "claude", "--url", "http://agent.example.com/v1"]
        )
        assert args.connect_url == "https://group.example.com/v1"
        assert args.url == "http://agent.example.com/v1"
