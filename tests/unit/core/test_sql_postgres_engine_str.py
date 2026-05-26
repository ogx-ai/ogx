# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for PostgresSqlStoreConfig.engine_str URL encoding."""

from urllib.parse import unquote, urlparse

import pytest
from pydantic import SecretStr

from ogx.core.storage.datatypes import PostgresSqlStoreConfig


class TestSqlPostgresEngineStr:
    def test_simple_password(self):
        config = PostgresSqlStoreConfig(
            user="pguser", password=SecretStr("simple"), host="db.local", port=5432, db="mydb"
        )
        assert config.engine_str == "postgresql+asyncpg://pguser:simple@db.local:5432/mydb"

    def test_password_with_at_sign(self):
        config = PostgresSqlStoreConfig(
            user="pguser", password=SecretStr("p@ss"), host="db.local", port=5432, db="mydb"
        )
        url = config.engine_str
        assert "db.local" in url
        assert url.startswith("postgresql+asyncpg://")
        parsed = urlparse(url)
        assert parsed.username == "pguser"
        assert unquote(parsed.password) == "p@ss"
        assert parsed.hostname == "db.local"
        assert parsed.port == 5432
        assert parsed.path == "/mydb"

    def test_password_with_special_characters(self):
        config = PostgresSqlStoreConfig(
            user="pguser", password=SecretStr("my@pass:word/foo%bar"), host="db.local", port=5432, db="mydb"
        )
        url = config.engine_str
        assert "db.local:5432/mydb" in url
        assert url.startswith("postgresql+asyncpg://pguser:")
        parsed = urlparse(url)
        assert parsed.username == "pguser"
        assert unquote(parsed.password) == "my@pass:word/foo%bar"
        assert parsed.hostname == "db.local"
        assert parsed.port == 5432
        assert parsed.path == "/mydb"

    def test_no_password(self):
        config = PostgresSqlStoreConfig(user="pguser", password=None, host="db.local", port=5432, db="mydb")
        assert config.engine_str == "postgresql+asyncpg://pguser@db.local:5432/mydb"
        parsed = urlparse(config.engine_str)
        assert parsed.username == "pguser"
        assert parsed.password is None
        assert parsed.hostname == "db.local"
        assert parsed.port == 5432
        assert parsed.path == "/mydb"

    @pytest.mark.parametrize(
        "password",
        ["p@ss", "p:ss", "p/ss", "p%ss", "p@ss:w/o%rd", "a@b:c/d%e#f"],
    )
    def test_special_chars_produce_valid_url(self, password):
        config = PostgresSqlStoreConfig(user="u", password=SecretStr(password), host="h", port=5432, db="d")
        url = config.engine_str
        assert url.startswith("postgresql+asyncpg://u:")
        assert "@h:5432/d" in url
        parsed = urlparse(url)
        assert parsed.username == "u"
        assert unquote(parsed.password) == password
        assert parsed.hostname == "h"
        assert parsed.port == 5432
        assert parsed.path == "/d"
