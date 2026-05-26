# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Unit tests for PostgresSqlStoreConfig.engine_str URL construction."""

import pytest
from pydantic import SecretStr
from sqlalchemy.engine import URL

from ogx.core.storage.datatypes import PostgresSqlStoreConfig


class TestSqlPostgresEngineStr:
    def test_simple_credentials(self):
        config = PostgresSqlStoreConfig(
            user="pguser", password=SecretStr("simple"), host="db.local", port=5432, db="mydb"
        )
        url = config.engine_str
        assert isinstance(url, URL)
        assert url.drivername == "postgresql+asyncpg"
        assert url.username == "pguser"
        assert url.password == "simple"
        assert url.host == "db.local"
        assert url.port == 5432
        assert url.database == "mydb"

    def test_no_password(self):
        config = PostgresSqlStoreConfig(user="pguser", password=None, host="db.local", port=5432, db="mydb")
        url = config.engine_str
        assert url.username == "pguser"
        assert url.password is None
        assert url.host == "db.local"
        assert url.port == 5432
        assert url.database == "mydb"

    @pytest.mark.parametrize(
        "password",
        ["p@ss", "p:ss", "p/ss", "p%ss", "p@ss:w/o%rd", "a@b:c/d%e#f"],
    )
    def test_special_chars_in_password(self, password):
        config = PostgresSqlStoreConfig(user="u", password=SecretStr(password), host="h", port=5432, db="d")
        url = config.engine_str
        assert url.password == password
        assert url.username == "u"
        assert url.host == "h"
        assert url.port == 5432
        assert url.database == "d"

    @pytest.mark.parametrize(
        "username",
        ["u@ss", "u:ss", "u/ss", "u%ss", "u@ss:w/o%rd", "a@b:c/d%e#f"],
    )
    def test_special_chars_in_username(self, username):
        config = PostgresSqlStoreConfig(user=username, password=SecretStr("p"), host="h", port=5432, db="d")
        url = config.engine_str
        assert url.username == username
        assert url.password == "p"
        assert url.host == "h"
        assert url.port == 5432
        assert url.database == "d"

    @pytest.mark.parametrize(
        "db",
        ["my-db", "my.db", "db_name"],
    )
    def test_special_chars_in_db(self, db):
        config = PostgresSqlStoreConfig(user="u", password=SecretStr("p"), host="h", port=5432, db=db)
        url = config.engine_str
        assert url.username == "u"
        assert url.password == "p"
        assert url.host == "h"
        assert url.port == 5432
        assert url.database == db
