# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import ssl
from unittest.mock import MagicMock, patch

from ogx.core.storage.datatypes import PostgresSqlStoreConfig
from ogx.core.storage.sqlstore.sqlalchemy_sqlstore import SqlAlchemySqlStoreImpl


def _make_config(**kwargs) -> PostgresSqlStoreConfig:
    defaults = {
        "host": "localhost",
        "port": 5432,
        "db": "testdb",
        "user": "testuser",
        "password": "testpass",
    }
    defaults.update(kwargs)
    return PostgresSqlStoreConfig(**defaults)


class TestBuildSsl:
    def test_ssl_mode_none_returns_none(self):
        store = SqlAlchemySqlStoreImpl(_make_config())
        assert store._build_ssl() is None

    def test_ssl_mode_disable_returns_none(self):
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="disable"))
        assert store._build_ssl() is None

    def test_ssl_mode_require_returns_mode_string(self):
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="require"))
        assert store._build_ssl() == "require"

    def test_ssl_mode_prefer_returns_mode_string(self):
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="prefer"))
        assert store._build_ssl() == "prefer"

    def test_ssl_mode_verify_ca_without_ca_returns_mode_string(self):
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="verify-ca"))
        assert store._build_ssl() == "verify-ca"

    def test_ssl_mode_verify_ca_with_ca_returns_ssl_context_no_hostname_check(self, tmp_path):
        ca_file = tmp_path / "ca.pem"
        ca_file.write_text("")

        with patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.ssl.create_default_context") as mock_ctx:
            mock_context = MagicMock(spec=ssl.SSLContext)
            mock_ctx.return_value = mock_context
            store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="verify-ca", ca_cert_path=str(ca_file)))
            result = store._build_ssl()
            mock_ctx.assert_called_once_with(cafile=ca_file)
            assert mock_context.check_hostname is False
            assert result == mock_context

    def test_ssl_mode_verify_full_without_ca_returns_mode_string(self):
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="verify-full"))
        assert store._build_ssl() == "verify-full"

    def test_ssl_mode_verify_full_with_ca_returns_ssl_context(self, tmp_path):
        ca_file = tmp_path / "ca.pem"
        ca_file.write_text("")

        with patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.ssl.create_default_context") as mock_ctx:
            mock_ctx.return_value = MagicMock(spec=ssl.SSLContext)
            store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="verify-full", ca_cert_path=str(ca_file)))
            result = store._build_ssl()
            mock_ctx.assert_called_once_with(cafile=ca_file)
            assert result == mock_ctx.return_value


class TestCreateEngineWithSsl:
    @patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.create_async_engine")
    def test_no_ssl_in_connect_args_when_ssl_mode_none(self, mock_create_engine):
        mock_create_engine.return_value = MagicMock()
        store = SqlAlchemySqlStoreImpl(_make_config())
        store.create_engine()
        _, kwargs = mock_create_engine.call_args
        assert "ssl" not in kwargs.get("connect_args", {})

    @patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.create_async_engine")
    def test_no_ssl_in_connect_args_when_ssl_mode_disable(self, mock_create_engine):
        mock_create_engine.return_value = MagicMock()
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="disable"))
        store.create_engine()
        _, kwargs = mock_create_engine.call_args
        assert "ssl" not in kwargs.get("connect_args", {})

    @patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.create_async_engine")
    def test_ssl_in_connect_args_when_ssl_mode_require(self, mock_create_engine):
        mock_create_engine.return_value = MagicMock()
        store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="require"))
        store.create_engine()
        _, kwargs = mock_create_engine.call_args
        assert kwargs["connect_args"]["ssl"] == "require"

    @patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.create_async_engine")
    def test_ssl_context_in_connect_args_when_verify_full_with_ca(self, mock_create_engine, tmp_path):
        ca_file = tmp_path / "ca.pem"
        ca_file.write_text("")
        mock_create_engine.return_value = MagicMock()

        with patch("ogx.core.storage.sqlstore.sqlalchemy_sqlstore.ssl.create_default_context") as mock_ctx:
            mock_ctx.return_value = MagicMock(spec=ssl.SSLContext)
            store = SqlAlchemySqlStoreImpl(_make_config(ssl_mode="verify-full", ca_cert_path=str(ca_file)))
            store.create_engine()
            _, kwargs = mock_create_engine.call_args
            assert kwargs["connect_args"]["ssl"] == mock_ctx.return_value
