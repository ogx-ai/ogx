# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from ogx.providers.remote.inference.vertexai.config import (
    VertexAIConfig,
    VertexAIProviderDataValidator,
)


@pytest.fixture
def project_name() -> str:
    """Return a project value for parameterized tests."""
    return "test-project"


class TestVertexAIConfig:
    """Test VertexAIConfig initialization and field handling."""

    @pytest.mark.parametrize(
        "config_kwargs,expected_location,expected_auth_credential",
        [
            ({"project": "test-project"}, "global", None),
            (
                {
                    "project": "test-project",
                    "location": "us-central1",
                    "access_token": SecretStr("test-token"),
                },
                "us-central1",
                "test-token",
            ),
        ],
    )
    def test_config_field_population(
        self,
        config_kwargs,
        expected_location,
        expected_auth_credential,
    ):
        """Test that config field population."""
        config = VertexAIConfig(**config_kwargs)
        assert config.project == "test-project"
        assert config.location == expected_location
        assert (config.auth_credential.get_secret_value() if config.auth_credential is not None else None) == (
            expected_auth_credential
        )

    def test_config_explicit_overrides_env(self):
        """Test that explicit values override environment variables."""
        env_vars = {
            "VERTEX_AI_PROJECT": "env-project",
            "VERTEX_AI_LOCATION": "europe-west1",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = VertexAIConfig(
                project="explicit-project",
                location="us-west1",
            )
            assert config.project == "explicit-project"
            assert config.location == "us-west1"

    def test_access_token_alias_accepts_plain_string(self):
        """Test that access token alias accepts plain string."""
        config = VertexAIConfig.model_validate(
            {
                "project": "test-project",
                "access_token": "token-from-alias",
            }
        )
        assert config.auth_credential is not None
        assert config.auth_credential.get_secret_value() == "token-from-alias"

    def test_auth_credential_field_name_accepts_plain_string(self):
        """Test that auth credential field name accepts plain string."""
        config = VertexAIConfig.model_validate(
            {
                "project": "test-project",
                "auth_credential": "token-from-field-name",
            }
        )
        assert config.auth_credential is not None
        assert config.auth_credential.get_secret_value() == "token-from-field-name"

    @pytest.mark.parametrize(
        "sample_kwargs,expected_project,expected_location",
        [
            ({}, "${env.VERTEX_AI_PROJECT:=}", "${env.VERTEX_AI_LOCATION:=global}"),
            (
                {"project": "custom-project", "location": "custom-location"},
                "custom-project",
                "custom-location",
            ),
        ],
    )
    def test_sample_run_config(self, sample_kwargs, expected_project, expected_location):
        """Test that sample run config."""
        sample = VertexAIConfig.sample_run_config(**sample_kwargs)
        assert sample["project"] == expected_project
        assert sample["location"] == expected_location

    def test_config_missing_required_project(self):
        """Test that project field is required."""
        with pytest.raises(ValidationError):
            VertexAIConfig.model_validate({})

    def test_auth_credential_excluded_from_serialization(self, project_name):
        """Credential must not leak via model_dump()."""
        config = VertexAIConfig(
            project=project_name,
            access_token=SecretStr("secret-creds"),
        )
        dumped = config.model_dump(by_alias=True, exclude_unset=False)
        assert "access_token" not in dumped
        assert "auth_credential" not in dumped


class TestVertexAIProviderDataValidator:
    """Test VertexAIProviderDataValidator initialization."""

    @pytest.mark.parametrize(
        "validator_kwargs,expected_project,expected_location,expected_access_token",
        [
            ({}, None, None, None),
            (
                {
                    "vertex_project": "test-project",
                    "vertex_location": "us-central1",
                    "vertex_access_token": "test-token",
                },
                "test-project",
                "us-central1",
                "test-token",
            ),
            ({"vertex_project": "test-project"}, "test-project", None, None),
        ],
    )
    def test_validator_field_population(
        self,
        validator_kwargs,
        expected_project,
        expected_location,
        expected_access_token,
    ):
        """Test that validator field population."""
        validator = VertexAIProviderDataValidator(**validator_kwargs)
        assert validator.vertex_project == expected_project
        assert validator.vertex_location == expected_location
        assert (
            validator.vertex_access_token.get_secret_value() if validator.vertex_access_token else None
        ) == expected_access_token


class TestVertexAIConfigBackwardCompatibility:
    def test_access_token_alias_populates_auth_credential(self, project_name):
        """Test that access token alias populates auth credential."""
        config = VertexAIConfig(
            project=project_name,
            access_token=SecretStr("creds"),
        )
        assert config.project == project_name
        assert config.auth_credential is not None
        assert config.auth_credential.get_secret_value() == "creds"

    def test_auth_credential_field_name_still_supported(self, project_name):
        """Test that auth credential field name still supported."""
        config = VertexAIConfig.model_validate(
            {
                "project": project_name,
                "auth_credential": SecretStr("creds"),
            }
        )
        assert config.project == project_name
        assert config.auth_credential is not None
        assert config.auth_credential.get_secret_value() == "creds"
