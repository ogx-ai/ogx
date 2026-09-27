# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from ogx.core.admin import AdminImpl, AdminImplConfig
from ogx.core.inspect import DistributionInspectConfig, DistributionInspectImpl
from ogx.core.jobs.runtime import JobRuntime, register_job_runtime, reset_job_runtime
from ogx_api import HealthStatus


class _UnhealthyPool:
    is_healthy = False


@pytest.fixture
def unhealthy_job_runtime():
    register_job_runtime(JobRuntime(queue=object(), pool=_UnhealthyPool()))
    yield
    reset_job_runtime()


async def test_inspect_health_reports_unhealthy_worker_pool(unhealthy_job_runtime) -> None:
    impl = DistributionInspectImpl(DistributionInspectConfig.model_construct(config=None), {})

    assert (await impl.health()).status == HealthStatus.ERROR


async def test_admin_health_reports_unhealthy_worker_pool(unhealthy_job_runtime) -> None:
    impl = AdminImpl(AdminImplConfig.model_construct(config=None), {})

    assert (await impl.health()).status == HealthStatus.ERROR
