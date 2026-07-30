# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from pydantic import BaseModel

from ogx_api.schema_utils import json_schema_type


class JobStatus(Enum):
    """Status of a job execution.
    :cvar completed: Job has finished successfully
    :cvar in_progress: Job is currently running
    :cvar failed: Job has failed during execution
    :cvar scheduled: Job is scheduled but not yet started
    :cvar cancelled: Job was cancelled before completion
    """

    completed = "completed"
    in_progress = "in_progress"
    failed = "failed"
    scheduled = "scheduled"
    cancelled = "cancelled"


@json_schema_type
class Job(BaseModel):
    """A job execution instance with status tracking.

    :param job_id: Unique identifier for the job
    :param status: Current execution status of the job
    """

    job_id: str
    status: JobStatus
