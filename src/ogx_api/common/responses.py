# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any

from pydantic import BaseModel

from ogx_api.schema_utils import json_schema_type


class Order(Enum):
    """Sort order for paginated responses.
    :cvar asc: Ascending order
    :cvar desc: Descending order
    """

    asc = "asc"
    desc = "desc"


@json_schema_type
class PaginatedResponse(BaseModel):
    """A generic paginated response that follows a simple format.

    :param data: The list of items for the current page
    :param has_more: Whether there are more items available after this set
    :param url: The URL for accessing this list
    """

    data: list[dict[str, Any]]
    has_more: bool
    url: str | None = None


# This is a short term solution to allow inference API to return metrics
# The ideal way to do this is to have a way for all response types to include metrics
# and all metric events logged to the telemetry API to be included with the response
# To do this, we will need to augment all response types with a metrics field.
# We have hit a blocker from stainless SDK that prevents us from doing this.
# The blocker is that if we were to augment the response types that have a data field
# in them like so
# class ListModelsResponse(BaseModel):
# metrics: Optional[List[MetricEvent]] = None
# data: List[Models]
# ...
# The client SDK will need to access the data by using a .data field, which is not
# ergonomic. Stainless SDK does support unwrapping the response type, but it
# requires that the response type to only have a single field.

# We will need a way in the client SDK to signal that the metrics are needed
# and if they are needed, the client SDK has to return the full response type
# without unwrapping it.


@json_schema_type
class MetricInResponse(BaseModel):
    """A metric value included in API responses.
    :param metric: The name of the metric
    :param value: The numeric value of the metric
    :param unit: (Optional) The unit of measurement for the metric value
    """

    metric: str
    value: int | float
    unit: str | None = None


class MetricResponseMixin(BaseModel):
    """Mixin class for API responses that can include metrics.
    :param metrics: (Optional) List of metrics associated with the API response
    """

    metrics: list[MetricInResponse] | None = None
