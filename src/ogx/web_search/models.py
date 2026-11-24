# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, Field

# The Responses API's web_search_call output items already define the
# source model; providers build the same objects so results flow into the
# output items without conversion.
from ogx_api.openai_responses import WebSearchSource  # noqa: F401


class WebSearchResult(BaseModel):
    """The result of a web search invocation.

    `content` is the formatted result text fed back to the model. `sources`
    carries the result URLs for the response output items; `query` is the
    query that was searched.
    """

    content: str
    query: str | None = None
    sources: list[WebSearchSource] = Field(default_factory=list)
