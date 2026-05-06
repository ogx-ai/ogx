# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from starlette.testclient import TestClient

from ogx.core.server.fastapi_router_registry import build_fastapi_router
from ogx.core.server.server import global_exception_handler
from ogx.telemetry.constants import RESPONSES_PARAMETER_USAGE_TOTAL
from ogx_api import Api, Responses
from ogx_api.openai_responses import (
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseObject,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
)
from ogx_api.responses.models import (
    CreateResponseRequest,
    DeleteResponseRequest,
    ListResponseInputItemsRequest,
    ListResponsesRequest,
    RetrieveResponseRequest,
)


async def _collect_stream_events(app: FastAPI, model: str) -> list[object]:
    transport = httpx.ASGITransport(app=app)
    client = AsyncOpenAI(
        base_url="http://test/v1",
        api_key="test",
        http_client=httpx.AsyncClient(transport=transport, base_url="http://test"),
    )
    try:
        stream = await client.responses.create(input="hi", model=model, stream=True)
        return [event async for event in stream]
    finally:
        await client.close()


def test_openapi_create_response_advertises_json_and_sse_200():
    """Regression test for OpenAPI shape of POST /v1/responses.

    We expect:
    - 200 response (not 204)
    - both JSON and SSE content types documented
    """

    app = FastAPI()
    impl = AsyncMock(spec=Responses)
    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    schema = app.openapi()

    post = schema["paths"]["/v1/responses"]["post"]
    responses = post["responses"]
    assert "200" in responses
    assert "204" not in responses

    content = responses["200"]["content"]
    assert "application/json" in content
    assert "text/event-stream" in content

    assert content["application/json"]["schema"]["$ref"] == "#/components/schemas/OpenAIResponseObject"
    assert content["text/event-stream"]["schema"]["$ref"] == "#/components/schemas/OpenAIResponseObjectStream"


async def test_create_response_returns_sse_streaming_response_when_impl_streams():
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    async def _stream():
        yield {"type": "response.output_text.delta", "delta": "hello"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=True)
    response = await create(request)

    assert response.media_type == "text/event-stream"


def test_create_response_maps_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError to HTTP 400."""
    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    impl = AsyncMock(spec=Responses)
    impl.create_openai_response.side_effect = ValueError("not found")

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})

    assert resp.status_code == 400
    assert resp.json()["detail"] == "not found"


async def test_create_response_returns_json_for_non_streaming():
    """Test POST /v1/responses returns OpenAIResponseObject when stream=False."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_123",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=False)
    response = await create(request)

    assert not hasattr(response, "media_type")  # Not a StreamingResponse
    assert response.id == "resp_123"
    assert response.status == "completed"


async def test_sse_format_is_correct():
    """Test that streaming responses produce valid SSE format."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    async def _stream():
        yield {"type": "response.output_text.delta", "delta": "hello"}
        yield {"type": "response.completed"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=True)
    response = await create(request)

    # Collect SSE events from the body iterator
    events = []
    async for chunk in response.body_iterator:
        events.append(chunk)

    assert len(events) == 2
    # Verify SSE format: "data: {...}\n\n"
    assert events[0].startswith("data: ")
    assert events[0].endswith("\n\n")
    assert '"type": "response.output_text.delta"' in events[0]


async def test_sse_stream_keeps_provider_context():
    from ogx.core.request_headers import PROVIDER_DATA_VAR

    app = FastAPI()
    impl = AsyncMock(spec=Responses)
    provider_data = {"provider": "test"}

    async def _stream():
        yield {"provider_data": PROVIDER_DATA_VAR.get()}
        yield {"type": "response.completed"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    token = PROVIDER_DATA_VAR.set(provider_data)
    try:
        request = CreateResponseRequest(input="hi", model="test", stream=True)
        response = await create(request)
    finally:
        PROVIDER_DATA_VAR.reset(token)

    first_event = None
    async for chunk in response.body_iterator:
        first_event = chunk
        break

    assert first_event is not None
    assert '"provider_data": {"provider": "test"}' in first_event


async def test_sse_stream_reports_value_error_as_http_exception():
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    async def _stream():
        raise ValueError("not found")
        yield {"type": "response.completed"}

    impl.create_openai_response.return_value = _stream()

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    create = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "POST" in getattr(r, "methods", set())
    )

    request = CreateResponseRequest(input="hi", model="test", stream=True)
    response = await create(request)

    first_event = None
    async for chunk in response.body_iterator:
        first_event = chunk
        break

    assert first_event is not None
    assert '"type":"error"' in first_event
    assert '"code":"400"' in first_event
    assert '"message":"not found"' in first_event


async def test_openai_client_stream_reports_error_before_first_event():
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    async def _error_before_stream():
        raise ValueError("Model not found")
        yield  # make this an async generator  # noqa: E303

    async def _create_response(request: CreateResponseRequest):
        assert request.model == "stack-accepted-model"
        return _error_before_stream()

    impl.create_openai_response.side_effect = _create_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    events = await _collect_stream_events(app, "stack-accepted-model")
    assert len(events) == 1
    error_event = events[0]
    assert error_event.type == "error"
    assert error_event.code == "400"
    assert error_event.message == "Model not found"
    assert error_event.sequence_number == 1


async def test_openai_client_stream_reports_error_midstream():
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    async def _error_midstream():
        yield OpenAIResponseObjectStreamResponseOutputTextDelta(
            content_index=0,
            delta="hello",
            item_id="item_123",
            output_index=0,
            sequence_number=7,
        )
        raise ValueError("Model not found")

    async def _create_response(request: CreateResponseRequest):
        assert request.model == "stack-accepted-model"
        return _error_midstream()

    impl.create_openai_response.side_effect = _create_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    events = await _collect_stream_events(app, "stack-accepted-model")
    assert len(events) == 2
    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "hello"
    assert events[0].sequence_number == 7
    assert events[1].type == "error"
    assert events[1].code == "400"
    assert events[1].message == "Model not found"
    assert events[1].sequence_number == 8


async def test_get_response_returns_response_object():
    """Test GET /v1/responses/{response_id} returns OpenAIResponseObject."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_123",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.get_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    get_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}" and "GET" in getattr(r, "methods", set())
    )

    request = RetrieveResponseRequest(response_id="resp_123")
    response = await get_endpoint(request)

    assert response.id == "resp_123"
    assert response.status == "completed"
    impl.get_openai_response.assert_awaited_once()


def test_get_response_maps_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError on GET to HTTP 400."""
    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    impl = AsyncMock(spec=Responses)
    impl.get_openai_response.side_effect = ValueError("Response not found")

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/v1/responses/nonexistent")

    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


async def test_list_responses_returns_list():
    """Test GET /v1/responses returns ListOpenAIResponseObject."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = ListOpenAIResponseObject(
        object="list",
        data=[],
        has_more=False,
        first_id="resp_first",
        last_id="resp_last",
    )
    impl.list_openai_responses.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    list_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses" and "GET" in getattr(r, "methods", set())
    )

    request = ListResponsesRequest()
    response = await list_endpoint(request)

    assert response.object == "list"
    assert response.has_more is False
    impl.list_openai_responses.assert_awaited_once()


async def test_list_input_items_returns_items():
    """Test GET /v1/responses/{response_id}/input_items returns input items."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = ListOpenAIResponseInputItem(
        object="list",
        data=[],
    )
    impl.list_openai_response_input_items.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    list_endpoint = next(
        r.endpoint for r in router.routes if getattr(r, "path", None) == "/v1/responses/{response_id}/input_items"
    )

    request = ListResponseInputItemsRequest(response_id="resp_123")
    response = await list_endpoint(request)

    assert response.object == "list"
    impl.list_openai_response_input_items.assert_awaited_once()


async def test_delete_response_returns_confirmation():
    """Test DELETE /v1/responses/{response_id} returns deletion confirmation."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIDeleteResponseObject(
        id="resp_123",
        object="response",
        deleted=True,
    )
    impl.delete_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    delete_endpoint = next(
        r.endpoint
        for r in router.routes
        if getattr(r, "path", None) == "/v1/responses/{response_id}" and "DELETE" in getattr(r, "methods", set())
    )

    request = DeleteResponseRequest(response_id="resp_123")
    response = await delete_endpoint(request)

    assert response.id == "resp_123"
    assert response.deleted is True
    impl.delete_openai_response.assert_awaited_once()


def test_delete_response_maps_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError on DELETE to HTTP 400."""
    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    impl = AsyncMock(spec=Responses)
    impl.delete_openai_response.side_effect = ValueError("Response not found")

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.delete("/v1/responses/nonexistent")

    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


def test_openapi_create_response_advertises_form_urlencoded_request_body():
    """OpenAPI schema should advertise application/x-www-form-urlencoded as accepted request content type."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)
    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    schema = app.openapi()
    post = schema["paths"]["/v1/responses"]["post"]
    request_body_content = post["requestBody"]["content"]
    assert "application/x-www-form-urlencoded" in request_body_content


def test_create_response_accepts_form_urlencoded():
    """POST /v1/responses accepts application/x-www-form-urlencoded content type."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_form",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/responses",
        data={"input": "hi", "model": "test", "stream": "false"},
    )

    assert resp.status_code == 200
    assert resp.json()["id"] == "resp_form"
    # Verify the impl received the correct request
    impl.create_openai_response.assert_awaited_once()
    call_args = impl.create_openai_response.call_args[0][0]
    assert call_args.input == "hi"
    assert call_args.model == "test"
    assert call_args.stream is False


def test_create_response_form_urlencoded_with_json_encoded_complex_fields():
    """Form-urlencoded requests support JSON-encoded strings for complex fields like tools."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_complex",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    import json

    tools_json = json.dumps([{"type": "web_search_preview"}])
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/responses",
        data={
            "input": "search for cats",
            "model": "test",
            "stream": "false",
            "tools": tools_json,
            "temperature": "0.7",
        },
    )

    assert resp.status_code == 200
    call_args = impl.create_openai_response.call_args[0][0]
    assert call_args.input == "search for cats"
    assert call_args.temperature == 0.7
    assert call_args.tools is not None
    assert len(call_args.tools) == 1


def test_create_response_form_urlencoded_validation_error():
    """Form-urlencoded requests with invalid data return 422."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    # Missing required 'model' field
    resp = client.post(
        "/v1/responses",
        data={"input": "hi"},
    )

    assert resp.status_code == 422


def test_create_response_form_urlencoded_raw_wire_format():
    """POST /v1/responses parses raw key=value&key=value form-urlencoded body."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_raw",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/responses",
        content="input=hello+world&model=test-model&stream=false&temperature=0.5",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert resp.status_code == 200
    assert resp.json()["id"] == "resp_raw"
    call_args = impl.create_openai_response.call_args[0][0]
    assert call_args.input == "hello world"
    assert call_args.model == "test-model"
    assert call_args.stream is False
    assert call_args.temperature == 0.5


def test_create_response_form_urlencoded_repeated_keys_collected_as_list():
    """Repeated form keys are collected into a list (e.g. include=a&include=b)."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_multi",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/responses",
        content="input=hi&model=test&stream=false&include=file_search_call.results&include=reasoning.encrypted_content",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert resp.status_code == 200
    call_args = impl.create_openai_response.call_args[0][0]
    assert call_args.include is not None
    assert len(call_args.include) == 2


def test_create_response_form_urlencoded_with_charset():
    """Content-Type with charset parameter is handled correctly."""
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    expected_response = OpenAIResponseObject(
        id="resp_charset",
        created_at=1234567890,
        model="test-model",
        object="response",
        output=[],
        status="completed",
        store=True,
    )
    impl.create_openai_response.return_value = expected_response

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post(
        "/v1/responses",
        content="input=hi&model=test&stream=false",
        headers={"Content-Type": "application/x-www-form-urlencoded; charset=utf-8"},
    )

    assert resp.status_code == 200
    assert resp.json()["id"] == "resp_charset"


def test_request_validation_error_passes_through_route_class():
    """RequestValidationError must NOT be caught by _ExceptionTranslatingRoute.

    FastAPI has its own handler for RequestValidationError that returns a
    422 response with detailed validation errors.  The route class must let
    it pass through so that invalid request bodies (e.g. max_tool_calls=0)
    get a proper 422 instead of a generic 500.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Responses)

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)

    # max_tool_calls has ge=1 constraint — sending 0 triggers RequestValidationError
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False, "max_tool_calls": 0})

    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body
    # FastAPI returns a list of validation errors
    assert isinstance(body["detail"], list)
    assert any("max_tool_calls" in str(err) for err in body["detail"])


def test_exception_translating_route_converts_value_error_to_400():
    """_ExceptionTranslatingRoute converts ValueError to HTTP 400.

    ValueError (including pydantic.ValidationError) is translated to a
    400 response by the route class.  This test exercises the route class
    via TestClient (full ASGI stack) to verify end-to-end behavior.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Responses)
    impl.create_openai_response.side_effect = ValueError("bad input value")

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})

    assert resp.status_code == 400
    assert resp.headers["content-type"] == "application/json"
    assert resp.json()["detail"] == "bad input value"


def test_unknown_exception_propagates_to_global_handler():
    """Unknown exception types (e.g. RuntimeError) propagate past the route class.

    The route class only translates known types (ValueError, OGXError).
    Unknown exceptions are left for the server's global exception handler,
    which uses the full translate_exception pipeline from ogx.core.
    """
    app = FastAPI()
    app.add_exception_handler(Exception, global_exception_handler)
    impl = AsyncMock(spec=Responses)
    impl.create_openai_response.side_effect = RuntimeError("something broke")

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})

    assert resp.status_code == 500
    # Global handler returns {"error": {"message": ...}}
    assert "error" in resp.json()


def test_consecutive_value_errors_keep_connection_alive():
    """The route-level try/except converts exceptions to HTTPException before
    they reach ServerErrorMiddleware, which would otherwise re-raise and
    cause uvicorn to close the transport (TCP RST on Linux).  Two requests
    on the same TestClient verify the connection stays alive.
    """
    app = FastAPI()
    impl = AsyncMock(spec=Responses)
    impl.create_openai_response.side_effect = ValueError("bad request")

    router = build_fastapi_router(Api.responses, impl)
    assert router is not None
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)

    # First request — triggers the error path
    resp1 = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})
    assert resp1.status_code == 400

    # Second request on the same connection — must NOT get a connection error
    resp2 = client.post("/v1/responses", json={"input": "hi", "model": "test", "stream": False})
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "bad request"


def test_parameter_usage_records_only_explicitly_provided_params():
    """_record_parameter_usage increments a counter for each optional parameter
    that was explicitly provided in the request body (via model_fields_set),
    and ignores required fields (input, model) and default-valued fields.
    """
    import ogx.providers.inline.responses.builtin.impl as responses_mod
    from ogx.providers.inline.responses.builtin.impl import _record_parameter_usage

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    # Patch the module-level meter so our counter uses the in-memory reader
    original_meter = responses_mod._meter
    responses_mod._meter = provider.get_meter("test")
    responses_mod._parameter_usage_total = responses_mod._meter.create_counter(
        name=RESPONSES_PARAMETER_USAGE_TOTAL,
        description="test counter",
        unit="1",
    )

    try:
        # Only temperature and tools are explicitly provided; stream/store get defaults
        request = CreateResponseRequest(
            input="hello",
            model="test-model",
            temperature=0.7,
            tools=[],
        )
        _record_parameter_usage(request, operation="create_response")

        metrics_data = reader.get_metrics_data()
        resource_metrics = metrics_data.resource_metrics
        assert len(resource_metrics) > 0

        # Collect all data points as {(operation, parameter): sum}
        param_counts: dict[tuple[str, str], int] = {}
        for rm in resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == RESPONSES_PARAMETER_USAGE_TOTAL:
                        for dp in metric.data.data_points:
                            key = (dp.attributes["operation"], dp.attributes["parameter"])
                            param_counts[key] = dp.value

        # temperature and tools were explicitly set
        assert ("create_response", "temperature") in param_counts
        assert ("create_response", "tools") in param_counts
        # required fields should not appear
        assert ("create_response", "input") not in param_counts
        assert ("create_response", "model") not in param_counts
        # fields with defaults that were NOT explicitly provided should not appear
        assert ("create_response", "stream") not in param_counts
        assert ("create_response", "store") not in param_counts
    finally:
        responses_mod._meter = original_meter
        responses_mod._parameter_usage_total = original_meter.create_counter(
            name=RESPONSES_PARAMETER_USAGE_TOTAL,
            description="Tracks which optional parameters are explicitly provided in Responses API calls",
            unit="1",
        )


def test_parameter_usage_ignores_extra_keys():
    """_record_parameter_usage must not record arbitrary extra keys from the
    request body.  CreateResponseRequest uses ConfigDict(extra="allow"), so
    user-supplied extra keys end up in model_fields_set.  Without filtering,
    this would cause unbounded Prometheus label cardinality.
    """
    import ogx.providers.inline.responses.builtin.impl as responses_mod
    from ogx.providers.inline.responses.builtin.impl import _record_parameter_usage

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    original_meter = responses_mod._meter
    responses_mod._meter = provider.get_meter("test")
    responses_mod._parameter_usage_total = responses_mod._meter.create_counter(
        name=RESPONSES_PARAMETER_USAGE_TOTAL,
        description="test counter",
        unit="1",
    )

    try:
        # Include arbitrary extra keys that should NOT become metric labels
        request = CreateResponseRequest(
            input="hello",
            model="test-model",
            temperature=0.7,
            random_extra_key="foo",
            another_unknown="bar",
        )
        _record_parameter_usage(request, operation="create_response")

        metrics_data = reader.get_metrics_data()
        param_counts: dict[tuple[str, str], int] = {}
        for rm in metrics_data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    if metric.name == RESPONSES_PARAMETER_USAGE_TOTAL:
                        for dp in metric.data.data_points:
                            key = (dp.attributes["operation"], dp.attributes["parameter"])
                            param_counts[key] = dp.value

        # temperature is a declared field and was explicitly set
        assert ("create_response", "temperature") in param_counts
        # extra keys must NOT appear as metric labels
        assert ("create_response", "random_extra_key") not in param_counts
        assert ("create_response", "another_unknown") not in param_counts
    finally:
        responses_mod._meter = original_meter
        responses_mod._parameter_usage_total = original_meter.create_counter(
            name=RESPONSES_PARAMETER_USAGE_TOTAL,
            description="Tracks which optional parameters are explicitly provided in Responses API calls",
            unit="1",
        )
