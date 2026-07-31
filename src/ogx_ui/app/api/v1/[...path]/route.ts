import { NextRequest, NextResponse } from "next/server";

// Get backend URL from environment variable or default to localhost for development
const BACKEND_URL =
  process.env.OGX_BACKEND_URL ||
  `http://localhost:${process.env.OGX_PORT || 8321}`;

async function proxyRequest(request: NextRequest, method: string) {
  try {
    // Extract the path from the request URL
    const url = new URL(request.url);
    const pathSegments = url.pathname.split("/");

    // Remove /api from the path to get the actual API path
    // /api/v1/models/list -> /v1/models/list
    const apiPath = pathSegments.slice(2).join("/"); // Remove 'api' segment
    const targetUrl = `${BACKEND_URL}/${apiPath}${url.search}`;

    console.log(`Proxying ${method} ${url.pathname} -> ${targetUrl}`);

    // Check if this is a multipart/form-data request (file uploads)
    const contentType = request.headers.get("content-type") || "";
    const isMultipart = contentType.includes("multipart/form-data");

    // Prepare headers (exclude host and other problematic headers)
    const headers = new Headers();
    request.headers.forEach((value, key) => {
      const lower = key.toLowerCase();
      // Skip headers that cause issues in proxy
      // Keep content-length for multipart uploads so the backend can parse them
      if (lower === "host" || lower === "connection") return;
      if (lower === "content-length" && !isMultipart) return;
      headers.set(key, value);
    });

    // Prepare the request options
    const requestOptions: RequestInit = {
      method,
      headers,
    };

    // Add body for methods that support it
    if (["POST", "PUT", "PATCH"].includes(method) && request.body) {
      if (isMultipart) {
        // Buffer multipart bodies so content-length is accurate
        const bodyBuffer = await request.arrayBuffer();
        requestOptions.body = bodyBuffer;
        headers.set("content-length", bodyBuffer.byteLength.toString());
      } else {
        requestOptions.body = request.body;
        // Required for ReadableStream bodies in newer Node.js versions
        requestOptions.duplex = "half" as RequestDuplex;
      }
    }

    // Make the request to FastAPI backend
    const response = await fetch(targetUrl, requestOptions);

    console.log(
      `Response from FastAPI: ${response.status} ${response.statusText}`
    );

    // Handle 204 No Content responses specially
    if (response.status === 204) {
      const proxyResponse = new NextResponse(null, { status: 204 });
      // Copy response headers (except problematic ones)
      response.headers.forEach((value, key) => {
        if (!["connection", "transfer-encoding"].includes(key.toLowerCase())) {
          proxyResponse.headers.set(key, value);
        }
      });
      return proxyResponse;
    }

    // Check response content type to handle different response types
    const responseContentType = response.headers.get("content-type") || "";

    // Handle SSE (Server-Sent Events) streaming responses
    const isStreamingResponse =
      responseContentType.includes("text/event-stream") ||
      responseContentType.includes("application/x-ndjson");

    if (isStreamingResponse && response.body) {
      // Pass through the stream directly without buffering
      const proxyResponse = new NextResponse(response.body, {
        status: response.status,
        statusText: response.statusText,
      });

      // Copy response headers
      response.headers.forEach((value, key) => {
        if (!["connection", "transfer-encoding"].includes(key.toLowerCase())) {
          proxyResponse.headers.set(key, value);
        }
      });

      return proxyResponse;
    }

    const isBinaryContent =
      responseContentType.includes("application/pdf") ||
      responseContentType.includes("application/msword") ||
      responseContentType.includes(
        "application/vnd.openxmlformats-officedocument"
      ) ||
      responseContentType.includes("application/octet-stream") ||
      responseContentType.includes("image/") ||
      responseContentType.includes("video/") ||
      responseContentType.includes("audio/");

    let responseData: string | ArrayBuffer;

    if (isBinaryContent) {
      // Handle binary content (PDFs, Word docs, images, etc.)
      responseData = await response.arrayBuffer();
    } else {
      // Handle text content (JSON, plain text, etc.)
      responseData = await response.text();
    }

    // Create response with same status and headers
    const proxyResponse = new NextResponse(responseData, {
      status: response.status,
      statusText: response.statusText,
    });

    // Copy response headers (except problematic ones)
    response.headers.forEach((value, key) => {
      if (!["connection", "transfer-encoding"].includes(key.toLowerCase())) {
        proxyResponse.headers.set(key, value);
      }
    });

    return proxyResponse;
  } catch (error) {
    console.error("Proxy request failed:", error);

    return NextResponse.json(
      {
        error: "Proxy request failed",
        message: error instanceof Error ? error.message : "Unknown error",
        backend_url: BACKEND_URL,
        timestamp: new Date().toISOString(),
      },
      { status: 500 }
    );
  }
}

// HTTP method handlers
export async function GET(request: NextRequest) {
  return proxyRequest(request, "GET");
}

export async function POST(request: NextRequest) {
  return proxyRequest(request, "POST");
}

export async function PUT(request: NextRequest) {
  return proxyRequest(request, "PUT");
}

export async function DELETE(request: NextRequest) {
  return proxyRequest(request, "DELETE");
}

export async function PATCH(request: NextRequest) {
  return proxyRequest(request, "PATCH");
}

export async function OPTIONS(request: NextRequest) {
  return proxyRequest(request, "OPTIONS");
}
