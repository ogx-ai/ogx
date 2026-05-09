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

    // Prepare headers (exclude host and other problematic headers)
    const headers = new Headers();
    request.headers.forEach((value, key) => {
      // Skip headers that might cause issues in proxy
      if (
        !["host", "connection", "content-length"].includes(key.toLowerCase())
      ) {
        headers.set(key, value);
      }
    });

    // Prepare the request options
    const requestOptions: RequestInit = {
      method,
      headers,
    };

    // Add body for methods that support it
    if (["POST", "PUT", "PATCH"].includes(method) && request.body) {
      requestOptions.body = request.body;
      // Required for ReadableStream bodies in newer Node.js versions
      requestOptions.duplex = "half" as RequestDuplex;
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

    // Check content type to handle binary vs text responses appropriately
    const contentType = response.headers.get("content-type") || "";
    const isBinaryContent =
      contentType.includes("application/pdf") ||
      contentType.includes("application/msword") ||
      contentType.includes("application/vnd.openxmlformats-officedocument") ||
      contentType.includes("application/octet-stream") ||
      contentType.includes("image/") ||
      contentType.includes("video/") ||
      contentType.includes("audio/");

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
