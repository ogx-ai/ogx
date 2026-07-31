import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL =
  process.env.OGX_BACKEND_URL ||
  `http://localhost:${process.env.OGX_PORT || 8321}`;

async function proxyRequest(request: NextRequest, method: string) {
  try {
    const url = new URL(request.url);
    const pathSegments = url.pathname.split("/");

    // Remove /api from the path to get the actual API path
    // /api/v1alpha/admin/health -> /v1alpha/admin/health
    const apiPath = pathSegments.slice(2).join("/");
    const targetUrl = `${BACKEND_URL}/${apiPath}${url.search}`;

    console.log(`Proxying ${method} ${url.pathname} -> ${targetUrl}`);

    const headers = new Headers();
    request.headers.forEach((value, key) => {
      if (
        !["host", "connection", "content-length"].includes(key.toLowerCase())
      ) {
        headers.set(key, value);
      }
    });

    const requestOptions: RequestInit = {
      method,
      headers,
    };

    if (["POST", "PUT", "PATCH"].includes(method) && request.body) {
      requestOptions.body = request.body;
      requestOptions.duplex = "half" as RequestDuplex;
    }

    const response = await fetch(targetUrl, requestOptions);

    console.log(
      `Response from FastAPI: ${response.status} ${response.statusText}`
    );

    if (response.status === 204) {
      const proxyResponse = new NextResponse(null, { status: 204 });
      response.headers.forEach((value, key) => {
        if (!["connection", "transfer-encoding"].includes(key.toLowerCase())) {
          proxyResponse.headers.set(key, value);
        }
      });
      return proxyResponse;
    }

    const responseData = await response.text();

    const proxyResponse = new NextResponse(responseData, {
      status: response.status,
      statusText: response.statusText,
    });

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
