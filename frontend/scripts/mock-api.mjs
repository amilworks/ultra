import http from "node:http";

const port = Number(process.env.MOCK_API_PORT || "8000");
const guestCookieName = "bisque_ultra_session";
const bisqueRoot = "https://bisque.example.org";

const navLinks = {
  home: `${bisqueRoot}/client_service/`,
  datasets: `${bisqueRoot}/client_service/browser?resource=/data_service/dataset`,
  images: `${bisqueRoot}/client_service/browser?resource=/data_service/image`,
  tables: `${bisqueRoot}/client_service/browser?resource=/data_service/table`,
};

const sendJson = (response, statusCode, payload, headers = {}) => {
  response.writeHead(statusCode, {
    "Content-Type": "application/json",
    ...headers,
  });
  response.end(JSON.stringify(payload));
};

const browserLogoutRedirectUrl = (value) => {
  const candidate = String(value || "").trim();
  if (!candidate) {
    return "/";
  }
  try {
    const parsed = new URL(candidate, "http://localhost");
    if (
      ["localhost", "127.0.0.1", "[::1]"].includes(parsed.hostname) ||
      candidate.startsWith("/")
    ) {
      return candidate;
    }
  } catch {
    // Fall through to local app root.
  }
  return "/";
};

const readSessionMode = (request) =>
  String(request.headers.cookie || "")
    .split(";")
    .map((value) => value.trim())
    .find((value) => value.startsWith(`${guestCookieName}=`))
    ?.slice(guestCookieName.length + 1) || null;

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || "127.0.0.1"}`);

  if (request.method === "GET" && url.pathname === "/v1/health") {
    sendJson(response, 200, { status: "ok", ts: new Date().toISOString() });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/config/public") {
    sendJson(response, 200, {
      bisque_root: bisqueRoot,
      bisque_browser_url: navLinks.images,
      bisque_urls: navLinks,
      bisque_auth_enabled: true,
      bisque_guest_enabled: true,
      admin_enabled: false,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/auth/session") {
    const sessionMode = readSessionMode(request);
    if (!sessionMode) {
      sendJson(response, 200, { authenticated: false });
      return;
    }
    sendJson(response, 200, {
      authenticated: true,
      username: sessionMode === "guest" ? "Mobile Smoke" : "Mock BisQue User",
      bisque_root: bisqueRoot,
      expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
      mode: sessionMode === "guest" ? "guest" : "bisque",
      guest_profile:
        sessionMode === "guest"
          ? {
              name: "Mobile Smoke",
              email: "mobile.smoke@example.com",
              affiliation: "BisQue Ultra QA",
            }
          : null,
      is_admin: false,
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/auth/login") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    sendJson(
      response,
      200,
      {
        authenticated: true,
        username: String(payload.username || "Mock BisQue User"),
        bisque_root: bisqueRoot,
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        mode: "bisque",
        guest_profile: null,
        is_admin: false,
      },
      {
        "Set-Cookie": `${guestCookieName}=bisque; Path=/; SameSite=Lax`,
      }
    );
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/auth/logout") {
    sendJson(
      response,
      200,
      { authenticated: false },
      {
        "Set-Cookie": `${guestCookieName}=; Path=/; Max-Age=0; SameSite=Lax`,
      }
    );
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/auth/logout/browser") {
    response.writeHead(302, {
      Location: browserLogoutRedirectUrl(url.searchParams.get("next")),
      "Set-Cookie": `${guestCookieName}=; Path=/; Max-Age=0; SameSite=Lax`,
    });
    response.end();
    return;
  }

  if (request.method === "POST" && url.pathname === "/v1/auth/guest") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    sendJson(
      response,
      200,
      {
        authenticated: true,
        username: String(payload.name || "Mobile Smoke"),
        bisque_root: bisqueRoot,
        expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
        mode: "guest",
        guest_profile: {
          name: String(payload.name || "Mobile Smoke"),
          email: String(payload.email || "mobile.smoke@example.com"),
          affiliation: String(payload.affiliation || "BisQue Ultra QA"),
        },
        is_admin: false,
      },
      {
        "Set-Cookie": `${guestCookieName}=guest; Path=/; SameSite=Lax`,
      }
    );
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/resources") {
    sendJson(response, 200, { count: 0, resources: [] });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/conversations") {
    sendJson(response, 200, {
      count: 0,
      conversations: [],
      offset: 0,
      limit: 50,
      has_more: false,
      next_offset: null,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v1/fun/weather/santa-barbara") {
    sendJson(response, 200, {
      success: false,
      location: "Santa Barbara, CA",
      micro_location: "Campus Point",
      blip: "Weather is unavailable in smoke mode.",
      summary: "Weather is unavailable in smoke mode.",
      source: "mock",
    });
    return;
  }

  sendJson(response, 404, { detail: `Unhandled mock endpoint: ${request.method} ${url.pathname}` });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Mock API listening on http://127.0.0.1:${port}`);
});
