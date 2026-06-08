import http from "node:http";

const port = Number(process.env.MOCK_API_PORT || "8000");
const guestCookieName = "bisque_ultra_session";
const bisqueRoot = "https://bisque2.ece.ucsb.edu";

const navLinks = {
  home: `${bisqueRoot}/client_service/`,
  datasets: `${bisqueRoot}/client_service/browser?resource=/data_service/dataset`,
  images: `${bisqueRoot}/client_service/browser?resource=/data_service/image`,
  tables: `${bisqueRoot}/client_service/browser?resource=/data_service/table`,
};
const nowIso = new Date("2026-06-07T12:00:00Z").toISOString();

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

const workosSession = () => ({
  authenticated: true,
  provider: "workos",
  mode: "workos",
  username: "mobile.smoke@example.com",
  user: {
    id: "user_mobile_smoke",
    email: "mobile.smoke@example.com",
    first_name: "Mobile",
    last_name: "Smoke",
  },
  bisque_root: bisqueRoot,
  bisque_linked: true,
  expires_at: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
  is_admin: false,
});

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
      auth_provider: "local",
      admin_enabled: false,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/config/public") {
    sendJson(response, 200, {
      bisque_root: bisqueRoot,
      bisque_browser_url: navLinks.images,
      bisque_urls: navLinks,
      bisque_auth_enabled: true,
      bisque_guest_enabled: false,
      auth_provider: "workos",
      admin_enabled: false,
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/auth/session") {
    sendJson(response, 200, workosSession());
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/auth/login") {
    sendJson(response, 200, {
      authenticated: false,
      provider: "workos",
      mode: "workos",
      authorization_url: "/",
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/auth/logout") {
    sendJson(response, 200, {
      authenticated: false,
      provider: "workos",
      mode: "workos",
      logout_url: "/",
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/auth/guest") {
    sendJson(response, 200, workosSession());
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/threads") {
    sendJson(response, 200, {
      threads: [],
      count: 0,
      total_count: 0,
      offset: Number(url.searchParams.get("offset") || "0"),
      limit: Number(url.searchParams.get("limit") || "50"),
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/resources") {
    sendJson(response, 200, { count: 0, resources: [] });
    return;
  }

  if (request.method === "POST" && url.pathname === "/v2/bisque/search") {
    let body = "";
    for await (const chunk of request) {
      body += chunk;
    }
    const payload = body ? JSON.parse(body) : {};
    const resourceType = String(payload.resource_type || "image").toLowerCase();
    const counts = { image: 142, dataset: 12, table: 8 };
    sendJson(response, 200, {
      count: counts[resourceType] ?? 0,
      results: [],
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/models") {
    sendJson(response, 200, {
      count: 1,
      models: [
        {
          key: "prairie_yolo",
          name: "Prairie YOLO",
          framework: "yolov5",
          task_type: "object_detection",
          description: "Mock prairie detection model for frontend performance tests.",
          supports_training: true,
          supports_finetune: true,
          supports_inference: true,
          dimensions: ["image"],
          default_config: {},
        },
      ],
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/prairie/status") {
    sendJson(response, 200, {
      dataset_name: "Prairie_Dog_Active_Learning",
      dataset_id: "dataset_prairie",
      last_sync_at: nowIso,
      next_sync_at: nowIso,
      active_model_version: "version_active",
      model_health: "healthy",
      reviewed_images: 24,
      unreviewed_images: 3,
      class_counts: { prairie_dog: 48, burrow: 19 },
      unsupported_class_counts: {},
      detection_counts: {},
      latest_metrics: { map50: 0.91 },
      benchmark_baseline: {},
      benchmark_latest_candidate: {},
      last_benchmark_at: nowIso,
      benchmark_ready: true,
      canonical_benchmark_ready: true,
      promotion_benchmark_ready: true,
      retrain_gate: true,
      retrain_gate_reasons: [],
      retrain_gate_counts: {},
    });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/prairie/retrain-requests") {
    sendJson(response, 200, { count: 0, requests: [] });
    return;
  }

  if (request.method === "GET" && url.pathname === "/v2/training/domains") {
    sendJson(response, 200, {
      count: 1,
      domains: [
        {
          domain_id: "domain_prairie",
          name: "Prairie",
          description: "Mock training domain",
          owner_scope: "shared",
          owner_user_id: "user_mobile_smoke",
          metadata: {},
          created_at: nowIso,
          updated_at: nowIso,
        },
      ],
    });
    return;
  }

  if (
    request.method === "GET" &&
    url.pathname === "/v2/training/domains/domain_prairie/lineages"
  ) {
    sendJson(response, 200, {
      count: 1,
      lineages: [
        {
          lineage_id: "lineage_prairie",
          domain_id: "domain_prairie",
          scope: "shared",
          owner_user_id: "user_mobile_smoke",
          model_key: "prairie_yolo",
          parent_lineage_id: null,
          active_version_id: "version_active",
          metadata: {},
          created_at: nowIso,
          updated_at: nowIso,
        },
      ],
    });
    return;
  }

  if (
    request.method === "GET" &&
    url.pathname === "/v2/training/lineages/lineage_prairie/versions"
  ) {
    sendJson(response, 200, {
      count: 1,
      versions: [
        {
          version_id: "version_active",
          lineage_id: "lineage_prairie",
          source_job_id: null,
          artifact_run_id: null,
          status: "active",
          metrics: { benchmark_ready: true, promotion_benchmark_ready: true },
          metadata: { guardrails: { passed: true, reasons: [] } },
          created_at: nowIso,
          updated_at: nowIso,
        },
      ],
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
