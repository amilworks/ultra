# syntax=docker/dockerfile:1
#
# Frontend: Vite production build served by nginx, which proxies the API to the
# Go control plane. Build context = frontend.  docker build -t ultra-frontend:local frontend
FROM node:22-bookworm-slim AS build
WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml ./
# --ignore-scripts: the only flagged build script is msw (a test-only mock dep);
# it's irrelevant to the production bundle, and skipping it avoids pnpm's
# ERR_PNPM_IGNORED_BUILDS hard-fail. Native bundler binaries (rolldown/oxide/
# esbuild) come from platform packages, not postinstall scripts.
RUN pnpm install --frozen-lockfile --ignore-scripts
COPY . .
RUN pnpm run build

# Precompress text assets so the edge (Caddy file_server `precompressed`) can
# serve .br/.zst/.gz sidecars with a real Content-Length instead of paying
# per-request compression CPU — and gain brotli, which on-the-fly encoding
# doesn't offer. Images/fonts are already compressed formats; skip them.
RUN apt-get update && apt-get install -y --no-install-recommends brotli zstd \
  && rm -rf /var/lib/apt/lists/* \
  && find dist -type f \( -name '*.js' -o -name '*.css' -o -name '*.html' \
       -o -name '*.svg' -o -name '*.json' -o -name '*.txt' -o -name '*.map' \
       -o -name '*.webmanifest' \) -size +1k \
       -exec brotli -q 11 -k {} \; \
       -exec zstd -19 -q -k {} \; \
       -exec gzip -9 -k {} \;

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY <<'EOF' /etc/nginx/conf.d/default.conf
server {
  listen 80;
  server_name _;
  root /usr/share/nginx/html;
  index index.html;

  # Large scientific images/videos upload through here.
  client_max_body_size 0;

  # This nginx is the LOCAL EDGE and deliberately mirrors the production edge
  # contract so behaviour a developer sees here matches what ships: same-origin
  # API, identical security headers, identical cache policy, precompressed
  # static assets. Only deliberate difference: no HSTS (production terminates
  # TLS; local is plain http, where HSTS would be meaningless and would poison
  # the browser).
  server_tokens off; # production strips the Server header too
  add_header Content-Security-Policy-Report-Only "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; object-src 'none'" always;
  add_header X-Content-Type-Options nosniff always;
  add_header X-Frame-Options SAMEORIGIN always;
  add_header Referrer-Policy strict-origin-when-cross-origin always;

  # Serve the .gz sidecars the build stage precompressed, like the production
  # edge's precompressed file serving. (Production also serves .br; the stock
  # nginx alpine image has no brotli module, so brotli is prod-only.)
  gzip_static on;

  # API -> Go control plane (service name on the compose network). Streaming-
  # friendly so chat token streams and long uploads are not buffered/timed out,
  # matching the production proxy's unbuffered streaming.
  location ~ ^/(v1|v2|v3)/ {
    proxy_pass http://control-plane:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_request_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
  }

  # The production edge also proxies the API docs surfaces to the control plane.
  location = /openapi.json {
    proxy_pass http://control-plane:8000;
    proxy_set_header Host $host;
  }
  location /docs {
    proxy_pass http://control-plane:8000;
    proxy_set_header Host $host;
  }

  # Cache contract, identical to production: hashed /assets/* are immutable; the
  # app shell (index.html and every SPA fallback) is no-cache so a deploy takes
  # effect on the next navigation and a stale index can never reference a
  # 404'd hashed asset; stable images/fonts get a week.
  location /assets/ {
    add_header Cache-Control "public, max-age=31536000, immutable" always;
    try_files $uri =404;
  }
  location ~* \.(webp|png|svg|ico|woff2)$ {
    add_header Cache-Control "public, max-age=604800" always;
    try_files $uri =404;
  }

  # SPA — the app has no client-side router, so serve index.html for any path.
  location / {
    add_header Cache-Control "no-cache" always;
    try_files $uri $uri/ /index.html;
  }
}
EOF
EXPOSE 80
