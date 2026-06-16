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

  # API -> Go control plane (service name on the compose network). Streaming-
  # friendly so chat token streams and long uploads are not buffered/timed out.
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

  # SPA — the app has no client-side router, so serve index.html for any path.
  location / {
    try_files $uri $uri/ /index.html;
  }
}
EOF
EXPOSE 80
