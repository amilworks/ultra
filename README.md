# BisQue Ultra

BisQue Ultra is a scientific imaging system with a clear division of labor. BisQue stores images, datasets, and metadata. Keycloak handles login. A FastAPI backend routes tools, runs, and model calls. A React frontend keeps the whole process visible. The only part you swap freely is the inference engine. This repo already knows how to talk to both vLLM and Ollama through the same OpenAI-compatible interface, so you can choose the model stack that fits your hardware instead of rewriting the application around one vendor.

If you want one sentence to hold the whole design in your head, use this one: BisQue Ultra is a local scientific workbench whose storage layer is BisQue, whose control plane is FastAPI, whose interface is React, and whose language model can come from any OpenAI-compatible server.

## What You Are Launching

You are starting four layers:

1. `platform/bisque/` brings up BisQue, Postgres, and Keycloak in Docker.
2. `src/` serves the BisQue Ultra API on `http://127.0.0.1:8000`.
3. `frontend/` serves the web client on `http://localhost:5173`.
4. Your model server, usually vLLM or Ollama, answers OpenAI-style chat requests.

Those layers are deliberately separate. If a page loads but chat fails, the frontend is alive and the API or model server is not. If login fails, the problem is usually in the BisQue or Keycloak layer, not the LLM layer. That separation is a feature, because it lets you debug the system by following the symptom instead of guessing.

## Before You Start

Install the three tools this repo assumes:

- [uv](https://github.com/astral-sh/uv) for Python dependency management
- [pnpm](https://pnpm.io/) for the frontend
- Docker with Compose for BisQue, Keycloak, and Postgres

You also need one model backend:

- [vLLM](https://docs.vllm.ai/) if you want high-throughput serving for large open-weight models
- [Ollama](https://docs.ollama.com/openai) if you want the shortest path from a workstation to a working local assistant

## The Fastest Mental Model for Configuration

BisQue Ultra resolves model settings in this order:

1. `LLM_BASE_URL`, `LLM_MODEL`, and `LLM_API_KEY` override everything.
2. If `LLM_PROVIDER=ollama`, the app falls back to `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
3. Otherwise, the app falls back to `OPENAI_BASE_URL` and `OPENAI_MODEL`.
4. Code-generation tools can use a different model family through `CODEGEN_PROVIDER`, `CODEGEN_BASE_URL`, and `CODEGEN_MODEL`.

That design matters. It means you can keep the whole app stable while changing only one layer:

- vLLM for all chat and tool reasoning
- Ollama for all chat and tool reasoning
- vLLM for reasoning but Ollama for code generation
- one remote OpenAI-compatible server today, another tomorrow

The application code does not need to know which story you chose. It only needs a working OpenAI-compatible endpoint.

## Step 1: Create Your Local `.env`

Start from the template:

```bash
cp .env.example .env
```

The public template is now local-first. It no longer points to an internal lab server. Out of the box it assumes:

- BisQue on `http://localhost:8080`
- API on `http://localhost:8000`
- frontend on `http://localhost:5173`
- vLLM on `http://localhost:8001/v1`
- Ollama on `http://localhost:11434/v1`

You do not need to fill every variable. The important move is to decide which inference engine you are using, then set the model section cleanly.

## Step 2: Choose an Inference Engine

### Option A: vLLM

Choose vLLM when you want a stronger open-weight model, better throughput, or a server that can feed multiple users without turning sluggish. In this repo, vLLM is treated as an OpenAI-compatible endpoint. That is why the environment keys still say `OPENAI_BASE_URL` and `OPENAI_MODEL` even when the actual server is vLLM.

If you want the `.env.example` defaults to work without extra renaming, launch vLLM with a served model name that matches the config:

```bash
vllm serve openai/gpt-oss-120b \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name gpt-oss-120b \
  --api-key EMPTY
```

Then keep this shape in `.env`:

```bash
LLM_PROVIDER=vllm
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_MODEL=gpt-oss-120b
OPENAI_API_KEY=EMPTY
```

Three details are worth understanding:

- The app talks to the OpenAI-compatible route, so the base URL must end in `/v1`.
- The model name in BisQue Ultra must match the model name vLLM exposes.
- For local OpenAI-compatible servers, a placeholder key like `EMPTY` is often enough. This app already handles that convention.

If you want a different model, change both the vLLM launch command and `OPENAI_MODEL`. Keep them synchronized. When they drift, the API may stay healthy while completions fail with a model-not-found error.

### Option B: Ollama

Choose Ollama when you value simplicity more than throughput. The setup is lighter, the commands are easier to remember, and the cost of experimentation is lower. The tradeoff is that very heavy reasoning or large multimodal workloads may feel better on vLLM-backed hardware.

Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen2.5:14b-instruct
```

Then set:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=qwen2.5:14b-instruct
```

The `/v1` suffix matters here too. BisQue Ultra uses an OpenAI client under the hood. It does not talk to Ollama’s older native endpoints directly. That single design choice is why the same backend can pivot between Ollama and vLLM without changing the orchestration code.

### Option C: One Model for Reasoning, Another for Code

Sometimes the strongest chat model is not the cheapest model for long code-execution loops. This repo supports that split explicitly:

```bash
LLM_PROVIDER=vllm
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_MODEL=gpt-oss-120b

CODEGEN_PROVIDER=ollama
CODEGEN_BASE_URL=http://localhost:11434/v1
CODEGEN_MODEL=qwen2.5:14b-instruct
```

That split is not cosmetic. It lets you keep expensive scientific reasoning on the stronger model while routing repair-style code generation to a cheaper local model.

## Step 3: Install Dependencies

Install the backend:

```bash
uv sync
```

Install the frontend:

```bash
pnpm --dir frontend install
```

If you skip one half, the failure mode will tell on itself. Missing Python dependencies usually break the API on startup. Missing frontend packages usually leave Vite unable to build or serve.

## Step 4: Bring Up the BisQue Platform

Start BisQue, Keycloak, and Postgres through the root Makefile:

```bash
make platform-up
```

Why the root Makefile? Because this repo treats the root `.env` as the shared contract between BisQue Ultra and the absorbed BisQue platform. Starting services from subdirectories invites drift. Starting them from the root keeps the whole stack reading from one configuration source.

When this succeeds, you should have:

- BisQue at `http://localhost:8080`
- Keycloak at `http://localhost:18080`

## Step 5: Start the API and Frontend

Use the helper script:

```bash
./scripts/restart_dev.sh restart
```

That script starts:

- the FastAPI backend on `http://127.0.0.1:8000`
- the React frontend on `http://localhost:5173`

It also gives you a quick status command:

```bash
./scripts/restart_dev.sh status
```

If you want to stop the app layer without tearing down BisQue:

```bash
./scripts/restart_dev.sh stop
```

## Step 6: Verify the System

Run the platform smoke test first:

```bash
make verify-platform-smoke
```

Then run the app-to-platform smoke test:

```bash
make verify-integration
```

Those two checks answer two different questions:

- Is BisQue itself healthy?
- Can BisQue Ultra actually talk to BisQue?

That distinction saves time. A green platform check with a red integration check usually means your local API is misconfigured. A red platform check means the bug is lower in the stack.

You can also check the live endpoints directly:

```bash
curl -fsS http://127.0.0.1:8000/v1/health
curl -I -fsS http://localhost:5173
```

## What the Ports Mean

These ports are easy to confuse because they all belong to one system but not to one process.

- `8080`: BisQue itself
- `18080`: Keycloak
- `8000`: BisQue Ultra API
- `5173`: BisQue Ultra frontend
- `8001`: example local vLLM endpoint
- `11434`: default Ollama endpoint

If the frontend says the API is unavailable, look at `8000`. If the API is healthy but chat hangs, inspect the model endpoint you configured. If auth redirects loop or fail, look at `8080` and `18080`.

## Common Failure Modes

### The frontend loads, but chat fails

Usually the API or the model backend is down.

Check:

```bash
./scripts/restart_dev.sh status
curl -fsS http://127.0.0.1:8000/v1/health
```

If the API is healthy, your next suspect is the model server. Make sure the base URL ends in `/v1` and the model name in `.env` matches the model name the server actually exposes.

### The API starts, then dies on import

This usually means the environment is incomplete, not that the whole architecture is broken. Run:

```bash
uv sync
uv run python -m py_compile src/api/main.py src/agno_backend/runtime.py src/tools.py
```

### Login works poorly or BisQue looks half-alive

Run:

```bash
make verify-platform-smoke
```

If that fails, the issue is in the BisQue or Keycloak layer. Do not debug the frontend first.

### vLLM is up, but the app says the model is missing

The usual mistake is a served-model-name mismatch. If you launch:

```bash
vllm serve openai/gpt-oss-120b
```

then your request model may need to be `openai/gpt-oss-120b` unless you set `--served-model-name gpt-oss-120b`.

### Ollama is running, but requests still fail

Make sure you are pointing at the OpenAI-compatible route:

```bash
OLLAMA_BASE_URL=http://localhost:11434/v1
```

not just:

```bash
http://localhost:11434
```

## Optional Assets

The repo does not vendor large model weights or scientific checkpoints. If you want the full imaging tool surface locally, provision these separately:

- `data/models/medsam2/checkpoints/`
- `data/models/sam3/`
- YOLO or prairie-dog weights such as `RareSpotWeights.pt` and `yolo26x.pt`

The absence of those assets does not stop the web stack from booting. It only narrows which tools can run successfully.

## Repo Layout

- `src/`: FastAPI backend, tool runtime, Agno orchestration, scientific logic
- `src/evals/`: runtime evaluation and review helpers imported by the backend
- `frontend/`: React and Vite client
- `platform/bisque/`: absorbed BisQue platform, Docker build context, Keycloak assets
- `scripts/`: startup and smoke-check helpers

## The Shortest Path to a Working System

If you already know which model you want, this is the whole story:

```bash
cp .env.example .env
uv sync
pnpm --dir frontend install
make platform-up
./scripts/restart_dev.sh restart
make verify-platform-smoke
make verify-integration
```

Then point your browser at [http://localhost:5173](http://localhost:5173).

The rest of this README exists to make that path legible, not longer.
