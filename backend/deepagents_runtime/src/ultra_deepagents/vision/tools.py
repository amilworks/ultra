"""``inspect_images`` — the vision-reasoner's host-side bridge to the Qwen VLM.

Self-contained: it resolves a sandbox/host image path (guarded to known roots — no
SSRF, no traversal), crops to an optional bbox (zoomed so small objects are legible),
resizes the longest edge below the V100-safe cap, base64-encodes, and calls the
on-prem Qwen3.6-27B VLM with thinking on, returning the analysis + a reasoning excerpt.

Design: the api key + endpoint live only here (host side); the network-none code
sandbox never sees them. The vision-reasoner subagent uses the INHERITED text model
to orchestrate (decide which images/crops to look at, loop, synthesize) and calls
this tool for the actual seeing — so no per-subagent VLM model or multimodal
middleware exemption is needed (the subagent's model only ever receives text).
"""

from __future__ import annotations

import base64
import io
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from PIL import Image

from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_vision_chat_model

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
_MAX_IMAGE_BYTES = 100_000_000  # 100MB on-disk cap (reject before decode)
_MAX_IMAGE_PIXELS = 50_000_000  # ~50MP decode cap — a 24MP image once crashed the V100 engine
# Defuse PIL decompression bombs globally: raise instead of warn past this pixel count.
Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS


class _VlmDeadlineError(Exception):
    """The whole VLM call exceeded its hard wall-clock deadline.

    httpx's ``read`` timeout is INTER-BYTE — it resets on every byte/keepalive — so a
    half-dead tesla TCP connection that dribbles bytes can hang ``.invoke()`` forever
    (this is exactly how a live run wedged for 1h43m). This caps the WHOLE call.
    """


def _invoke_with_deadline(call: Any, *, timeout: float) -> Any:
    """Run a blocking call in a daemon thread and abandon it past ``timeout`` seconds.

    On expiry the worker thread is *abandoned* — a running thread is uncancellable, but
    a daemon thread cannot block process exit and (unlike a fixed ThreadPoolExecutor)
    cannot exhaust a pool, so a leaked hung thread is harmless and bounded by rarity.
    The caller MUST hold/release any semaphore itself so the permit is freed in this
    surviving thread, never in the abandoned one.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - relay every error to the caller thread
            box["error"] = exc

    worker = threading.Thread(target=_run, name="vlm-invoke", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise _VlmDeadlineError(f"vision call exceeded {timeout:.0f}s wall-clock deadline")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return path == root


_FAST_MAX_TOKENS = 768  # triage budget: a quick label needs no long CoT


def _mode_max_tokens(mode: str, default_max_tokens: int) -> int:
    return min(default_max_tokens, _FAST_MAX_TOKENS) if str(mode).lower() == "fast" else default_max_tokens


def _message_text(resp: Any) -> str:
    """Extract text from an AIMessage whose content may be a str OR a list of content
    blocks (some vLLM response shapes) — str(list) would otherwise emit a Python repr."""
    content = getattr(resp, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                out.append(str(block["text"]))
        return " ".join(out).strip()
    return str(content).strip()


def _message_reasoning(resp: Any) -> str:
    """The non-streamed response may carry the CoT under 'reasoning' OR 'reasoning_content'."""
    ak = getattr(resp, "additional_kwargs", {}) or {}
    return str(ak.get("reasoning_content") or ak.get("reasoning") or "").strip()


def _sampling_preset(mode: str, max_tokens: int) -> dict[str, Any]:
    """Per-request sampling. Modes, all with ``top_k`` + ``chat_template_kwargs``
    via ``extra_body`` (vLLM extensions):
      - fast: thinking OFF, low (768) budget — quick triage/classification of MANY images.
      - grounded (default): thinking OFF, full budget — a careful descriptive/diagnostic
        read of ONE image that stays anchored to the pixels.
      - precise: thinking on, temp 0.6 — exact figure/number/OCR reads.
      - reasoning: thinking on, temp 1.0 — step-by-step verification of a SPECIFIC hypothesis.

    WHY grounded is the default for holistic judgment: traced + benchmarked (2026-06-20) on a
    head-CT NPH question, the EXTENDED-THINKING modes (reasoning temp 1.0 AND precise temp 0.6)
    repeatedly CONFABULATED the full DESH triad on a NORMAL patient — the model reasons itself
    INTO the disease the question primes. Thinking-OFF read the same slice correctly 4/4 on a
    normal patient AND 4/4 on an NPH patient (perfect discrimination) in ~1/5 the wall-clock.
    For an open-ended "what does this image show / does it show condition D" judgment, extended
    thinking is a liability, not an asset; reserve it (reasoning/precise) for a narrow, specific
    check (a single detection crop, an exact figure/number) where a verifiable target backstops it.
    """
    mode = str(mode).lower()
    if mode == "fast":
        return {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        }
    if mode == "grounded":
        # thinking OFF (no confabulation) but the FULL output budget, unlike fast's 768 cap —
        # a thorough grounded read, the validated default for descriptive/diagnostic judgment.
        return {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        }
    base: dict[str, Any] = {
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True}},
    }
    if mode == "precise":
        return {**base, "temperature": 0.6}
    return {**base, "temperature": 1.0, "presence_penalty": 0.0}


def build_vision_tools(
    settings: RuntimeSettings,
    *,
    workspace_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    upload_roots: tuple[str, ...] = (),
) -> list[Any]:
    """Build the vision-reasoner's tool surface, closing over the VLM + the host
    roots the subagent's sandbox paths (``/workspace``, ``/outputs``) map to."""
    vision_model = build_vision_chat_model(settings)
    max_edge = int(settings.qwen_vlm_client_max_edge)
    default_max_tokens = int(settings.qwen_vlm_max_tokens)
    max_images = max(1, int(settings.qwen_vlm_max_images_per_call))  # the server's hard per-prompt cap
    # Shared across every inspect_images call this run: when the agent fans out many
    # parallel image inspections (a 100-image analysis), this caps how many actually
    # hit the VLM at once so we never exceed the server's max-num-seqs. langgraph runs
    # sync tools in a thread pool, so a threading.Semaphore bounds them correctly.
    call_semaphore = threading.Semaphore(max(1, int(settings.qwen_vlm_max_concurrency)))
    # Soft cap on per-run deep reads: looping inspect_images over a whole slice stack bloats
    # the (vision) subagent's own context and stalled its final synthesis on a live run (a
    # 28-deep-read NPH workup hung). screen_images is the right tool for bulk; past this count
    # we append a nudge to steer back to screen-then-conclude. It NUDGES, never blocks, so a
    # legitimate many-crop verification still completes. next() on itertools.count is atomic.
    inspect_call_counter = itertools.count(1)
    inspect_soft_cap = 12
    # Hard wall-clock cap on a single VLM HTTP call. The VLM answers well within the
    # request timeout; 1.5x is headroom. httpx's inter-byte read timeout cannot bound a
    # dribbling half-dead connection, so this is the real guarantee the call returns.
    vlm_wall_clock_cap = max(1.0, float(settings.qwen_vlm_request_timeout_seconds) * 1.5)

    ws = Path(workspace_dir).resolve() if workspace_dir else None
    art = Path(artifact_dir).resolve() if artifact_dir else None
    allowed_roots = [p for p in (ws, art, *[Path(r).resolve() for r in upload_roots if r]) if p]

    def _resolve(raw: str) -> Path:
        p = str(raw or "").strip()
        if not p:
            raise ValueError("empty image path")
        if p.startswith("/workspace/") and ws is not None:
            cand = ws / p[len("/workspace/") :]
        elif p == "/workspace" and ws is not None:
            cand = ws
        elif p.startswith("/outputs/") and art is not None:
            cand = art / p[len("/outputs/") :]
        elif p == "/outputs" and art is not None:
            cand = art
        else:
            cand = Path(p).expanduser()
        cand = cand.resolve(strict=False)  # resolves symlinks too, so the guard can't be escaped
        # SECURITY: never skip the containment check. With no roots configured the only
        # safe action is to refuse (an empty allowed_roots must NOT mean "allow anything").
        if not allowed_roots:
            raise ValueError(
                f"vision tool has no allowed image roots configured; cannot safely read: {raw}"
            )
        if not any(_is_under(cand, r) for r in allowed_roots):
            raise ValueError(
                f"image path is outside the allowed roots (/workspace, /outputs, uploads): {raw}"
            )
        if cand.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"not a recognized image file: {raw}")
        if not cand.is_file():
            raise FileNotFoundError(f"image not found: {raw} -> {cand}")
        if cand.stat().st_size > _MAX_IMAGE_BYTES:
            raise ValueError(f"image file too large ({cand.stat().st_size} bytes): {raw}")
        return cand

    def _prep(path: Path, bbox: list[float] | None) -> tuple[str, tuple[int, int]]:
        # Pixel-cap BEFORE convert(): a decompression bomb / pathological image must be
        # rejected before it allocates gigabytes (the late max_edge resize is too late).
        with Image.open(path) as probe:
            if probe.size[0] * probe.size[1] > _MAX_IMAGE_PIXELS:
                raise ValueError(f"image dimensions too large ({probe.size[0]}x{probe.size[1]})")
        im = Image.open(path).convert("RGB")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = (float(v) for v in bbox)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            longer = max(x2 - x1, y2 - y1, 1.0)
            pad = max(2.0 * longer, 96.0)  # context padding so the object isn't edge-cropped
            box = (
                max(0, int(x1 - pad)),
                max(0, int(y1 - pad)),
                min(im.size[0], int(x2 + pad)),
                min(im.size[1], int(y2 + pad)),  # clamp: out-of-bounds bbox -> empty crop -> hallucination
            )
            if box[2] > box[0] and box[3] > box[1]:
                im = im.crop(box)
                # upsample tiny crops so fine structure is legible (the burrow/cell detail)
                target = int(max_edge * 0.6)
                if max(im.size) < target:
                    scale = min(4.0, target / max(im.size))
                    im = im.resize(
                        (max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale))),
                        Image.LANCZOS,
                    )
        if max(im.size) > max_edge:
            im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        return url, im.size

    def _call_vlm(message: HumanMessage, mode: str) -> dict[str, Any]:
        """One VLM call with the mode's preset, bounded concurrency, transient-error
        backoff (3 attempts), and a monotonic budget-doubling retry on mid-<think>
        truncation. Returns {content, reasoning, finish, usage} or {error}."""
        max_tokens = _mode_max_tokens(mode, default_max_tokens)
        content = reasoning = finish = ""
        usage: dict[str, Any] = {}
        for attempt in range(3):
            transient: Exception | None = None
            # Acquire OUTSIDE the deadline-bounded call so the permit is always released
            # in this (surviving) thread — never in an abandoned worker thread (which
            # would leak the permit and starve every other vision call, as it did live).
            call_semaphore.acquire()  # bound concurrent VLM calls to the server's capacity
            try:
                resp = _invoke_with_deadline(
                    lambda: vision_model.bind(**_sampling_preset(mode, max_tokens)).invoke([message]),
                    timeout=vlm_wall_clock_cap,
                )
            except _VlmDeadlineError:
                # A stalled/half-dead connection httpx's inter-byte timeout cannot catch.
                # Fail fast — retrying re-hits the dead connection and leaks another thread.
                return {"error": (
                    f"ERROR: vision model did not respond within {vlm_wall_clock_cap:.0f}s "
                    "(connection stalled). Proceed without the second opinion."
                )}
            except (TimeoutError, ConnectionError, OSError) as exc:
                transient = exc  # backoff + retry happens after the semaphore is released
            except Exception as exc:  # noqa: BLE001 - any other model error -> structured, never raise
                return {"error": (
                    f"ERROR: vision model call failed ({type(exc).__name__}: {str(exc)[:160]}). "
                    "Proceed without the second opinion."
                )}
            finally:
                call_semaphore.release()
            if transient is not None:
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))  # brief backoff on a transient endpoint blip
                    continue
                return {"error": (
                    f"ERROR: vision model unavailable after retries ({type(transient).__name__}: "
                    f"{str(transient)[:160]}). Proceed without the second opinion."
                )}
            content = _message_text(resp)
            reasoning = _message_reasoning(resp)
            finish = str((resp.response_metadata or {}).get("finish_reason") or "")
            usage = resp.usage_metadata or {}
            if not content and finish == "length" and attempt < 2:
                grown = min(max_tokens * 2, settings.qwen_vlm_max_input_tokens - 4000)
                if grown > max_tokens:  # only retry if the budget can actually grow within the window
                    max_tokens = grown
                    continue
            break
        return {"content": content, "reasoning": reasoning, "finish": finish, "usage": usage, "budget": max_tokens}

    @tool
    def inspect_images(
        question: str,
        image_paths: list[str],
        bbox: list[float] | None = None,
        mode: str = "grounded",
    ) -> str:
        """Carefully look at ONE image (or a few, <=4) with the vision-language model.

        Use for DEEP, careful judgment of a SINGLE thing: describe one image in detail, judge
        what a structure/region shows, verify whether a detector's box is a real object or a
        false positive (pass its bbox), read/verify one plot or scientific figure, OCR labels,
        or compare 2-4 images side by side. >>> For screening/classifying MANY images (more
        than ~4), DO NOT loop this — use screen_images instead (far faster). <<<

        DEFAULT mode is 'grounded' (thinking OFF) — for any open-ended "what does this image
        show / does it show condition D" judgment, KEEP it grounded: extended-thinking modes
        make this VLM reason itself INTO a plausible-but-false finding (it confabulated a full
        disease pattern on a normal scan; grounded read it correctly). Only pass
        mode='reasoning'/'precise' for a NARROW, specific check (one detection crop, one exact
        figure/number) where a verifiable target backstops the extra thinking.

        Args:
            question: A precise instruction or question. For verification, ask for a
                structured verdict (e.g. "Is the centered object a real <class> or a
                false positive? Cite >=2 visual observations. End with: VERDICT:
                <real|false_positive|uncertain> CONFIDENCE: <0-1>").
            image_paths: image paths (/workspace/..., /outputs/..., or an upload path).
                At most a few per call (the server caps images per prompt); for more,
                call inspect_images multiple times.
            bbox: Optional [x1,y1,x2,y2] in source-image pixels; crops a zoomed,
                context-padded region of the FIRST image (use for single-detection
                verification so the object is large and legible).
            mode: "grounded" (DEFAULT; NO extended thinking, full budget — the right mode for
                describing or judging what an image shows, including diagnostic gestalt; stays
                anchored to the pixels), "precise" (thinking on, low temp — exact figure/number/
                OCR reads), "reasoning" (thinking on — step-by-step check of ONE specific
                hypothesis on a crop; do NOT use for open-ended "what condition is this"), or
                "fast" (NO thinking, 768-token — triage of MANY images via screen_images).

        Returns the model's analysis, a reasoning excerpt, and token usage. NOTE: this
        model cannot reliably COUNT many small objects or measure pixels — never ask it
        to enumerate, measure, or produce/correct coordinates; that stays with the
        specialist detectors.
        """
        if not image_paths:
            return "ERROR: no image_paths provided."
        if len(image_paths) > max_images:
            # never silently drop images — the agent must batch correctly (the server
            # hard-rejects more than max_images per prompt with a 400).
            return (
                f"ERROR: too many images for one call ({len(image_paths)} > {max_images}). "
                f"Call inspect_images multiple times with at most {max_images} images each."
            )
        paths = list(image_paths)
        blocks: list[dict[str, Any]] = [{"type": "text", "text": str(question or "Describe these images.")}]
        sizes: list[str] = []
        for idx, raw in enumerate(paths):
            try:
                url, size = _prep(_resolve(raw), bbox if idx == 0 else None)
            except (ValueError, FileNotFoundError, OSError) as exc:
                return f"ERROR resolving/loading image {raw!r}: {exc}"
            blocks.append({"type": "image_url", "image_url": {"url": url}})
            sizes.append(f"{Path(raw).name}={size[0]}x{size[1]}")

        res = _call_vlm(HumanMessage(content=blocks), mode)
        if res.get("error"):
            return res["error"]
        content = res["content"]
        if not content:
            return (
                f"ERROR: vision model returned no answer (finish_reason={res['finish'] or 'unknown'}, "
                f"tokens in={res['usage'].get('input_tokens')}/out={res['usage'].get('output_tokens')}, "
                f"budget={res['budget']}). Treat as uncertain; retry with a tighter question or fewer/smaller images."
            )
        parts = [f"[inspected {len(paths)} image(s): {', '.join(sizes)}]", content]
        if res["reasoning"]:
            parts.append(f"\n--- model reasoning (excerpt) ---\n{res['reasoning'][:1200]}")
        if res["usage"]:
            parts.append(
                f"\n(vlm tokens: prompt={res['usage'].get('input_tokens')} "
                f"completion={res['usage'].get('output_tokens')}; finish={res['finish']})"
            )
        n_deep = next(inspect_call_counter)
        if n_deep > inspect_soft_cap:
            parts.append(
                f"\n[NOTE: this is your {n_deep}th inspect_images deep-read this run. You are "
                "deep-reading many images one-by-one — that bloats your context and risks stalling "
                "your synthesis. Use screen_images ONCE for bulk triage; otherwise CONCLUDE NOW "
                "from what you have already seen. Do not keep looping inspect_images.]"
            )
        return "\n".join(parts)

    @tool
    def screen_images(question: str, image_paths: list[str]) -> str:
        """USE THIS FIRST for ANY task over MORE THAN ~4 images — fast bulk screening, one call.

        Whenever you have many images and the same per-image question (triage, classification,
        screening, "which images show X", "classify each plot", "flag images containing Y"),
        call screen_images ONCE with the full list — never loop inspect_images over many
        images (that is slow and the wrong tool). It batches the images internally (a few per
        VLM prompt), runs the batches concurrently with NO extended thinking (fast), and
        returns one line per image. This is THE way to handle 10-500 images efficiently.
        For the careful FINAL judgment of the few flagged items, follow up with inspect_images
        (grounded default; add a bbox if verifying a single detection crop).

        Args:
            question: the SAME question to ask about every image. Ask for a short, structured
                per-image answer (e.g. "reply 'yes' or 'no': does this image contain X?").
            image_paths: the full list of image paths (/workspace/..., /outputs/..., uploads).

        Returns a per-image table ("<filename>: <answer>"). NOT for counting many small
        objects or measuring — that stays with the specialist detectors.
        """
        if not image_paths:
            return "ERROR: no image_paths provided."
        if len(image_paths) > 500:
            return f"ERROR: too many images ({len(image_paths)} > 500). Split into smaller screenings."
        prepped: list[tuple[str, str]] = []
        errors: list[str] = []
        for raw in image_paths:
            try:
                url, _ = _prep(_resolve(raw), None)
                prepped.append((Path(raw).name, url))
            except (ValueError, FileNotFoundError, OSError) as exc:
                errors.append(f"{Path(raw).name}: UNREADABLE ({exc})")
        if not prepped:
            return "ERROR: no readable images.\n" + "\n".join(errors)
        chunks = [prepped[i : i + max_images] for i in range(0, len(prepped), max_images)]

        def _run_chunk(chunk: list[tuple[str, str]]) -> str:
            names = [n for n, _ in chunk]
            text = (
                f"You are given {len(chunk)} images, named (in this order): {', '.join(names)}.\n"
                f"{question}\n"
                "Answer with EXACTLY one line per image, formatted as '<filename>: <answer>'. "
                "Be concise and precise; only assert a property when the image clearly shows it."
            )
            blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
            blocks.extend({"type": "image_url", "image_url": {"url": u}} for _, u in chunk)
            res = _call_vlm(HumanMessage(content=blocks), "fast")
            if res.get("error"):
                return "\n".join(f"{n}: ERROR ({res['error'][:80]})" for n in names)
            return res["content"] or "\n".join(f"{n}: (no answer)" for n in names)

        workers = max(1, int(settings.qwen_vlm_max_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunk_results = list(pool.map(_run_chunk, chunks))
        report = "\n".join(chunk_results)
        if errors:
            report += "\n\n[unreadable images]\n" + "\n".join(errors)
        return (
            f"[screened {len(prepped)} image(s) in {len(chunks)} fast batch(es) of <= {max_images}]\n"
            + report
        )

    return [inspect_images, screen_images]
