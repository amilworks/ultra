"""Phase-1 tests for the vision-reasoner subagent + inspect_images tool.

Covers: config wiring, the registration gate (which INCLUDES rarespot/prairie goals,
unlike scoped delegation), subagent registration shape (tool present, response_format,
no self-set permissions), host-side path resolution + SSRF/traversal guard, the
multimodal message construction + sampling preset, and the empty-output retry guard.
All mocked — no live VLM.
"""

from __future__ import annotations

import threading
import time

from langchain_core.messages import AIMessage
from PIL import Image
from ultra_deepagents import vision as vision_pkg
from ultra_deepagents.agent import (
    SCOPED_DELEGATION_RESPONSE_FORMAT,
    VISION_SUBAGENT,
    _should_register_vision_subagent,
    build_subagents,
)
from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext
from ultra_deepagents.vision import tools as vision_tools


def _settings(**over) -> RuntimeSettings:
    base = dict(
        openai_base_url="http://localhost:8001/v1",
        openai_model="deepseek_v4",
        qwen_vlm_enabled=True,
        qwen_vlm_base_url="http://tesla.test:8000/v1",
        qwen_vlm_model="Qwen3.6-27B",
        qwen_vlm_api_key="test-key",
        qwen_vlm_max_tokens=4096,
        qwen_vlm_client_max_edge=1280,
        qwen_vlm_max_input_tokens=131072,
    )
    base.update(over)
    return RuntimeSettings(**base)


def _ctx(goal: str, **over) -> AgentRunContext:
    base = dict(
        assistant_id="a", org_id="o", user_id="u", project_id="p", thread_id="t",
        run_id="r", goal=goal,
    )
    base.update(over)
    return AgentRunContext(**base)


class _FakeBound:
    def __init__(self, parent, resp):
        self.parent = parent
        self.resp = resp

    def invoke(self, messages):
        self.parent.calls.append(messages)
        return self.resp() if callable(self.resp) else self.resp


class _FakeModel:
    """Stands in for the Qwen ChatOpenAI: records bind kwargs + invoked messages."""

    def __init__(self, resp):
        self.resp = resp
        self.bind_kwargs = []
        self.calls = []

    def bind(self, **kwargs):
        self.bind_kwargs.append(kwargs)
        return _FakeBound(self, self.resp)


def _ai(content, *, reasoning: str = "", reasoning_key: str = "reasoning_content", finish: str = "stop") -> AIMessage:
    return AIMessage(
        content=content,  # may be a str OR a list of content blocks
        additional_kwargs={reasoning_key: reasoning} if reasoning else {},
        response_metadata={"finish_reason": finish},
        usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
    )


def _make_image(path, size=(800, 600), color=(120, 140, 90)):
    Image.new("RGB", size, color).save(path)
    return str(path)


# ----- config -------------------------------------------------------------

def test_config_loads_qwen_vlm_fields(monkeypatch, tmp_path):
    keyfile = tmp_path / "k.api-key"
    keyfile.write_text("secret-123\n")
    monkeypatch.setenv("QWEN_VLM_ENABLED", "true")
    monkeypatch.setenv("QWEN_VLM_API_KEY_FILE", str(keyfile))
    monkeypatch.delenv("QWEN_VLM_API_KEY", raising=False)
    s = RuntimeSettings.from_env()
    assert s.qwen_vlm_enabled is True
    assert s.qwen_vlm_model == "Qwen3.6-27B"
    assert s.qwen_vlm_api_key == "secret-123"  # resolved from the file
    assert s.qwen_vlm_client_max_edge == 1280


# ----- registration gate --------------------------------------------------

def test_gate_includes_rarespot_goals_unlike_scoped_delegation():
    s = _settings()
    # The headline case: verifying a prairie-dog/rarespot detection. scoped-delegation
    # EXCLUDES these; the vision gate must INCLUDE them.
    assert _should_register_vision_subagent(_ctx("verify the prairie dog detection"), s)
    assert _should_register_vision_subagent(_ctx("look at this figure"), s)
    assert _should_register_vision_subagent(_ctx("solve a stochastic ODE"), s) is False


def test_gate_off_when_disabled_or_image_context():
    assert _should_register_vision_subagent(_ctx("look at this image"), _settings(qwen_vlm_enabled=False)) is False
    # Image context (selected files) triggers it even on a terse goal.
    assert _should_register_vision_subagent(_ctx("analyze", selected_file_ids=("f1",)), _settings())


# ----- subagent registration shape ---------------------------------------

def test_vision_subagent_registers_with_tool_and_no_permissions():
    s = _settings()
    tools = vision_pkg.build_vision_tools(s, workspace_dir="/tmp/ws", artifact_dir="/tmp/out")
    subs = build_subagents(None, context=_ctx("verify detection"), vision_tools=tools)
    vision = [x for x in subs if x["name"] == "vision-reasoner"]
    assert len(vision) == 1
    v = vision[0]
    assert "inspect_images" in [t.name for t in v["tools"]]
    assert v["response_format"] is not SCOPED_DELEGATION_RESPONSE_FORMAT  # deep-copied
    assert v["response_format"] == SCOPED_DELEGATION_RESPONSE_FORMAT
    assert "permissions" not in v  # inherits parent read-only perms; never self-sets
    assert VISION_SUBAGENT["name"] == "vision-reasoner"


def test_vision_subagent_prompt_steers_grounded_not_reasoning_for_gestalt():
    """The prompt must teach grounded-by-default for holistic 'what does this show' judgment and
    warn that extended thinking confabulates — the root-cause fix. It must NOT tell the model to
    deep-read a stack in reasoning mode (the old steering that drove the NPH confabulation)."""
    p = VISION_SUBAGENT["system_prompt"].lower()
    assert "grounded" in p
    assert "without extended thinking" in p or "no extended thinking" in p
    assert "reason itself into" in p  # the confabulation warning
    assert "mode='reasoning')" not in p  # no longer steers the holistic deep-read to reasoning


# ----- the inspect_images tool -------------------------------------------

def _tool_with_model(monkeypatch, fake, **kw):
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    return vision_pkg.build_vision_tools(_settings(), **kw)[0]


def test_inspect_images_builds_multimodal_message_and_preset(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("It is a green rectangle.", reasoning="looks uniform"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    out = tool.invoke({"question": "What is this?", "image_paths": [img], "mode": "reasoning"})
    assert "green rectangle" in out
    assert "model reasoning" in out  # reasoning excerpt surfaced
    # the model saw a multimodal HumanMessage with a text + image_url block
    msg = fake.calls[0][0]
    blocks = msg.content
    assert blocks[0]["type"] == "text"
    assert any(b.get("type") == "image_url" and b["image_url"]["url"].startswith("data:image/jpeg;base64,") for b in blocks)
    # reasoning preset: thinking on, top_k via extra_body, temp 1.0
    kw = fake.bind_kwargs[0]
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert kw["extra_body"]["top_k"] == 20
    assert kw["temperature"] == 1.0


def test_inspect_images_fast_mode_disables_thinking_and_caps_budget(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("scatter"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    tool.invoke({"question": "classify", "image_paths": [img], "mode": "fast"})
    kw = fake.bind_kwargs[0]
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False  # no CoT -> fast
    assert kw["max_tokens"] <= 768  # capped triage budget, not the 32k reasoning budget


def test_inspect_images_precise_mode_lowers_temperature(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("Bar A = 100."))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    tool.invoke({"question": "Read the chart.", "image_paths": [img], "mode": "precise"})
    assert fake.bind_kwargs[0]["temperature"] == 0.6


def test_inspect_images_default_is_grounded_no_thinking_full_budget(monkeypatch, tmp_path):
    """ROOT-CAUSE FIX (traced 2026-06-20): the default holistic read must run WITHOUT extended
    thinking — reasoning/precise (thinking on) confabulated a disease pattern on a normal CT
    (3-4/4), while thinking-off read it correctly (4/4) AND discriminated the true NPH case
    (4/4). Grounded = thinking OFF but the FULL budget, unlike fast's 768 triage cap."""
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("Ventricles are normal in size; no NPH."))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    # NO mode passed -> the default must be grounded
    tool.invoke({"question": "Does this show NPH?", "image_paths": [img]})
    kw = fake.bind_kwargs[0]
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False  # no CoT confabulation
    assert kw["max_tokens"] > 768  # full budget, NOT the fast triage cap (a thorough read)


def test_inspect_images_soft_cap_nudges_after_threshold(monkeypatch, tmp_path):
    """Defense-in-depth against the live stall: a 28-deep-read NPH workup bloated the vision
    subagent's context and hung its synthesis. After a soft cap, inspect_images appends a nudge
    to screen/conclude — it NUDGES, never blocks (a legit many-crop verify still completes)."""
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("ok"))  # reused for every call
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    outs = [tool.invoke({"question": "q", "image_paths": [img]}) for _ in range(13)]
    assert "[NOTE:" not in outs[0]  # early deep reads are clean
    assert "[NOTE:" not in outs[10]  # at the cap (12) still clean
    assert "[NOTE:" in outs[12]  # the 13th read (past the cap of 12) is nudged
    assert "screen_images" in outs[12] and "conclude" in outs[12].lower()


def test_inspect_images_grounded_and_reasoning_presets_contrast(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("ok"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    tool.invoke({"question": "q", "image_paths": [img], "mode": "grounded"})
    tool.invoke({"question": "q", "image_paths": [img], "mode": "reasoning"})
    grounded, reasoning = fake.bind_kwargs[0], fake.bind_kwargs[1]
    assert grounded["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert reasoning["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    # grounded keeps the full budget (it is a careful read, not triage)
    assert grounded["max_tokens"] == reasoning["max_tokens"]


def test_inspect_images_path_guard_rejects_outside_root(monkeypatch, tmp_path):
    fake = _FakeModel(_ai("x"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path / "ws"))
    out = tool.invoke({"question": "?", "image_paths": ["/etc/passwd"]})
    assert "outside the allowed roots" in out or "not a recognized image" in out
    assert not fake.calls  # never reached the model


def test_inspect_images_empty_output_retries_then_succeeds(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    seq = [_ai("", finish="length"), _ai("Now an answer.", finish="stop")]
    fake = _FakeModel(lambda: seq.pop(0))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    out = tool.invoke({"question": "?", "image_paths": [img]})
    assert "Now an answer." in out
    assert len(fake.calls) == 2  # one budget-doubling retry
    assert fake.bind_kwargs[1]["max_tokens"] > fake.bind_kwargs[0]["max_tokens"]


def test_inspect_images_bbox_crops_first_image(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png", size=(2000, 2000))
    fake = _FakeModel(_ai("cropped view"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    out = tool.invoke({"question": "verify", "image_paths": [img], "bbox": [900, 900, 980, 980]})
    assert "inspected 1 image" in out
    # a crop around an 80px box (+ padding, capped) is far smaller than the 2000px source
    sent_size_note = out.splitlines()[0]
    assert "2000x2000" not in sent_size_note


def test_concurrency_bound_caps_parallel_vlm_calls(monkeypatch, tmp_path):
    """The shared semaphore must cap concurrent VLM calls so a many-image fan-out
    cannot exceed the server's max-num-seqs (the multi-image-at-scale safety)."""
    import threading
    import time

    img = _make_image(tmp_path / "a.png")
    state = {"inflight": 0, "peak": 0}
    lock = threading.Lock()

    class _SlowBound:
        def invoke(self, messages):
            with lock:
                state["inflight"] += 1
                state["peak"] = max(state["peak"], state["inflight"])
            time.sleep(0.15)
            with lock:
                state["inflight"] -= 1
            return _ai("ok")

    class _SlowModel:
        def bind(self, **kw):
            return _SlowBound()

    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: _SlowModel())
    tool = vision_pkg.build_vision_tools(_settings(qwen_vlm_max_concurrency=2), workspace_dir=str(tmp_path))[0]

    threads = [threading.Thread(target=lambda: tool.invoke({"question": "?", "image_paths": [img]})) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["peak"] <= 2, f"concurrency bound breached: peak={state['peak']} > cap 2"


def test_path_guard_refuses_when_no_roots_configured(monkeypatch, tmp_path):
    """CRITICAL: with no allowed roots (workspace/artifact None), the guard must REFUSE,
    not fall through to arbitrary host file read."""
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("x"))
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    tool = vision_pkg.build_vision_tools(_settings())[0]  # no workspace_dir/artifact_dir -> empty roots
    out = tool.invoke({"question": "?", "image_paths": [img]})
    assert "no allowed image roots" in out
    assert not fake.calls


def test_decompression_bomb_rejected_before_decode(monkeypatch, tmp_path):
    """CRITICAL: an over-cap-dimension image must be rejected before convert() allocs."""
    monkeypatch.setattr(vision_tools, "_MAX_IMAGE_PIXELS", 10_000)  # 100x100 px cap for the test
    big = _make_image(tmp_path / "big.png", size=(400, 400))  # 160k px > 10k cap
    fake = _FakeModel(_ai("x"))
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    tool = vision_pkg.build_vision_tools(_settings(), workspace_dir=str(tmp_path))[0]
    out = tool.invoke({"question": "?", "image_paths": [big]})
    assert "dimensions too large" in out
    assert not fake.calls


def test_too_many_images_errors_not_silently_dropped(monkeypatch, tmp_path):
    fake = _FakeModel(_ai("x"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    imgs = [_make_image(tmp_path / f"i{i}.png") for i in range(9)]
    out = tool.invoke({"question": "?", "image_paths": imgs})
    assert "too many images" in out
    assert not fake.calls  # refuse rather than silently analyze only 8


def test_budget_grows_monotonically_and_does_not_loop(monkeypatch, tmp_path):
    """The retry must never SHRINK the budget and must stop (not infinite-loop) when the
    window is too small to grow."""
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("", finish="length"))  # always truncated
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    # tiny window so growth is impossible; max_tokens default 4096 > (window-4000)
    tool = vision_pkg.build_vision_tools(
        _settings(qwen_vlm_max_tokens=4096, qwen_vlm_max_input_tokens=5000), workspace_dir=str(tmp_path)
    )[0]
    out = tool.invoke({"question": "?", "image_paths": [img]})
    assert "no answer" in out
    assert len(fake.calls) <= 3  # bounded, no infinite loop
    assert all(kw["max_tokens"] >= 4096 for kw in fake.bind_kwargs)  # never shrank below initial


def test_list_content_is_extracted_not_repr(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai([{"type": "text", "text": "a real burrow"}]))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    out = tool.invoke({"question": "?", "image_paths": [img]})
    assert "a real burrow" in out
    assert "'type'" not in out  # not the Python repr of the list


def test_reasoning_field_fallback_key(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")
    fake = _FakeModel(_ai("verdict", reasoning="careful CoT", reasoning_key="reasoning"))
    tool = _tool_with_model(monkeypatch, fake, workspace_dir=str(tmp_path))
    out = tool.invoke({"question": "?", "image_paths": [img]})
    assert "careful CoT" in out  # picked up from the 'reasoning' key, not only 'reasoning_content'


def test_screen_images_batches_to_server_limit_and_uses_fast_mode(monkeypatch, tmp_path):
    """screen_images must chunk a large set to <= the server's per-prompt image cap and
    use fast (no-thinking) mode — the 100-image method."""
    imgs = [_make_image(tmp_path / f"i{i:02d}.png") for i in range(10)]
    fake = _FakeModel(_ai("i00.png: line\ni01.png: bar"))
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    tools = {t.name: t for t in vision_pkg.build_vision_tools(
        _settings(qwen_vlm_max_images_per_call=4), workspace_dir=str(tmp_path))}
    assert "screen_images" in tools
    out = tools["screen_images"].invoke({"question": "type?", "image_paths": imgs})
    assert "screened 10 image(s) in 3 fast batch(es)" in out  # 10 / 4 = 3 chunks
    # every chunk sent <= 4 images and used fast mode (thinking off)
    for call_msgs in fake.calls:
        n_imgs = sum(1 for b in call_msgs[0].content if b.get("type") == "image_url")
        assert n_imgs <= 4
    for kw in fake.bind_kwargs:
        assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_gate_off_when_key_empty(monkeypatch):
    assert _should_register_vision_subagent(_ctx("look at this image"), _settings(qwen_vlm_api_key="EMPTY")) is False


def test_inspect_images_endpoint_error_is_structured_not_raised(monkeypatch, tmp_path):
    img = _make_image(tmp_path / "a.png")

    class _Boom:
        def bind(self, **kw):
            raise ConnectionError("tesla down")

    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: _Boom())
    tool = vision_pkg.build_vision_tools(_settings(), workspace_dir=str(tmp_path))[0]
    out = tool.invoke({"question": "?", "image_paths": [img]})
    assert "vision model unavailable" in out  # degraded, never raises into the coordinator


def test_vision_wall_clock_cap_returns_error_and_frees_semaphore(monkeypatch, tmp_path):
    """A stalled VLM connection (httpx's inter-byte read timeout cannot catch a dribbling
    half-dead socket) must NOT hang the call. The hard wall-clock cap returns a structured
    error AND releases the semaphore permit in the SURVIVING thread, so a follow-up call
    still works — proving no permit leak (the exact bug that wedged a worker for 1h43m)."""
    img = _make_image(tmp_path / "scan.png")
    release = threading.Event()
    calls = {"n": 0}

    def _resp():
        calls["n"] += 1
        if calls["n"] == 1:
            release.wait(20)  # first call stalls past the cap; freed at teardown
            return _ai("late")
        return _ai("second call ok")  # only reachable if the permit was released

    fake = _FakeModel(_resp)
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    # cap = max(1.0, 0.5*1.5) = 1.0s; concurrency 1 means a LEAKED permit would block call 2 forever.
    s = _settings(qwen_vlm_request_timeout_seconds=0.5, qwen_vlm_max_concurrency=1)
    tool = vision_pkg.build_vision_tools(s, workspace_dir=str(tmp_path))[0]

    t0 = time.monotonic()
    out1 = tool.invoke({"question": "Describe", "image_paths": [img], "mode": "fast"})
    assert time.monotonic() - t0 < 8.0  # returned at ~cap, not the 20s block
    assert "proceed without the second opinion" in out1.lower()
    assert "stalled" in out1.lower() or "did not respond" in out1.lower()

    # The permit must have been freed in the surviving thread -> the next call completes.
    out2 = tool.invoke({"question": "Describe again", "image_paths": [img], "mode": "fast"})
    assert "second call ok" in out2.lower()
    release.set()


def test_vision_transient_error_still_retries_then_degrades(monkeypatch, tmp_path):
    """The wall-clock restructure must preserve the existing transient-blip backoff/retry."""
    img = _make_image(tmp_path / "scan.png")
    attempts = {"n": 0}

    def _resp():
        attempts["n"] += 1
        raise ConnectionError("tesla blip")

    fake = _FakeModel(_resp)
    monkeypatch.setattr(vision_tools, "build_vision_chat_model", lambda settings: fake)
    tool = vision_pkg.build_vision_tools(
        _settings(qwen_vlm_request_timeout_seconds=30), workspace_dir=str(tmp_path)
    )[0]
    out = tool.invoke({"question": "?", "image_paths": [img], "mode": "fast"})
    assert attempts["n"] == 3  # 3 attempts before degrading (unchanged behavior)
    assert "vision model unavailable" in out.lower()


def test_part_a_vision_two_pass_contract_replaces_per_image_loop():
    """Part A: the vision-reasoner must screen-first for >~3-4 images then bound deep reads,
    not loop reasoning-mode inspect_images per slice (the 26-min NPH defect), and the
    50-100 screen gate the medical case never crossed must be gone."""
    from ultra_deepagents.agent import VISION_DELEGATION_GUIDANCE, VISION_SUBAGENT

    sp = " ".join(VISION_SUBAGENT["system_prompt"].lower().split())
    assert "more than ~3-4 images/slices you must screen first" in sp
    assert "never loop deep inspect_images over a whole stack" in sp
    assert "trust the screen result for the rest" in sp
    # the old defects are gone
    assert "50-100" not in sp
    assert "loop — call inspect_images per crop/image" not in sp
    # adaptive, not a hard cap: deep-read count is the subagent's to choose
    assert "how many deep reads is yours to choose" in sp

    dg = " ".join(VISION_DELEGATION_GUIDANCE.lower().split())
    assert "screen the whole set in one pass first" in dg
    # a measurable question is the coordinator's to compute, vision corroborates
    assert "is yours to compute in the sandbox" in dg
    assert "corroborates the number, it does not produce it" in dg
