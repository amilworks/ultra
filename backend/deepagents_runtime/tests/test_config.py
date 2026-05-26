from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.context import AgentRunContext


def test_runtime_settings_load_vllm_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vrl-h200.ece.ucsb.edu:9393/v1")
    monkeypatch.setenv("OPENAI_MODEL", "deepseek_v4")

    settings = RuntimeSettings.from_env()

    assert settings.openai_base_url == "http://vrl-h200.ece.ucsb.edu:9393/v1"
    assert settings.openai_model == "deepseek_v4"
    assert settings.openai_api_key == "EMPTY"
    assert settings.sandbox_network == "none"


def test_agent_run_context_payload_is_snake_case_and_scoped():
    context = AgentRunContext(
        assistant_id="ultra-research",
        org_id="allen",
        user_id="researcher-1",
        project_id="project-1",
        thread_id="thread-1",
        run_id="run-1",
        selected_file_ids=("file-1",),
        allowed_tool_packs=("workspace", "code"),
    )

    payload = context.to_payload()

    assert payload["run_id"] == "run-1"
    assert payload["selected_file_ids"] == ["file-1"]
    assert payload["allowed_tool_packs"] == ["workspace", "code"]
