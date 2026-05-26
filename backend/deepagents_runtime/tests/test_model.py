from ultra_deepagents.config import RuntimeSettings
from ultra_deepagents.model import build_chat_model


def test_build_chat_model_uses_openai_compatible_base_url():
    settings = RuntimeSettings(
        openai_base_url="http://vrl-h200.ece.ucsb.edu:9393/v1",
        openai_model="deepseek_v4",
        openai_api_key="EMPTY",
        request_timeout_seconds=42.0,
        max_retries=0,
    )

    model = build_chat_model(settings)

    assert model.openai_api_base == "http://vrl-h200.ece.ucsb.edu:9393/v1"
    assert model.model_name == "deepseek_v4"
    assert model.openai_api_key.get_secret_value() == "EMPTY"
    assert model.request_timeout == 42.0
    assert model.max_retries == 0
