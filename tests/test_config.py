from mlflow_harness.config import DEFAULT_EXPERIMENT_NAME, DEFAULT_TRACKING_URI, get_settings


def test_get_settings_uses_dotenv(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "MLFLOW_TRACKING_URI=from-dotenv\n"
        "MLFLOW_EXPERIMENT_NAME=dotenv-exp\n"
        "MLFLOW_QUERY_TOKEN=dotenv-token\n"
    )

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_QUERY_TOKEN", raising=False)

    settings = get_settings(dotenv_path=str(dotenv_path))

    assert settings.tracking_uri == "from-dotenv"
    assert settings.experiment_name == "dotenv-exp"
    assert settings.query_token == "dotenv-token"


def test_get_settings_env_overrides_dotenv(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "MLFLOW_TRACKING_URI=from-dotenv\n"
        "MLFLOW_EXPERIMENT_NAME=dotenv-exp\n"
        "MLFLOW_QUERY_TOKEN=dotenv-token\n"
    )

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "from-env")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env-exp")
    monkeypatch.setenv("MLFLOW_QUERY_TOKEN", "env-token")

    settings = get_settings(dotenv_path=str(dotenv_path))

    assert settings.tracking_uri == "from-env"
    assert settings.experiment_name == "env-exp"
    assert settings.query_token == "env-token"


def test_get_settings_defaults_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    monkeypatch.delenv("MLFLOW_QUERY_TOKEN", raising=False)

    settings = get_settings(dotenv_path=str(tmp_path / "missing.env"))

    assert settings.tracking_uri == DEFAULT_TRACKING_URI
    assert settings.experiment_name == DEFAULT_EXPERIMENT_NAME
    assert settings.query_token is None
