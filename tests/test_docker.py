"""
Tests for Step 11: Docker containerization.

These tests validate the Docker configuration without building images:
  - Dockerfile syntax and required instructions
  - docker-compose.yml service definitions and dependency graph
  - Requirements split correctness
  - .dockerignore coverage for sensitive files
  - Entrypoint script MODEL_TYPE routing logic
"""

import os
import re
import yaml
import pytest

DOCKER_DIR = os.path.join(os.path.dirname(__file__), "..", "docker")
ROOT_DIR   = os.path.join(os.path.dirname(__file__), "..")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _compose() -> dict:
    with open(os.path.join(DOCKER_DIR, "docker-compose.yml")) as f:
        return yaml.safe_load(f)


# ── .dockerignore ─────────────────────────────────────────────────────────────

class TestDockerignore:
    def setup_method(self):
        self.content = _read(os.path.join(ROOT_DIR, ".dockerignore"))

    def test_excludes_git_directory(self):
        assert ".git" in self.content

    def test_excludes_env_files(self):
        assert ".env" in self.content

    def test_excludes_pycache(self):
        assert "__pycache__" in self.content

    def test_excludes_mlruns(self):
        assert "mlruns" in self.content

    def test_excludes_raw_csv_data(self):
        assert "data/raw/*.csv" in self.content or "data/" in self.content

    def test_excludes_test_directory(self):
        assert "tests/" in self.content

    def test_excludes_notebooks(self):
        assert "notebooks" in self.content or "*.ipynb" in self.content


# ── Dockerfile.serving ────────────────────────────────────────────────────────

class TestDockerfileServing:
    def setup_method(self):
        self.content = _read(os.path.join(DOCKER_DIR, "Dockerfile.serving"))

    def test_uses_multi_stage_build(self):
        assert self.content.count("FROM ") >= 2

    def test_builder_stage_named(self):
        assert "AS builder" in self.content

    def test_runtime_stage_named(self):
        assert "AS runtime" in self.content

    def test_uses_python_311(self):
        assert "python:3.11" in self.content

    def test_creates_non_root_user(self):
        assert "useradd" in self.content or "adduser" in self.content

    def test_runs_as_non_root_user(self):
        assert "USER appuser" in self.content or "USER " in self.content

    def test_has_healthcheck(self):
        assert "HEALTHCHECK" in self.content

    def test_exposes_port_8000(self):
        assert "EXPOSE 8000" in self.content

    def test_sets_pythonunbuffered(self):
        assert "PYTHONUNBUFFERED=1" in self.content

    def test_copies_src_not_root(self):
        # Should copy src/ not entire project root
        assert "COPY" in self.content
        lines = [l.strip() for l in self.content.splitlines() if l.strip().startswith("COPY")]
        src_copies = [l for l in lines if "src/" in l]
        assert len(src_copies) > 0

    def test_installs_from_serving_requirements(self):
        assert "serving.txt" in self.content or "requirements/serving" in self.content

    def test_removes_wheels_after_install(self):
        assert "rm -rf /wheels" in self.content


# ── Dockerfile.training ───────────────────────────────────────────────────────

class TestDockerfileTraining:
    def setup_method(self):
        self.content = _read(os.path.join(DOCKER_DIR, "Dockerfile.training"))

    def test_uses_multi_stage_build(self):
        assert self.content.count("FROM ") >= 2

    def test_creates_non_root_user(self):
        assert "useradd" in self.content or "adduser" in self.content

    def test_has_model_type_env(self):
        assert "MODEL_TYPE" in self.content

    def test_uses_entrypoint_script(self):
        assert "ENTRYPOINT" in self.content
        assert "entrypoint" in self.content.lower()

    def test_installs_from_training_requirements(self):
        assert "training.txt" in self.content or "requirements/training" in self.content


# ── Dockerfile.airflow ────────────────────────────────────────────────────────

class TestDockerfileAirflow:
    def setup_method(self):
        self.content = _read(os.path.join(DOCKER_DIR, "Dockerfile.airflow"))

    def test_extends_official_airflow_image(self):
        assert "apache/airflow" in self.content

    def test_installs_airflow_requirements(self):
        assert "airflow.txt" in self.content or "requirements/airflow" in self.content

    def test_sets_pythonpath(self):
        assert "PYTHONPATH" in self.content

    def test_copies_src_package(self):
        assert "src/" in self.content


# ── entrypoint.training.sh ────────────────────────────────────────────────────

class TestTrainingEntrypoint:
    def setup_method(self):
        self.content = _read(os.path.join(DOCKER_DIR, "entrypoint.training.sh"))

    def test_routes_lstm(self):
        assert "train_lstm" in self.content

    def test_routes_tft(self):
        assert "train_tft" in self.content

    def test_routes_timesfm(self):
        assert "train_timesfm" in self.content

    def test_routes_arm(self):
        assert "train_arm" in self.content

    def test_routes_all(self):
        assert "all)" in self.content or '"all"' in self.content

    def test_exits_on_unknown_model_type(self):
        assert "exit 1" in self.content

    def test_uses_set_e_for_error_safety(self):
        assert "set -e" in self.content


# ── docker-compose.yml ────────────────────────────────────────────────────────

class TestDockerCompose:
    def setup_method(self):
        self.compose = _compose()
        self.services = self.compose.get("services", {})

    def test_has_required_services(self):
        required = {"postgres", "mlflow", "redis", "api", "prometheus", "grafana"}
        missing  = required - set(self.services)
        assert not missing, f"Missing services: {missing}"

    def test_has_airflow_webserver(self):
        assert "airflow-webserver" in self.services

    def test_has_airflow_scheduler(self):
        assert "airflow-scheduler" in self.services

    def test_has_airflow_init(self):
        assert "airflow-init" in self.services

    def test_postgres_has_healthcheck(self):
        hc = self.services["postgres"].get("healthcheck")
        assert hc is not None
        assert "pg_isready" in str(hc)

    def test_redis_has_healthcheck(self):
        hc = self.services["redis"].get("healthcheck")
        assert hc is not None
        assert "redis-cli" in str(hc) or "ping" in str(hc)

    def test_api_has_healthcheck(self):
        hc = self.services["api"].get("healthcheck")
        assert hc is not None

    def test_mlflow_has_healthcheck(self):
        hc = self.services["mlflow"].get("healthcheck")
        assert hc is not None

    def test_api_uses_redis_url_not_redis_host(self):
        env = self.services["api"].get("environment", {})
        env_str = str(env)
        assert "REDIS_URL" in env_str
        # Should NOT use the old REDIS_HOST key
        assert "REDIS_HOST" not in env_str

    def test_api_depends_on_mlflow_and_redis(self):
        deps = self.services["api"].get("depends_on", {})
        assert "mlflow" in deps or "mlflow" in str(deps)
        assert "redis"  in deps or "redis"  in str(deps)

    def test_airflow_scheduler_depends_on_init(self):
        deps = self.services["airflow-scheduler"].get("depends_on", {})
        assert "airflow-init" in deps or "airflow-init" in str(deps)

    def test_airflow_webserver_depends_on_init(self):
        deps = self.services["airflow-webserver"].get("depends_on", {})
        assert "airflow-init" in deps or "airflow-init" in str(deps)

    def test_has_named_networks(self):
        networks = self.compose.get("networks", {})
        assert "backend" in networks
        assert "frontend" in networks

    def test_backend_network_is_internal(self):
        backend = self.compose["networks"]["backend"]
        assert backend.get("internal") is True

    def test_api_on_both_networks(self):
        api_nets = self.services["api"].get("networks", [])
        assert "backend"  in api_nets or "backend"  in str(api_nets)
        assert "frontend" in api_nets or "frontend" in str(api_nets)

    def test_postgres_only_on_backend(self):
        pg_nets = self.services["postgres"].get("networks", [])
        nets = pg_nets if isinstance(pg_nets, list) else list(pg_nets.keys())
        assert "backend" in nets
        assert "frontend" not in nets

    def test_has_persistent_volumes(self):
        volumes = self.compose.get("volumes", {})
        for vol in ("postgres_data", "mlflow_artifacts", "grafana_data", "prometheus_data"):
            assert vol in volumes, f"Missing volume: {vol}"

    def test_trainer_service_has_profile(self):
        profiles = self.services.get("trainer", {}).get("profiles", [])
        assert len(profiles) > 0, "trainer should be profile-gated so it doesn't start by default"

    def test_grafana_depends_on_prometheus(self):
        deps = self.services["grafana"].get("depends_on", {})
        assert "prometheus" in deps or "prometheus" in str(deps)

    def test_services_have_restart_policy(self):
        # Core always-on services should restart automatically
        for name in ("postgres", "mlflow", "redis", "api"):
            restart = self.services[name].get("restart", "")
            assert restart, f"{name} has no restart policy"

    def test_mlflow_tracking_uri_points_to_postgres(self):
        cmd = str(self.services["mlflow"].get("command", ""))
        assert "postgresql" in cmd or "postgres" in cmd


# ── requirements split ────────────────────────────────────────────────────────

class TestRequirementsSplit:
    def _reqs(self, name):
        path = os.path.join(ROOT_DIR, "requirements", f"{name}.txt")
        return _read(path)

    def test_base_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT_DIR, "requirements", "base.txt"))

    def test_serving_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT_DIR, "requirements", "serving.txt"))

    def test_training_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT_DIR, "requirements", "training.txt"))

    def test_dev_txt_exists(self):
        assert os.path.exists(os.path.join(ROOT_DIR, "requirements", "dev.txt"))

    def test_serving_includes_base(self):
        assert "-r base.txt" in self._reqs("serving")

    def test_training_includes_base(self):
        assert "-r base.txt" in self._reqs("training")

    def test_serving_has_fastapi(self):
        assert "fastapi" in self._reqs("serving")

    def test_serving_has_redis(self):
        assert "redis" in self._reqs("serving")

    def test_training_has_mlflow(self):
        assert "mlflow" in self._reqs("training") or "mlflow" in self._reqs("base")

    def test_training_has_torch(self):
        assert "torch" in self._reqs("training")

    def test_dev_has_pytest(self):
        assert "pytest" in self._reqs("dev")

    def test_base_does_not_import_serving_packages(self):
        base = self._reqs("base")
        assert "fastapi" not in base
        assert "uvicorn" not in base
        assert "redis" not in base

    def test_base_does_not_import_heavy_ml(self):
        base = self._reqs("base")
        assert "torch" not in base
        assert "tensorflow" not in base
