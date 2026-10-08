import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _deploy_module():
    spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_public_space_gets_code_only(tmp_path):
    """The Space repo is public: no keys, datasets, caches or model files may ever be staged."""
    _deploy_module().stage_space(tmp_path)
    files = [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()]
    forbidden = (".env", ".pkl", ".parquet", ".jsonl", ".npz", ".npy", ".csv", ".db")
    assert not [f for f in files if f.endswith(forbidden) or f.startswith(("data/", "artifacts/")) or "egg-info" in f]
    assert {"Dockerfile", "README.md", "start.py", "apps/ui/index.html", "apps/api/main.py"} <= set(files)
    assert not any(f.startswith("apps/web/") for f in files)  # the Streamlit research UI stays private
    assert "sdk: docker" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_cloud_run_stage_has_code_and_model_but_no_keys_or_datasets(tmp_path):
    spec = importlib.util.spec_from_file_location("deploy_cloudrun", ROOT / "scripts" / "deploy_cloudrun.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = tmp_path / "model"
    model.mkdir()
    (model / "engine.pkl").write_bytes(b"x")
    (model / "manifest.json").write_text("{}")
    out = tmp_path / "stage"
    out.mkdir()
    mod.stage(out, "vX", model)
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"Dockerfile", "artifacts/models/PRODUCTION", "artifacts/models/vX/engine.pkl", "apps/ui/index.html"} <= files
    assert not [f for f in files if f.endswith((".env", ".parquet", ".jsonl", ".csv")) or f.startswith("data/")]
    assert (out / "artifacts/models/PRODUCTION").read_text() == "vX"
