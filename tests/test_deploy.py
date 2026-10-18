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


def test_cloud_run_stage_has_code_and_model_but_no_keys_or_datasets(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("deploy_cloudrun", ROOT / "scripts" / "deploy_cloudrun.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The TMDB/IGDB match table (catalog franchises) goes next to the model for the scheduled refresh job;
    # it is the only table allowed in the (private) image. Nothing from data/ is.
    from types import SimpleNamespace

    metadata = tmp_path / "metadata.parquet"
    metadata.write_bytes(b"x")
    monkeypatch.setattr(mod, "get_settings", lambda: SimpleNamespace(data=SimpleNamespace(external_metadata=metadata)))
    model = tmp_path / "model"
    model.mkdir()
    (model / "engine.pkl").write_bytes(b"x")
    (model / "manifest.json").write_text("{}")
    out = tmp_path / "stage"
    out.mkdir()
    mod.stage(out, "vX", model, portable=False)
    files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"Dockerfile", "artifacts/models/PRODUCTION", "artifacts/models/vX/engine.pkl", "apps/ui/index.html"} <= files
    assert "artifacts/models/vX/metadata.parquet" in files
    assert not [f for f in files if f.endswith((".env", ".parquet", ".jsonl", ".csv")) or f.startswith("data/")
                if f != "artifacts/models/vX/metadata.parquet"]
    assert (out / "artifacts/models/PRODUCTION").read_text() == "vX"


def test_exported_engine_has_no_filesystem_paths(trained, tmp_path):
    """A pickled WindowsPath crashed the Linux container; export_portable must strip such state."""
    import pickle

    from crossverse.retrieval.semantic import SemanticRetriever
    from crossverse.serving.engine import CrossVerseEngine, _find_paths

    engine, _, _ = trained
    sem = SemanticRetriever(cache_dir=tmp_path)  # holds a Path, like a trained v11+ engine
    sem.embeddings_ = engine.content.embeddings_
    engine.generator.retrievers["semantic"] = sem
    try:
        assert _find_paths(engine) == []  # __getstate__ already drops it
        engine.save(tmp_path / "e.pkl")
        CrossVerseEngine.export_portable(tmp_path / "e.pkl", tmp_path / "portable.pkl")
        with open(tmp_path / "portable.pkl", "rb") as fh:
            assert "Path" not in str(pickle.load(fh).generator.retrievers["semantic"].cache_dir)
    finally:
        del engine.generator.retrievers["semantic"]


def test_refresh_job_dry_run_changes_nothing_and_never_prints_keys(monkeypatch, capsys):
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("deploy_refresh_job", ROOT / "scripts" / "deploy_refresh_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    keys = {"TMDB_TOKEN": "val-zq9-1", "IGDB_CLIENT_ID": "val-zq9-2", "IGDB_CLIENT_SECRET": "val-zq9-3"}
    monkeypatch.setattr(mod, "load_env", lambda: keys)
    answers = {"config": "my-proj", "projects": "123", "run": "europe-west1-docker.pkg.dev/my-proj/r/crossverse@sha"}
    monkeypatch.setattr(mod, "out", lambda *a: answers[a[0]])

    def no_run(*a, **kw):
        raise AssertionError("a dry run must not call gcloud")

    monkeypatch.setattr(mod.subprocess, "run", no_run)
    assert mod.main([]) == 0
    printed = capsys.readouterr().out
    assert "val-zq9" not in printed  # key values go through stdin only
    assert "--image europe-west1-docker.pkg.dev/my-proj/r/crossverse@sha" in printed  # the service's own image
    assert "refresh-new-releases,--out,/mnt/new-releases" in printed
    assert "'--schedule=0 5 * * 1,4'" in printed
    assert "bucket=my-proj-crossverse-new-releases,readonly=true" in printed  # the service only reads
    assert "--public-access-prevention" in printed
    assert "Nothing changed" in printed
    sys.modules.pop("deploy_cloudrun", None)
