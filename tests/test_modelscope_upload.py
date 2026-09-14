from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prometheus.workflows.upload_dataset import _exact_pattern, run_upload


CONFIG_DIR = Path(__file__).resolve().parents[1] / "prometheus" / "configs"


@pytest.fixture
def config(tmp_path):
    cfg = OmegaConf.load(CONFIG_DIR / "upload_dataset.yaml")
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "description.txt").write_text("dataset", encoding="utf-8")
    cfg.local_dir = str(root)
    cfg.repo_id = "owner/dataset"
    cfg.run.output_dir = str(tmp_path / "upload-log")
    return cfg


def saved_episode(root: Path, status="saved"):
    episode = root / "0000"
    (episode / "manifests").mkdir(parents=True)
    (episode / "camera").mkdir()
    (episode / "camera" / "base_rgb.mp4").write_bytes(b"fixture")
    for group in ("camera", "tactile", "robot", "event"):
        payload = {} if group == "tactile" else {"recorder": {"status": status, "failure": ""}}
        (episode / "manifests" / f"{group}.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode


@pytest.fixture
def sdk(monkeypatch):
    module = ModuleType("modelscope_hub")
    errors = ModuleType("modelscope_hub.errors")
    errors.NotExistError = type("NotExistError", (Exception,), {})
    api = Mock()
    api.get_repo.return_value = SimpleNamespace(visibility="private")
    api.create_repo.return_value = SimpleNamespace(visibility="private")
    api.upload_folder.return_value = {"commit_id": "test-commit"}
    module.HubApi = Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "modelscope_hub", module)
    monkeypatch.setitem(sys.modules, "modelscope_hub.errors", errors)
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "unit-test-secret")
    return SimpleNamespace(factory=module.HubApi, api=api, missing=errors.NotExistError)


def test_hydra_config_is_safe_by_default():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="upload_dataset")
    assert cfg.dry_run is True
    assert cfg.create_repo is False
    assert cfg.visibility == "private"
    assert OmegaConf.is_missing(cfg, "repo_id")
    assert OmegaConf.is_missing(cfg, "local_dir")
    assert cfg.license == "Apache-2.0"
    assert cfg.max_workers == 4


def test_preview_is_offline_and_accepts_disabled_streams(config, sdk):
    root = Path(config.local_dir)
    saved_episode(root)
    (root / ".env").write_text("private", encoding="utf-8")
    (root / "unfinished.tmp").write_text("temporary", encoding="utf-8")
    result = run_upload(config)
    assert result["status"] == "preview"
    assert result["saved_episodes"] == 1
    assert result["file_count"] == 6
    sdk.factory.assert_not_called()
    files = json.loads((Path(config.run.output_dir) / "files.json").read_text())
    assert not any(item["path"] in {".env", "unfinished.tmp"} for item in files)


@pytest.mark.parametrize("status", ["recording", "failed", "rejected"])
def test_unsaved_episode_refuses_upload(config, sdk, status):
    saved_episode(Path(config.local_dir), status=status)
    config.dry_run = False
    config.repo_id = "owner/dataset"
    with pytest.raises(ValueError, match="尚未正常保存"):
        run_upload(config)
    sdk.factory.assert_not_called()


def test_incomplete_manifest_set_is_rejected(config):
    episode = saved_episode(Path(config.local_dir))
    (episode / "manifests" / "event.json").unlink()
    with pytest.raises(ValueError, match="缺少保存清单"):
        run_upload(config)


def test_symlink_is_not_uploaded(config, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (Path(config.local_dir) / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="符号链接"):
        run_upload(config)


def test_action_file_requires_its_saved_manifest(config):
    episode = saved_episode(Path(config.local_dir))
    (episode / "action").mkdir()
    (episode / "action" / "executed_action_dict.pkl").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="action.json"):
        run_upload(config)


def test_upload_preserves_paths_and_never_deletes_remote(config, sdk):
    config.dry_run = False
    config.repo_id = "owner/dataset"
    config.path_in_repo = "raw/egg"
    result = run_upload(config)
    sdk.factory.assert_called_once_with(token="unit-test-secret", endpoint="https://modelscope.cn")
    sdk.api.get_repo.assert_called_once_with("owner/dataset", repo_type="dataset")
    sdk.api.create_repo.assert_not_called()
    args = sdk.api.upload_folder.call_args.kwargs
    assert args["repo_type"] == "dataset"
    assert args["path_in_repo"] == "raw/egg"
    assert args["folder_path"] == config.local_dir
    assert args["allow_patterns"] == ["description.txt"]
    assert args["use_cache"] is True
    assert args["sync_remote_repo"] is False
    assert result["status"] == "completed"
    assert result["commits"] == [{"commit_id": "test-commit"}]
    assert "unit-test-secret" not in (Path(config.run.output_dir) / "result.json").read_text()


def test_missing_repo_requires_explicit_creation(config, sdk):
    config.dry_run = False
    config.repo_id = "owner/dataset"
    sdk.api.get_repo.side_effect = sdk.missing("not found")
    with pytest.raises(RuntimeError, match="create_repo=true"):
        run_upload(config)
    sdk.api.create_repo.assert_not_called()
    sdk.api.upload_folder.assert_not_called()
    config.create_repo = True
    assert run_upload(config)["status"] == "completed"
    sdk.api.create_repo.assert_called_once_with(
        "owner/dataset", repo_type="dataset", visibility="private", license=config.license,
    )


def test_upload_failure_is_recorded_without_token(config, sdk):
    config.dry_run = False
    config.repo_id = "owner/dataset"
    sdk.api.upload_folder.side_effect = RuntimeError("failed unit-test-secret")
    with pytest.raises(RuntimeError, match=r"failed \[REDACTED\]"):
        run_upload(config)
    result = json.loads((Path(config.run.output_dir) / "result.json").read_text())
    assert result["status"] == "failed"
    assert "unit-test-secret" not in result["error"]


def test_changes_during_upload_are_not_reported_as_success(config, sdk):
    config.dry_run = False
    config.repo_id = "owner/dataset"
    sdk.api.upload_folder.side_effect = lambda **kwargs: (Path(config.local_dir) / "new.txt").touch()
    with pytest.raises(RuntimeError, match="上传期间本地数据发生变化"):
        run_upload(config)
    result = json.loads((Path(config.run.output_dir) / "result.json").read_text())
    assert result["status"] == "failed"


def test_manifest_glob_metacharacters_are_literal():
    name = "folder/file[1]*?.mp4"
    pattern = _exact_pattern(name)
    assert fnmatch.fnmatchcase(name, pattern)
    assert not fnmatch.fnmatchcase("folder/file1abc.mp4", pattern)


@pytest.mark.parametrize("prefix", ["/outside", "../outside", "raw/../outside", "raw\\outside"])
def test_unsafe_remote_paths_are_rejected(config, prefix):
    config.path_in_repo = prefix
    with pytest.raises(ValueError, match="仓库内相对路径"):
        run_upload(config)


def test_upload_logs_cannot_be_inside_dataset(config):
    config.run.output_dir = str(Path(config.local_dir) / "logs")
    with pytest.raises(ValueError, match="上传日志目录不能"):
        run_upload(config)
