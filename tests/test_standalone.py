from __future__ import annotations

import importlib
import json
import subprocess
import sys

from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
import pytest

from prometheus.workflows.record_drag import run_from_config


@pytest.mark.parametrize("robot", ["yam", "yam_drag", "x5_1", "x5_2"])
def test_packaged_collection_config_resolves_without_model_components(robot, monkeypatch, tmp_path):
    monkeypatch.setenv("PROMETHEUS_DATA_ROOT", str(tmp_path / "data"))
    with initialize_config_module(version_base=None, config_module="prometheus.configs"):
        cfg = compose(config_name="collect", overrides=[f"robot={robot}"])
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["user"]["collect_root"] == str(tmp_path / "data")
    assert resolved["workflow"]["home_capture"]["path"] == str(tmp_path / "data/yam_home.json")
    assert not {"policy", "scheduler", "dagger"}.intersection(resolved)
    module_name, function = resolved["robot"]["_target_"].rsplit(".", 1)
    resolver = getattr(importlib.import_module(module_name), function)
    assert resolver(**resolved["robot"])["streams"]


def test_installed_modules_import_without_original_repo_or_training_packages(tmp_path):
    # -I removes the source directory and PYTHONPATH from import resolution.
    # This exercises an installed wheel/editable package from a different cwd.
    code = """
import importlib
import importlib.abc
import json
import pkgutil
import sys

blocked = ('prometheus.policy', 'prometheus.dagger', 'prometheus.runtime',
           'prometheus.transport', 'prometheus_env', 'torch', 'transformers')
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, RejectTraining())
import prometheus
for item in pkgutil.walk_packages(prometheus.__path__, 'prometheus.'):
    importlib.import_module(item.name)
print(json.dumps({'file': prometheus.__file__}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    loaded = json.loads(result.stdout)
    assert "PrometheusV2" not in loaded["file"]


def test_packaged_noop_workflow_saves_episode_without_test_targets(tmp_path):
    with initialize_config_module(version_base=None, config_module="prometheus.configs"):
        cfg = compose(config_name="collect_noop", overrides=[f"run.output_dir={tmp_path}"])
    assert "tests." not in OmegaConf.to_yaml(cfg)
    assert run_from_config(cfg) == 0
    manifest = json.loads((tmp_path / "workflow_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["episodes_saved"] == 1
    assert (tmp_path / "episodes/0000/event/event_dict.pkl").is_file()


def test_upload_has_no_preselected_dataset_or_account():
    with initialize_config_module(version_base=None, config_module="prometheus.configs"):
        cfg = compose(config_name="upload_dataset")
    assert OmegaConf.is_missing(cfg, "local_dir")
    assert OmegaConf.is_missing(cfg, "repo_id")
    assert cfg.dry_run is True
