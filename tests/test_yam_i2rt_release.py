import hashlib
import json
from pathlib import Path

import pytest

from prometheus.nodes.robots.yam import _validate_i2rt_api
from scripts.prepare_yam_i2rt import RELEASE, prepare


def test_i2rt_patch_matches_release_lock():
    lock = json.loads((RELEASE / "lock.json").read_text())
    assert len(lock["base_commit"]) == 40
    assert (
        hashlib.sha256((RELEASE / "compat.patch").read_bytes()).hexdigest()
        == lock["patch_sha256"]
    )


def test_dependency_preparation_refuses_tampered_patch(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "lock.json").write_text((RELEASE / "lock.json").read_text())
    (release / "compat.patch").write_bytes(b"modified patch")
    with pytest.raises(ValueError, match="checksum"):
        prepare(Path("/nonexistent-source"), tmp_path / "output", release=release)
    assert not (tmp_path / "output").exists()


def test_stock_i2rt_is_rejected_before_any_can_is_opened(monkeypatch):
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    monkeypatch.delattr(MotorChainRobot, "get_runtime_status")
    with pytest.raises(RuntimeError, match="Incompatible I2RT"):
        _validate_i2rt_api(drag=True)
