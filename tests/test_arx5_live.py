from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from prometheus.nodes.robots.arx5 import Arx5Driver
from prometheus.utils.workflow import instantiate_config


def make_driver(config_name: str) -> Arx5Driver:
    cfg = _load_robot_config(config_name)
    driver_cfg = dict(instantiate_config(cfg, "robot", _recursive_=False)["driver"])
    driver_cfg.pop("_target_", None)
    return Arx5Driver(**driver_cfg)


def read_once(config_name: str) -> dict[str, Any]:
    driver = make_driver(config_name)
    driver.connect()
    try:
        state = driver.read_state()
        status = driver.status()
        print(f"status: {status}")
        print(f"joint_names: {state.joint_names}")
        print(f"qpos: {np.asarray(state.qpos, dtype=np.float32).round(5).tolist()}")
        print(f"eef_pose_6d: {np.asarray(state.eef_pose_6d, dtype=np.float32).round(5).tolist()}")
        return {
            "status": status,
            "joint_count": len(state.joint_names),
            "qpos_shape": tuple(np.asarray(state.qpos).shape),
            "eef_shape": tuple(np.asarray(state.eef_pose_6d).shape),
        }
    finally:
        driver.close()


def test_live_arx5_read_state():
    if os.environ.get("PROMETHEUS_LIVE_ARX5") != "1":
        pytest.skip("set PROMETHEUS_LIVE_ARX5=1 to connect to physical ARX5 hardware")

    result = read_once(os.environ.get("PROMETHEUS_ROBOT_CONFIG", "x5_2"))
    assert result["joint_count"] == 14
    assert result["qpos_shape"] == (14,)
    assert result["eef_shape"] == (12,)


def main() -> int:
    if os.environ.get("PROMETHEUS_LIVE_ARX5") != "1":
        print("refusing to connect to ARX5 without PROMETHEUS_LIVE_ARX5=1")
        return 2
    config_name = os.environ.get("PROMETHEUS_ROBOT_CONFIG", "x5_2")
    read_once(config_name)
    return 0


def _load_robot_config(config_name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "prometheus" / "configs" / "robot" / f"{config_name}.yaml"
    root = OmegaConf.create({"robot": OmegaConf.load(path)})
    return OmegaConf.to_container(root.robot, resolve=True)


if __name__ == "__main__":
    raise SystemExit(main())
