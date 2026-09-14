from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from prometheus.nodes.sensors.realsense import RealsenseSensorNode
from prometheus.utils.workflow import instantiate_config


def make_sensor(config_name: str) -> RealsenseSensorNode:
    cfg = _load_robot_config(config_name)
    sensor_cfg = dict(instantiate_config(cfg, "robot", _recursive_=False)["sensors"]["realsense"])
    sensor_cfg.pop("_target_", None)
    if os.environ.get("PROMETHEUS_REALSENSE_CAMERAS"):
        sensor_cfg["cameras"] = json.loads(os.environ["PROMETHEUS_REALSENSE_CAMERAS"])
    return RealsenseSensorNode(**sensor_cfg)


def wait_once(config_name: str, timeout_s: float) -> dict[str, Any]:
    sensor = make_sensor(config_name)
    sensor.start()
    try:
        sensor.wait_ready(timeout_s=timeout_s)
        status = sensor.status()
        print(f"status: {status}")
        return status
    finally:
        sensor.close()


def test_live_realsense_streams_ready():
    if os.environ.get("PROMETHEUS_LIVE_REALSENSE") != "1":
        pytest.skip("set PROMETHEUS_LIVE_REALSENSE=1 to stream physical RealSense cameras")

    status = wait_once(
        os.environ.get("PROMETHEUS_REALSENSE_CONFIG", "yam"),
        float(os.environ.get("PROMETHEUS_REALSENSE_TIMEOUT_S", "10.0")),
    )
    assert status["ready"] is True


def main() -> int:
    if os.environ.get("PROMETHEUS_LIVE_REALSENSE") != "1":
        print("refusing to stream RealSense without PROMETHEUS_LIVE_REALSENSE=1")
        return 2
    wait_once(
        os.environ.get("PROMETHEUS_REALSENSE_CONFIG", "yam"),
        float(os.environ.get("PROMETHEUS_REALSENSE_TIMEOUT_S", "10.0")),
    )
    return 0


def _load_robot_config(config_name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "prometheus" / "configs" / "robot" / f"{config_name}.yaml"
    root = OmegaConf.create({"robot": OmegaConf.load(path)})
    return OmegaConf.to_container(root.robot, resolve=True)


if __name__ == "__main__":
    raise SystemExit(main())
