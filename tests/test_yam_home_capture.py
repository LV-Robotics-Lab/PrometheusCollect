from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from prometheus.workflows.record_drag import _capture_home_snapshot


class _Driver:
    def __init__(self) -> None:
        self.read_count = 0

    def read_state(self) -> SimpleNamespace:
        self.read_count += 1
        return SimpleNamespace(
            qpos=np.arange(14, dtype=np.float64),
            leader_qpos=np.arange(100, 114, dtype=np.float64),
        )


class _Hardware:
    def __init__(self) -> None:
        self.robot_driver = _Driver()

    def robot_driver_status(self) -> dict[str, object]:
        return {
            "robot_id": "yam",
            "leader_can": {"left": "can_leader_l", "right": "can_leader_r"},
            "follower_can": {"left": "can_follower_l", "right": "can_follower_r"},
        }


class _DragDriver:
    def read_state(self) -> SimpleNamespace:
        return SimpleNamespace(qpos=np.arange(14, dtype=np.float64))


class _DragHardware:
    def __init__(self) -> None:
        self.robot_driver = _DragDriver()

    def robot_driver_status(self) -> dict[str, object]:
        return {
            "robot_id": "yam_drag",
            "follower_can": {"left": "can_follower_l", "right": "can_follower_r"},
        }


def test_capture_yam_home_snapshot_is_read_only(tmp_path):
    hardware = _Hardware()
    path = _capture_home_snapshot(
        hardware,
        {"home_capture": {"enabled": True, "path": str(tmp_path / "yam_home.json")}},
    )

    payload = json.loads(path.read_text())
    assert hardware.robot_driver.read_count == 1
    assert payload["leader"]["left"] == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    assert payload["follower"]["right"] == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    assert payload["can_interfaces"]["leader"]["left"] == "can_leader_l"


def test_capture_yam_drag_home_snapshot_has_only_follower_state(tmp_path):
    path = _capture_home_snapshot(
        _DragHardware(),
        {"home_capture": {"enabled": True, "path": str(tmp_path / "yam_drag_home.json")}},
    )

    payload = json.loads(path.read_text())

    assert payload["robot_id"] == "yam_drag"
    assert payload["follower"]["left"] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert payload["can_interfaces"]["follower"]["right"] == "can_follower_r"
    assert "leader" not in payload
    assert "leader" not in payload["can_interfaces"]
