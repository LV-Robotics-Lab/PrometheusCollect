from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prometheus.utils.workflow import RunPaths, build_hardware_session


CONFIG_DIR = Path(__file__).resolve().parents[1] / "prometheus" / "configs"


def test_x5_2_uses_compact_robot_config(tmp_path):
    robot = _robot_config("x5_2")
    data = {"robot": robot, "hardware": _hardware_auto_config()}

    assert data["robot"]["robot"]["state_publish_hz"] == 120
    assert data["robot"]["cameras"]["capture_hz"] == 30
    assert data["robot"]["robot"]["can"] == {"left": "can1", "right": "can3"}
    assert "driver" not in data["robot"]
    assert "owner" not in data["robot"]
    assert "client" not in data["robot"]
    assert "topics" not in data["robot"]
    assert "streams" not in data["robot"]
    session = _session(data, tmp_path)
    assert session.sensors["realsense"]["enable_infra"] is True
    assert session.sensors["realsense"]["cameras"] == {
        "base_0": "260422273830",
        "left_wrist_0": "412622272007",
        "right_wrist_0": "412622271819",
    }
    assert session.sensors["realsense"]["camera_namespace"] == "/prometheus/sensors/cameras"
    assert session.streams["robot_state"].ros_domain_id == 21
    assert session.streams["right_wrist_0_infra2_info"].ros_domain_id == 22






def test_collect_uses_reliable_full_resolution_camera_without_info(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect", overrides=["robot=x5_1"])
    data = OmegaConf.to_container(cfg, resolve=True)

    assert data["robot"]["robot"]["state_publish_hz"] == 120
    assert data["robot"]["cameras"]["ir"] is True
    assert data["robot"]["cameras"]["prepub_resize"] == 1.0
    assert data["robot"]["cameras"]["qos"] == "reliable"
    assert data["robot"]["cameras"]["camera_info"] is False
    assert data["ros"]["localhost_only"] is True
    assert data["data"]["bridge"]["sensor_qos"] == "reliable"

    session = _session(data, tmp_path)
    assert session.robot["owner"]["hz"] == 120
    assert session.robot["driver"]["hz"] == 120
    assert session.sensors["realsense"]["qos"] == "reliable"
    assert session.sensors["realsense"]["camera_info"] is False
    assert "base_0_color" in session.streams
    assert "base_0_color_info" not in session.streams




def test_hardware_session_does_not_import_specific_hardware_types():
    source = (Path(__file__).resolve().parents[1] / "prometheus" / "sessions" / "hardware.py").read_text()

    assert "arx5" not in source
    assert "realsense" not in source
    assert "expand_robot_runtime" not in source


def test_noop_auto_resolves(tmp_path):
    data = {
        "robot": {
            "_target_": "prometheus.nodes.robots.noop.resolve_robot",
            "id": "noop",
            "control_domain_id": 0,
            "sensor_domain_id": 0,
            "action_space": "noop_action",
            "action_dim": 2,
            "sensors": {},
            "streams": {
                "robot_state": {
                    "topic": "/prometheus/noop/robot_state",
                    "msg_type": "prometheus.noop.RobotState",
                    "domain": "robot",
                    "ros_domain_id": 0,
                }
            },
        },
        "hardware": _hardware_auto_config(),
    }

    session = _session(data, tmp_path)

    assert session.robot["id"] == "noop"
    assert session.sensors == {}
    assert session.streams["robot_state"].ros_domain_id == 0


def _session(data, tmp_path):
    run = RunPaths(
        run_id="test",
        output_dir=tmp_path,
        runtime_dir=tmp_path / "runtime",
        manifest_path=tmp_path / "workflow_manifest.json",
    )
    return build_hardware_session(
        run=run,
        hardware_cfg=data["hardware"],
        robot_cfg=data["robot"],
    )


def _robot_config(name: str) -> dict:
    cfg = OmegaConf.load(CONFIG_DIR / "robot" / f"{name}.yaml")
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    return data


def _hardware_auto_config() -> dict:
    return {
        "name": "hardware",
        "mode": "owned",
        "sensors": "auto",
        "streams": "auto",
    }
