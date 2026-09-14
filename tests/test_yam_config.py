from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from prometheus.nodes.robots.yam import resolve_drag_robot, resolve_robot

CONFIG_DIR = Path(__file__).resolve().parents[1] / "prometheus" / "configs"


def test_collect_selects_yam_with_one_hydra_override():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect", overrides=["robot=yam"])
    data = OmegaConf.to_container(cfg, resolve=True)

    assert data["robot"]["id"] == "yam"
    assert data["robot"]["robot"]["leader_can"] == {
        "left": "can_leader_l",
        "right": "can_leader_r",
    }
    assert data["robot"]["robot"]["follower_can"] == {
        "left": "can_follower_l",
        "right": "can_follower_r",
    }

    resolved = resolve_robot(**data["robot"])
    assert resolved["type"] == "yam"
    assert (
        resolved["driver"]["_target_"]
        == "prometheus.nodes.robots.yam.YamBimanualDriver"
    )
    assert resolved["driver"]["follower_arm"] == "yam_ultra_2"
    assert resolved["driver"]["leader_arm"] == "yam"
    assert resolved["owner"]["action_dims"] == {"abs_qpos": 14}
    assert set(resolved["sensors"]) == {"realsense"}
    assert resolved["sensors"]["realsense"]["cameras"] == {
            "base_0": "260322278344",
        "left_wrist_0": "260322279589",
        "right_wrist_0": "260322270559",
    }
    assert resolved["sensors"]["realsense"]["fps"] == 30
    assert resolved["sensors"]["realsense"]["publish_hz"] == 30.0
    assert data["data"]["recording"]["fps"] == 30
    assert data["data"]["recording"]["camera_encoder"] == "libx265"
    assert data["event"]["tty"]["confirm_presses"] == 1
    assert "robot_state" in resolved["streams"]
    assert resolved["sensors"]["realsense"]["enable_infra"] is False
    assert not any("infra" in name for name in resolved["streams"])
    assert not any("tactile" in name for name in resolved["streams"])


def test_collect_selects_follower_only_yam_drag_with_one_hydra_override():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect", overrides=["robot=yam_drag"])
    data = OmegaConf.to_container(cfg, resolve=True)

    assert data["robot"]["id"] == "yam_drag"
    assert data["robot"]["robot"]["follower_can"] == {
        "left": "can_follower_l",
        "right": "can_follower_r",
    }
    assert "leader_can" not in data["robot"]["robot"]

    resolved = resolve_drag_robot(**data["robot"])
    assert resolved["type"] == "yam_drag"
    assert (
        resolved["driver"]["_target_"]
        == "prometheus.nodes.robots.yam.YamDragDriver"
    )
    assert resolved["driver"]["follower_arm"] == "yam_ultra_2"
    assert "leader_left_can" not in resolved["driver"]
    assert resolved["driver"]["ee_mass"] == data["robot"]["robot"]["ee_mass"]
    assert (
        resolved["driver"]["gripper_motion_assist_enabled"]
        == data["robot"]["robot"]["gripper_motion_assist_enabled"]
    )
    assert (
        resolved["driver"]["gripper_motion_assist_torque_nm"]
        == data["robot"]["robot"]["gripper_motion_assist_torque_nm"]
    )
    assert (
        resolved["driver"]["gripper_motion_assist_velocity_threshold_m_s"]
        == data["robot"]["robot"]["gripper_motion_assist_velocity_threshold_m_s"]
    )
    assert (
        resolved["driver"]["gripper_motion_assist_direction"]
        == data["robot"]["robot"]["gripper_motion_assist_direction"]
    )
    assert resolved["driver"]["return_to_start_enabled"] is True
    assert resolved["driver"]["return_to_start_duration_s"] == 8.0
    assert resolved["owner"]["action_dims"] == {"abs_qpos": 14}


def test_yam_drag_accepts_an_end_effector_mass_override():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect", overrides=["robot=yam_drag", "robot.robot.ee_mass=0.65"])
    resolved = resolve_drag_robot(**OmegaConf.to_container(cfg.robot, resolve=True))
    assert resolved["driver"]["ee_mass"] == 0.65
    assert resolved["streams"]["robot_state"]["topic"] == "/prometheus/robots/yam_drag/state"


def test_yam_drag_hydra_tuning_reaches_both_followers(monkeypatch):
    import i2rt.robots.get_robot as i2rt_factory

    calls = []

    def capture_robot(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(i2rt_factory, "get_yam_robot", capture_robot)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="collect", overrides=[
            "robot=yam_drag",
            "robot.robot.use_coulomb_friction=true",
            "robot.robot.coulomb_friction=[0.1,0.1,0.1,0.1,0.02,0.02]",
            "robot.robot.grav_comp_kd=[0.1,0.1,0.1,0.2,0.05,0.05]",
            "robot.robot.gravity_comp_factor=[1.0,1.2,1.1,1.1,1.2,1.0]",
            "robot.robot.gripper_motion_assist_torque_nm=0.05",
            "robot.robot.gripper_motion_assist_velocity_threshold_m_s=0.002",
        ])
    resolved = resolve_drag_robot(**OmegaConf.to_container(cfg.robot, resolve=True))
    driver = instantiate(resolved["driver"])
    for side in driver.sides:
        driver._make_follower(side)

    assert {call["channel"] for call in calls} == {"can_follower_l", "can_follower_r"}
    for call in calls:
        assert call["zero_gravity_mode"] is True
        assert call["use_coulomb_friction"] is True
        assert call["coulomb_friction"] == [0.1, 0.1, 0.1, 0.1, 0.02, 0.02]
        assert call["grav_comp_kd"] == [0.1, 0.1, 0.1, 0.2, 0.05, 0.05]
        assert call["gravity_comp_factor"] == [1.0, 1.2, 1.1, 1.1, 1.2, 1.0]
    assert driver.status()["drag_tuning"]["use_coulomb_friction"] is True
