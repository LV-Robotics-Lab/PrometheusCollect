from __future__ import annotations

import json
import pickle
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from prometheus.data.recorder import DataRecorder
from prometheus.nodes.robots.yam import GRIPPER_WIDTH_M, YamBimanualDriver, YamDragDriver
from prometheus.workflows.record_drag import _auto_return_after_saved_drag_episode, _drain_robot_actions


class _FakeRobot:
    def __init__(self, *, role: str, side: str) -> None:
        self.role = role
        self.side = side
        offset = 0.1 if side == "left" else 0.2
        self.arm_pos = (
            np.arange(6, dtype=np.float64) * 0.01 + offset
            if role == "leader"
            else np.zeros(6)
        )
        self.gripper_pos = 0.25
        self.gripper_vel = 0.0
        self.commands: list[np.ndarray] = []
        self.gripper_assist_torques: list[float] = []
        self.closed = False
        self.idle_count = 0
        self.encoder = SimpleNamespace(
            position=0.25, velocity=0.0, io_inputs=np.array([0.0])
        )
        self.motor_chain = SimpleNamespace(
            get_same_bus_device_states=lambda: [self.encoder]
        )

    def get_observations(self) -> dict[str, np.ndarray]:
        result = {
            "timestamp": np.array([time.time()]),
            "joint_pos": self.arm_pos.copy(),
            "joint_vel": np.zeros(6),
            "joint_eff": np.ones(6),
        }
        if self.role == "follower":
            result.update(
                {
                    "gripper_pos": np.array([self.gripper_pos]),
                    "gripper_vel": np.array([self.gripper_vel]),
                    "gripper_eff": np.array([2.0]),
                }
            )
        return result

    def get_runtime_status(self) -> dict[str, Any]:
        return {"running": not self.closed, "state_age_s": 0.0, "stamp_ns": time.time_ns()}

    def get_robot_info(self) -> dict[str, np.ndarray]:
        return {"kp": np.ones(6), "joint_limits": np.tile([[-3.0, 3.0]], (6, 1))}

    def command_joint_pos(self, command: np.ndarray) -> None:
        values = np.asarray(command, dtype=np.float64).copy()
        self.commands.append(values)
        if self.role == "follower":
            self.arm_pos = values[:6].copy()
            self.gripper_pos = float(values[6])

    def update_kp_kd(self, *, kp: np.ndarray, kd: np.ndarray) -> None:
        self.kp = np.asarray(kp).copy()
        self.kd = np.asarray(kd).copy()

    def enter_gravity_comp_idle(self) -> None:
        self.idle_count += 1

    def set_gripper_assist_torque(self, torque_nm: float) -> None:
        self.gripper_assist_torques.append(float(torque_nm))

    def close(self) -> None:
        self.closed = True


def test_yam_driver_uses_follower_state_and_records_applied_leader_action():
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    driver = YamBimanualDriver(
        robot_factory=factory,
        control_hz=200.0,
        alignment_duration_s=0.0,
        require_enable_button=False,
    )
    driver.connect()
    _wait_until(lambda: bool(driver.drain_actions()), timeout_s=1.0)
    time.sleep(0.02)

    state = driver.read_state()
    actions = driver.drain_actions()

    assert state.qpos.shape == (14,)
    assert state.qvel.shape == (14,)
    assert state.effort.shape == (14,)
    assert state.leader_qpos.shape == (14,)
    assert state.qpos[6] == np.float32(0.75 * GRIPPER_WIDTH_M)
    assert state.qpos[13] == np.float32(0.75 * GRIPPER_WIDTH_M)
    assert state.leader_qpos[6] == np.float32(0.75 * GRIPPER_WIDTH_M)
    assert actions
    action = actions[-1]
    np.testing.assert_allclose(action["action"], state.leader_qpos, atol=1e-6)
    assert action["action_space"] == "abs_qpos"
    assert action["metadata"]["source"] == "yam_leader_follower"
    np.testing.assert_allclose(action["metadata"]["leader_qpos"], state.leader_qpos)
    assert robots[("follower", "left")].commands[-1][6] == pytest.approx(0.75)
    assert driver.status()["ready_for_recording"] is True

    driver.close()
    assert all(robot.closed for robot in robots.values())


def test_yam_external_action_converts_gripper_meters_to_normalized_command():
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    driver = YamBimanualDriver(
        robot_factory=factory,
        teleoperation_enabled=False,
        control_hz=200.0,
    )
    driver.connect()
    command = np.zeros(14, dtype=np.float32)
    command[6] = GRIPPER_WIDTH_M
    command[13] = GRIPPER_WIDTH_M / 2.0

    driver.send_action("abs_qpos", command)
    actions = driver.drain_actions()

    assert robots[("follower", "left")].commands[-1][6] == 1.0
    assert robots[("follower", "right")].commands[-1][6] == pytest.approx(0.5)
    np.testing.assert_allclose(actions[-1]["action"], command)
    driver.close()


def test_yam_drag_driver_uses_only_followers_and_records_observed_positions():
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    driver = YamDragDriver(robot_factory=factory, hz=200.0)
    driver.connect()
    _wait_until(lambda: bool(driver.drain_actions()), timeout_s=1.0)
    time.sleep(0.02)

    state = driver.read_state()
    actions = driver.drain_actions()

    assert set(robots) == {("follower", "left"), ("follower", "right")}
    assert all(robot.idle_count >= 1 for robot in robots.values())
    assert not any(robot.commands for robot in robots.values())
    assert state.qpos.shape == (14,)
    assert state.qpos[6] == np.float32(0.25 * GRIPPER_WIDTH_M)
    assert state.qpos[13] == np.float32(0.25 * GRIPPER_WIDTH_M)
    assert actions
    action = actions[-1]
    np.testing.assert_allclose(action["action"], state.qpos, atol=1e-6)
    assert action["action_space"] == "abs_qpos"
    assert action["metadata"]["source"] == "yam_drag_follower_state"
    assert action["metadata"]["action_semantics"] == "observed_follower_qpos"
    assert action["metadata"]["mode"] == "gravity_compensation"
    assert driver.status()["ready_for_recording"] is True
    assert driver.status()["supports_reset_home"] is True
    assert driver.status()["auto_return_after_episode"] is True

    with pytest.raises(RuntimeError, match="disables external position commands"):
        driver.send_action("abs_qpos", np.zeros(14, dtype=np.float32))

    driver.close()
    assert all(robot.closed for robot in robots.values())


def test_yam_drag_gripper_motion_assist_is_bidirectional_and_zero_at_rest():
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    driver = YamDragDriver(
        robot_factory=factory,
        hz=200.0,
        gripper_motion_assist_torque_nm=0.05,
        gripper_motion_assist_velocity_threshold_m_s=0.002,
    )
    driver.connect()
    left = robots[("follower", "left")]
    right = robots[("follower", "right")]
    # Fake gripper velocity is normalized; the driver converts it to m/s using
    # the 0.096 m configured stroke before applying the threshold.
    left.gripper_vel = 0.1
    right.gripper_vel = -0.1
    _wait_until(
        lambda: bool(left.gripper_assist_torques)
        and bool(right.gripper_assist_torques)
        and left.gripper_assist_torques[-1] > 0
        and right.gripper_assist_torques[-1] < 0,
        timeout_s=1.0,
    )
    active_assist = driver.status()["gripper_motion_assist"]
    assert active_assist["velocity_m_s"]["left"] == pytest.approx(0.1 * GRIPPER_WIDTH_M)
    assert active_assist["velocity_m_s"]["right"] == pytest.approx(-0.1 * GRIPPER_WIDTH_M)
    assert active_assist["commanded_torque_nm"] == {"left": 0.05, "right": -0.05}
    left.gripper_vel = 0.0
    right.gripper_vel = 0.0
    _wait_until(
        lambda: left.gripper_assist_torques[-1] == 0.0
        and right.gripper_assist_torques[-1] == 0.0,
        timeout_s=1.0,
    )
    assist = driver.status()["gripper_motion_assist"]
    assert assist["enabled"] is True
    assert assist["torque_nm"] == 0.05
    assert assist["active_only_while_moving"] is True
    driver.close()


def test_yam_drag_gripper_motion_assist_direction_can_reverse_motor_torque():
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    driver = YamDragDriver(
        robot_factory=factory,
        hz=200.0,
        gripper_motion_assist_torque_nm=0.05,
        gripper_motion_assist_direction=-1,
    )
    driver.connect()
    left = robots[("follower", "left")]
    left.gripper_vel = 0.1
    _wait_until(
        lambda: bool(left.gripper_assist_torques) and left.gripper_assist_torques[-1] < 0,
        timeout_s=1.0,
    )
    assert driver.status()["gripper_motion_assist"]["direction"] == -1
    driver.close()


def test_yam_drag_passes_end_effector_mass_to_both_followers():
    calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> _FakeRobot:
        calls.append(kwargs)
        return _FakeRobot(role=kwargs["role"], side=kwargs["side"])

    driver = YamDragDriver(robot_factory=factory, ee_mass=0.65)
    for side in driver.sides:
        driver._make_follower(side)

    assert len(calls) == 2
    assert all(call["ee_mass"] == 0.65 for call in calls)
    assert driver.status()["ee_mass"] == 0.65


def test_yam_drag_rejects_nonpositive_end_effector_mass():
    with pytest.raises(ValueError, match="ee_mass"):
        YamDragDriver(ee_mass=0.0)


def test_yam_drag_driver_returns_to_its_startup_pose_after_a_demo():
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    driver = YamDragDriver(robot_factory=factory, hz=200.0, return_to_start_duration_s=0.01)
    driver.connect()
    _wait_until(lambda: all(side in driver._follower_state for side in driver.sides), timeout_s=1.0)
    for side, robot in robots.items():
        if side[0] == "follower":
            robot.arm_pos = np.full(6, 0.4 if side[1] == "left" else -0.4)
            robot.gripper_pos = 0.8
    _wait_until(
        lambda: np.max(np.abs(driver.read_state().qpos[:6])) > 0.3,
        timeout_s=1.0,
    )

    driver.reset_home()

    expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25])
    for side in ("left", "right"):
        robot = robots[("follower", side)]
        np.testing.assert_allclose(robot.commands[-1], expected)
        assert robot.idle_count >= 2
    assert driver.status()["returning_home"] is False
    driver.close()


def test_saved_yam_drag_episode_requests_its_startup_pose_return():
    calls: list[tuple[float, bool]] = []
    client = SimpleNamespace(
        reset_home=lambda *, timeout, wait: calls.append((timeout, wait)) or {"accepted": True}
    )
    hardware = SimpleNamespace(
        robot_driver_status=lambda: {
            "supports_reset_home": True,
            "auto_return_after_episode": True,
            "home_return_label": "this run's startup pose",
            "drag_mode": "gravity_compensation",
        },
        robot_client=client,
    )

    assert _auto_return_after_saved_drag_episode(
        hardware, {"home_return": {"timeout_s": 15.0}}
    )
    assert calls == [(15.0, True)]


@pytest.mark.parametrize("name", ["coulomb_friction", "grav_comp_kd", "gravity_comp_factor"])
@pytest.mark.parametrize("values", [[0.1] * 5, [0.1] * 7, [-0.1] * 6, [float("nan")] * 6])
def test_yam_drag_rejects_invalid_tuning_before_connect(name, values):
    with pytest.raises(ValueError, match=name):
        YamDragDriver(**{name: values})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("gripper_motion_assist_torque_nm", -0.01),
        ("gripper_motion_assist_torque_nm", 0.21),
        ("gripper_motion_assist_velocity_threshold_m_s", 0.0),
        ("gripper_motion_assist_direction", 0),
    ],
)
def test_yam_drag_rejects_unsafe_gripper_motion_assist_before_connect(name, value):
    with pytest.raises(ValueError, match=name):
        YamDragDriver(**{name: value})


def test_yam_home_return_moves_only_followers_to_saved_pose(tmp_path):
    robots: dict[tuple[str, str], _FakeRobot] = {}

    def factory(**kwargs: Any) -> _FakeRobot:
        robot = _FakeRobot(role=kwargs["role"], side=kwargs["side"])
        robots[(kwargs["role"], kwargs["side"])] = robot
        return robot

    home_path = tmp_path / "yam_home.json"
    target_left = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.04]
    target_right = [-0.10, -0.11, -0.12, -0.13, -0.14, -0.15, 0.05]
    home_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "can_interfaces": {
                    "follower": {"left": "can_follower_l", "right": "can_follower_r"}
                },
                "follower": {"left": target_left, "right": target_right},
            }
        )
    )
    driver = YamBimanualDriver(
        robot_factory=factory,
        control_hz=200.0,
        home_path=str(home_path),
        home_return_enabled=True,
        home_return_duration_s=0.01,
    )
    driver.connect()
    driver.reset_home()

    np.testing.assert_allclose(robots[("follower", "left")].commands[-1], np.array(target_left) / [1, 1, 1, 1, 1, 1, GRIPPER_WIDTH_M])
    np.testing.assert_allclose(robots[("follower", "right")].commands[-1], np.array(target_right) / [1, 1, 1, 1, 1, 1, GRIPPER_WIDTH_M])
    assert not robots[("leader", "left")].commands
    assert not robots[("leader", "right")].commands
    assert robots[("leader", "left")].idle_count >= 1
    assert robots[("leader", "right")].idle_count >= 1
    assert driver.status()["supports_reset_home"] is True
    driver.close()


def test_record_drag_drains_yam_actions_into_existing_action_schema(tmp_path):
    command = np.arange(14, dtype=np.float32)
    driver = SimpleNamespace(
        drain_actions=lambda: [
            {
                "action": command,
                "action_space": "abs_qpos",
                "hz": 120.0,
                "stamp_ns": 123,
                "metadata": {"source": "yam_leader_follower"},
            }
        ]
    )
    recorder = DataRecorder(
        streams={},
        config={
            "enabled": True,
            "output_dir": str(tmp_path),
            "queue_size": 16,
            "fps": 30.0,
            "enable_plots": False,
        },
        name="yam_test",
    )
    recorder.start()
    recorder.start_episode("0000")
    data_session = SimpleNamespace(
        recorder=recorder,
        record_action=recorder.record_action,
    )

    hardware = SimpleNamespace(drain_robot_actions=driver.drain_actions)
    count = _drain_robot_actions(hardware, data_session)
    recorder.stop_episode()
    recorder.close()

    with (tmp_path / "0000" / "action" / "executed_action_dict.pkl").open(
        "rb"
    ) as handle:
        recorded = pickle.load(handle)

    assert count == 1
    np.testing.assert_array_equal(recorded["actions"][0], command)
    assert recorded["action_space"] == ["abs_qpos"]
    assert recorded["timestamps"] == [0]
    assert recorded["metadata"][0]["source"] == "yam_leader_follower"


def test_record_drag_reports_background_recorder_failure():
    recorder = SimpleNamespace(
        recording=True,
        details=lambda: {"failure": "camera encoder stopped"},
    )
    data_session = SimpleNamespace(
        recorder=recorder,
        record_action=lambda *args, **kwargs: False,
    )
    hardware = SimpleNamespace(
        drain_robot_actions=lambda: [
            {
                "action": np.zeros(14, dtype=np.float32),
                "action_space": "abs_qpos",
                "hz": 120.0,
                "stamp_ns": 123,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="data recorder failed: camera encoder stopped"):
        _drain_robot_actions(hardware, data_session)


def _wait_until(predicate: Any, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("timed out waiting for YAM driver")
