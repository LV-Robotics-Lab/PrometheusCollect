"""Hardware-free release checks. Assertions describe the required behavior.

Run from the repository root with PYTHONPATH=. and pytest plugin autoload off.
No physical devices or external services are opened.
"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from prometheus.data.raw_episode import RawEpisodeWriter
from prometheus.nodes.robots.yam import YamBimanualDriver, YamDragDriver
from prometheus.workflows import record_drag as workflow
from tests.test_yam_driver import _FakeRobot, _wait_until


def config(tmp_path):
    cfg = OmegaConf.load("tests/configs/record_drag_noop.yaml")
    cfg.run.output_dir = str(tmp_path / "run")
    cfg.run.runtime_dir = str(tmp_path / "run/runtime")
    cfg.data.recording.output_dir = str(tmp_path / "episodes")
    return cfg


def capture_sessions(monkeypatch):
    sessions = {}
    for name in ("hardware", "data", "event"):
        original = getattr(workflow, f"build_{name}_session")

        def build(*args, _name=name, _original=original, **kwargs):
            sessions[_name] = _original(*args, **kwargs)
            return sessions[_name]

        monkeypatch.setattr(workflow, f"build_{name}_session", build)
    return sessions


def test_recorder_failure_still_closes_all_sessions(tmp_path, monkeypatch):
    sessions = capture_sessions(monkeypatch)
    original = workflow.build_hardware_session

    def build(**kwargs):
        hw = original(**kwargs)

        def drain():
            data = sessions.get("data")
            if data is None or not data.recorder.recording:
                return []
            data.recorder._fail("injected camera encoder failure")
            return [
                {
                    "action": np.zeros(14),
                    "action_space": "abs_qpos",
                    "hz": 120,
                    "stamp_ns": time.time_ns(),
                }
            ]

        hw.drain_robot_actions = drain
        return hw

    monkeypatch.setattr(workflow, "build_hardware_session", build)
    try:
        with pytest.raises(RuntimeError, match="camera encoder failure"):
            workflow.run_from_config(config(tmp_path))
        states = {name: obj.status().state.value for name, obj in sessions.items()}
        assert all(state == "closed" for state in states.values()), states
    finally:
        for obj in reversed(list(sessions.values())):
            obj.close()


def test_failed_finalize_does_not_report_saved_or_move_arms(tmp_path, monkeypatch):
    sessions = capture_sessions(monkeypatch)
    returns = []

    # The noop client is built in start(), so capture the actual return helper.
    def return_home(*args, **kwargs):
        returns.append(kwargs)

    def fail_finalize(self):
        raise OSError("injected disk full while closing video")

    monkeypatch.setattr(RawEpisodeWriter, "finalize", fail_finalize)
    monkeypatch.setattr(workflow, "_return_robot_home", return_home)
    original_status = workflow._auto_return_after_saved_drag_episode

    def auto_return(hw, cfg):
        hw.robot_driver_status = lambda: {"auto_return_after_episode": True}
        return original_status(hw, cfg)

    monkeypatch.setattr(workflow, "_auto_return_after_saved_drag_episode", auto_return)
    with pytest.raises(RuntimeError, match="episode save failed"):
        workflow.run_from_config(config(tmp_path))
    manifest = json.loads((tmp_path / "run/workflow_manifest.json").read_text())
    assert (
        manifest["status"] == "error"
        and manifest["episodes_saved"] == 0
        and not returns
    ), {
        "saved": manifest["episodes_saved"],
        "return_calls": len(returns),
        "failure": sessions["data"].recorder.details()["failure"],
    }


def test_home_return_excludes_an_inflight_alignment(tmp_path):
    robots = {}

    def factory(**kw):
        robot = _FakeRobot(role=kw["role"], side=kw["side"])
        robots[(kw["role"], kw["side"])] = robot
        return robot

    home = tmp_path / "home.json"
    home.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "can_interfaces": {
                    "follower": {"left": "can_follower_l", "right": "can_follower_r"}
                },
                "follower": {side: [0] * 6 + [0.024] for side in ("left", "right")},
            }
        )
    )
    driver = YamBimanualDriver(
        robot_factory=factory,
        control_hz=50,
        alignment_duration_s=0.1,
        home_path=str(home),
        home_return_enabled=True,
        home_return_duration_s=0.3,
    )
    driver.connect()
    entered, release = threading.Event(), threading.Event()
    overlap = []
    errors = []
    original = driver._command_follower

    def command(side, target):
        if threading.current_thread().name == "review_alignment":
            if not entered.is_set():
                entered.set()
                assert release.wait(1)
            if driver.status()["returning_home"]:
                overlap.append(target.copy())
        return original(side, target)

    driver._command_follower = command

    def run(fn, *args):
        try:
            fn(*args)
        except BaseException as exc:
            errors.append(repr(exc))

    alignment = threading.Thread(
        target=run,
        args=(
            driver._engage,
            "left",
            driver._follower_state["left"].qpos.copy(),
            driver._leader_state["left"].qpos.copy(),
        ),
        name="review_alignment",
    )
    home_thread = threading.Thread(target=run, args=(driver.reset_home,))
    try:
        alignment.start()
        assert entered.wait(1)
        home_thread.start()
        assert driver._home_requested.wait(1)
        release.set()
        alignment.join(2)
        home_thread.join(2)
        assert not errors, errors
        assert not overlap, (
            f"{len(overlap)} alignment commands issued during Home return"
        )
    finally:
        release.set()
        alignment.join(2)
        if home_thread.ident is not None:
            home_thread.join(2)
        driver.close()


def test_drag_return_cleanup_resets_flag_after_assist_failure():
    driver = YamDragDriver(
        robot_factory=lambda **kw: _FakeRobot(role=kw["role"], side=kw["side"])
    )
    driver.connect()
    original = driver._set_gripper_assist_torque

    def fail(side, torque):
        if (
            threading.current_thread() is threading.main_thread()
            and driver._returning_home
        ):
            raise RuntimeError("injected assist reset failure")
        return original(side, torque)

    driver._set_gripper_assist_torque = fail
    try:
        with pytest.raises(RuntimeError, match="assist reset failure"):
            driver.reset_home()
        assert not driver.status()["returning_home"], driver.status()
    finally:
        driver.close()


@pytest.mark.parametrize("mode", ["drag", "leader_follower"])
def test_dead_i2rt_runtime_is_not_reported_ready(mode):
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    robots = []

    class CachedRobot(_FakeRobot):
        # Exercise the actual I2RT cached-observation method, without CAN.
        get_observations = MotorChainRobot.get_observations
        get_runtime_status = MotorChainRobot.get_runtime_status

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._state_lock = threading.RLock()
            self._gripper_index = 6 if self.role == "follower" else None
            n = 7 if self._gripper_index else 6
            self._joint_state = SimpleNamespace(
                pos=np.zeros(n), vel=np.zeros(n), eff=np.zeros(n), timestamp=time.time()
            )
            self.temp_record_flag = False
            self.motor_chain.running = True
            self.motor_chain.get_runtime_status = lambda: {
                "running": self.motor_chain.running,
                "state_age_s": 0.0,
                "stamp_ns": time.time_ns(),
            }
            self._stop_event = threading.Event()
            self._last_update_monotonic_s = time.monotonic()
            self.alive = True
            self._server_thread = SimpleNamespace(is_alive=lambda: self.alive)

    def factory(**kwargs):
        robot = CachedRobot(role=kwargs["role"], side=kwargs["side"])
        robots.append(robot)
        return robot

    if mode == "drag":
        driver = YamDragDriver(robot_factory=factory)
    else:
        driver = YamBimanualDriver(
            robot_factory=factory, require_enable_button=False, alignment_duration_s=0
        )
    driver.connect()
    try:
        _wait_until(lambda: driver.status()["ready_for_recording"], timeout_s=1)
        for robot in robots:
            robot.alive = False
            robot.motor_chain.running = False
        time.sleep(0.05)
        status = driver.status()
        assert not status["ready_for_recording"], status
    finally:
        driver.close()


def test_home_capture_and_return_use_same_path_after_output_override(tmp_path):
    from pathlib import Path

    configs = Path("prometheus/configs").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(configs)):
        cfg = compose(
            config_name="collect",
            overrides=["robot=yam", f"user.collect_root={tmp_path}"],
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert cfg["workflow"]["home_capture"]["path"] == cfg["robot"]["robot"]["home_path"]


def test_record_drag_entrypoint_actually_requests_drag_return():
    calls = []
    hw = SimpleNamespace(
        robot_driver_status=lambda: {
            "auto_return_after_episode": True,
            "supports_reset_home": True,
        },
        robot_client=SimpleNamespace(
            reset_home=lambda **kwargs: calls.append(kwargs) or {"accepted": True}
        ),
    )
    cfg = OmegaConf.to_container(
        OmegaConf.load("prometheus/configs/record_drag.yaml"), resolve=False
    )
    workflow._auto_return_after_saved_drag_episode(hw, cfg["workflow"])
    assert calls, (
        "record_drag.yaml lacks home_return; the TypeError is caught and return is skipped"
    )


def test_one_disengaged_side_does_not_drop_the_other_sides_actions():
    robots = {}

    def factory(**kw):
        robot = _FakeRobot(role=kw["role"], side=kw["side"])
        robots[(kw["role"], kw["side"])] = robot
        return robot

    driver = YamBimanualDriver(
        robot_factory=factory, control_hz=200, alignment_duration_s=0
    )
    driver.connect()
    try:
        for side in ("left", "right"):
            robots[("leader", side)].encoder.io_inputs[0] = 1
        _wait_until(lambda: driver.status()["ready_for_recording"], timeout_s=1)
        for side in ("left", "right"):
            robots[("leader", side)].encoder.io_inputs[0] = 0
        time.sleep(0.03)
        robots[("leader", "right")].encoder.io_inputs[0] = 1
        _wait_until(lambda: not driver.status()["synchronized"]["right"], timeout_s=1)
        time.sleep(0.03)
        driver.drain_actions()
        before = len(robots[("follower", "left")].commands)
        robots[("leader", "left")].arm_pos += 0.2
        time.sleep(0.05)
        commands = len(robots[("follower", "left")].commands) - before
        actions = driver.drain_actions()
        assert commands > 0
        assert actions, (
            f"left follower received {commands} commands, but zero action records"
        )
    finally:
        driver.close()


def test_hardware_failure_during_episode_is_failed_and_cleaned_up(
    tmp_path, monkeypatch
):
    sessions = capture_sessions(monkeypatch)
    original = workflow.build_hardware_session

    def build(**kwargs):
        hardware = original(**kwargs)

        def health():
            if sessions["data"].recorder.recording:
                raise RuntimeError("injected CAN failure during recording")

        hardware.check_health = health
        return hardware

    monkeypatch.setattr(workflow, "build_hardware_session", build)
    with pytest.raises(RuntimeError, match="CAN failure"):
        workflow.run_from_config(config(tmp_path))
    assert all(obj.status().state.value == "closed" for obj in sessions.values())
    manifest = json.loads((tmp_path / "run/workflow_manifest.json").read_text())
    assert manifest["episodes_saved"] == 0
    assert manifest["status"] == "error"
    events = json.loads((tmp_path / "episodes/0000/manifests/event.json").read_text())
    assert all(item["status"] == "failed" for item in events.values())


def test_preview_close_error_does_not_skip_hardware_cleanup(tmp_path, monkeypatch):
    sessions = capture_sessions(monkeypatch)

    def fail_close(_preview):
        raise RuntimeError("injected preview close failure")

    monkeypatch.setattr(workflow, "close_preview", fail_close)
    assert workflow.run_from_config(config(tmp_path)) == 1
    assert all(obj.status().state.value == "closed" for obj in sessions.values())
    manifest = json.loads((tmp_path / "run/workflow_manifest.json").read_text())
    assert "preview.close" in manifest["cleanup_errors"][0]


def test_drag_home_restores_other_side_after_one_side_fails():
    driver = YamDragDriver(
        robot_factory=lambda **kw: _FakeRobot(role=kw["role"], side=kw["side"]),
        return_to_start_duration_s=0.01,
    )
    driver.connect()
    original = driver._enter_drag_mode
    attempted = []

    def restore(side):
        attempted.append(side)
        if side == "left":
            raise RuntimeError("left restore failure")
        original(side)

    driver._enter_drag_mode = restore
    try:
        with pytest.raises(RuntimeError, match="restoration failed"):
            driver.reset_home()
        assert attempted == ["left", "right"]
        assert not driver.status()["returning_home"]
        assert not driver.status()["ready_for_recording"]
    finally:
        driver.close()


def test_home_refusal_keeps_workflow_alive():
    hardware = SimpleNamespace(
        robot_driver_status=lambda: {"supports_reset_home": True},
        robot_client=SimpleNamespace(
            reset_home=lambda **kwargs: {
                "accepted": False,
                "message": "joint displacement exceeds configured limit",
            }
        ),
    )
    workflow._return_robot_home(
        hardware,
        {"home_return": {"on_manual_return": True}},
        trigger="on_manual_return",
    )


def test_stale_feedback_refuses_reads_and_recording():
    driver = YamDragDriver(
        robot_factory=lambda **kw: _FakeRobot(role=kw["role"], side=kw["side"])
    )
    driver.connect()
    try:
        driver._followers["left"].get_runtime_status = lambda: {
            "running": True,
            "state_age_s": 2.0,
        }
        assert not driver.status()["ready_for_recording"]
        with pytest.raises(RuntimeError, match="stale"):
            driver.read_state()
    finally:
        driver.close()
