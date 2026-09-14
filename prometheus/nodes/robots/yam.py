from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from prometheus.nodes.robots.robot_node import RobotState

YAM_ACTION_SPACES = ("abs_qpos",)
ACTION_DIMS = {"abs_qpos": 14}
SIDES = ("left", "right")
ARM_DOF = 6
SIDE_DOF = 7
GRIPPER_WIDTH_M = 0.096
# Upper ceiling for the direct motor-torque term used only by manual gripper
# motion assist.  This remains below the 0.50 Nm gripper-limit calibration
# torque, but requires an observed low-speed hardware check before collection.
MAX_GRIPPER_MOTION_ASSIST_TORQUE_NM = 0.20
ROBOT_TOPIC_SUFFIXES = {
    "state": "state",
    "control": "control",
    "result": "control_result",
    "status": "status",
}


@dataclass(frozen=True, kw_only=True)
class YamRobotState(RobotState):
    leader_qpos: np.ndarray
    applied_action: np.ndarray


@dataclass(frozen=True)
class _SideState:
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    stamp_ns: int


@contextmanager
def _home_motion(driver: Any):
    """Cancel side motion, then own both command paths until Home finishes."""
    with driver._home_lock:
        driver._home_requested.set()
        try:
            with ExitStack() as stack:
                for side in SIDES:
                    stack.enter_context(driver._motion_locks[side])
                if driver._stop.is_set():
                    raise RuntimeError("YAM Home return interrupted by shutdown")
                with driver._lock:
                    driver._returning_home = True
                yield
        finally:
            with driver._lock:
                driver._returning_home = False
            driver._home_requested.clear()


def _runtime_error(robot: Any, max_age_s: float) -> str:
    try:
        status = robot.get_runtime_status()
        if not status.get("running", False):
            return str(status.get("error") or "I2RT motor/control loop is not running")
        age_s = float(status.get("state_age_s", float("inf")))
    except Exception as exc:
        return f"I2RT health check failed: {exc}"
    if not np.isfinite(age_s) or age_s < 0 or age_s > max_age_s:
        return f"I2RT feedback is stale (age={age_s:.3f}s, limit={max_age_s:.3f}s)"
    return ""


def _run_home_trajectory(
    driver: Any,
    start: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    duration_s: float,
    hz: float,
    command: Callable[..., Any],
) -> None:
    steps = max(1, round(duration_s * hz))
    started_at = time.monotonic()
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix=f"{driver.robot_id}_home"
    ) as executor:
        for index in range(1, steps + 1):
            if driver._stop.is_set():
                raise RuntimeError("YAM Home return interrupted by shutdown")
            driver._raise_if_unhealthy()
            alpha = index / steps
            futures = [
                executor.submit(
                    command, side, start[side] * (1 - alpha) + target[side] * alpha
                )
                for side in SIDES
            ]
            for future in futures:
                future.result()
            remaining = started_at + index * duration_s / steps - time.monotonic()
            if remaining > 0:
                driver._stop.wait(remaining)


def _observations(robot: Any, max_age_s: float) -> Mapping[str, Any]:
    error = _runtime_error(robot, max_age_s)
    if error:
        raise RuntimeError(error)
    return robot.get_observations()


def _validate_i2rt_api(*, drag: bool) -> None:
    """Check the dependency before constructing any robot or opening CAN."""
    import inspect
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    required = {"get_runtime_status", "enter_gravity_comp_idle"}
    if drag:
        required.add("set_gripper_assist_torque")
    missing = sorted(
        name for name in required if not callable(getattr(MotorChainRobot, name, None))
    )
    if drag:
        missing.extend(
            sorted(
                {"grav_comp_kd", "coulomb_friction"}
                - set(inspect.signature(get_yam_robot).parameters)
            )
        )
    if missing:
        raise RuntimeError(
            f"Incompatible I2RT installation; missing {missing}. "
            "Install the pinned YAM dependency using scripts/prepare_yam_i2rt.py; see README.md."
        )


class YamBimanualDriver:
    """I2RT YAM bimanual leader-follower adapter for Prometheus.

    Each side has a dedicated teleoperation thread. I2RT continues to own the
    per-arm real-time motor loop; this adapter samples its cached observations,
    applies the teaching-handle command to the follower, and exposes follower
    feedback through the normal Prometheus RobotDriver contract.
    """

    sides = SIDES
    arm_dof = ARM_DOF
    action_dim = 14

    def __init__(
        self,
        *,
        robot_id: str = "yam",
        leader_left_can: str = "can_leader_l",
        leader_right_can: str = "can_leader_r",
        follower_left_can: str = "can_follower_l",
        follower_right_can: str = "can_follower_r",
        leader_arm: str = "yam",
        follower_arm: str = "yam_ultra_2",
        leader_gripper: str = "yam_teaching_handle",
        follower_gripper: str = "linear_4310",
        gripper_width_m: float = GRIPPER_WIDTH_M,
        hz: float = 120.0,
        control_hz: float = 120.0,
        connect_timeout_s: float = 15.0,
        state_timeout_s: float = 0.5,
        alignment_duration_s: float = 3.0,
        bilateral_kp: float = 0.0,
        teleoperation_enabled: bool = True,
        require_enable_button: bool = True,
        enable_auto_recovery: bool = False,
        home_path: str | None = None,
        home_return_enabled: bool = False,
        home_return_duration_s: float = 5.0,
        home_return_max_joint_delta_rad: float = 2.0,
        home_return_max_gripper_delta_m: float = GRIPPER_WIDTH_M,
        sim: bool = False,
        robot_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not robot_id:
            raise ValueError("YAM robot_id must be non-empty")
        if float(hz) <= 0 or float(control_hz) <= 0:
            raise ValueError("YAM publish and control rates must be positive")
        if float(gripper_width_m) <= 0:
            raise ValueError("YAM gripper_width_m must be positive")
        if float(connect_timeout_s) <= 0:
            raise ValueError("YAM connect_timeout_s must be positive")
        if not np.isfinite(state_timeout_s) or float(state_timeout_s) <= 0:
            raise ValueError("YAM state_timeout_s must be positive and finite")
        if float(alignment_duration_s) < 0:
            raise ValueError("YAM alignment_duration_s must be non-negative")
        if float(home_return_duration_s) <= 0:
            raise ValueError("YAM home_return_duration_s must be positive")
        if float(home_return_max_joint_delta_rad) <= 0:
            raise ValueError("YAM home_return_max_joint_delta_rad must be positive")
        if float(home_return_max_gripper_delta_m) <= 0:
            raise ValueError("YAM home_return_max_gripper_delta_m must be positive")

        self.robot_id = str(robot_id)
        self.leader_can = {
            "left": str(leader_left_can),
            "right": str(leader_right_can),
        }
        self.follower_can = {
            "left": str(follower_left_can),
            "right": str(follower_right_can),
        }
        self.leader_arm = str(leader_arm)
        self.follower_arm = str(follower_arm)
        self.leader_gripper = str(leader_gripper)
        self.follower_gripper = str(follower_gripper)
        self.gripper_width_m = float(gripper_width_m)
        self.hz = float(hz)
        self.control_hz = float(control_hz)
        self.connect_timeout_s = float(connect_timeout_s)
        self.state_timeout_s = float(state_timeout_s)
        self.alignment_duration_s = float(alignment_duration_s)
        self.bilateral_kp = float(bilateral_kp)
        self.teleoperation_enabled = bool(teleoperation_enabled)
        self.require_enable_button = bool(require_enable_button)
        self.enable_auto_recovery = bool(enable_auto_recovery)
        self.home_path = None if home_path is None else Path(home_path).expanduser()
        self.home_return_enabled = bool(home_return_enabled)
        self.home_return_duration_s = float(home_return_duration_s)
        self.home_return_max_joint_delta_rad = float(home_return_max_joint_delta_rad)
        self.home_return_max_gripper_delta_m = float(home_return_max_gripper_delta_m)
        self.sim = bool(sim)
        self._robot_factory = robot_factory

        self._robots: dict[str, dict[str, Any]] = {"leader": {}, "follower": {}}
        self._leader_kp: dict[str, np.ndarray] = {}
        self._follower_state: dict[str, _SideState] = {}
        self._leader_state: dict[str, _SideState] = {}
        self._applied_action: dict[str, np.ndarray] = {}
        self._synchronized = {side: False for side in SIDES}
        self._button_down = {side: False for side in SIDES}
        self._side_ready = {side: threading.Event() for side in SIDES}
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._home_lock = threading.RLock()
        self._home_requested = threading.Event()
        self._motion_locks = {side: threading.RLock() for side in SIDES}
        self._errors: dict[str, str] = {}
        self._action_sequence = 0
        self._action_dirty: set[str] = set()
        self._pending_actions: deque[dict[str, Any]] = deque(maxlen=4096)
        self._connected = False
        self._closed = True
        self._returning_home = False

    @property
    def joint_names(self) -> list[str]:
        names: list[str] = []
        for side in SIDES:
            names.extend([f"{side}_joint_{index + 1}" for index in range(ARM_DOF)])
            names.append(f"{side}_gripper")
        return names

    @property
    def is_connected(self) -> bool:
        return (
            self._connected
            and not self._closed
            and len(self._robots["leader"]) == 2
            and len(self._robots["follower"]) == 2
        )

    def connect(self) -> None:
        if self.is_connected:
            return
        if self._robot_factory is None:
            _validate_i2rt_api(drag=False)
        self._stop.clear()
        self._closed = False
        with self._lock:
            self._errors.clear()
            self._follower_state.clear()
            self._leader_state.clear()
            self._applied_action.clear()
            self._action_dirty.clear()
            self._pending_actions.clear()
            for side in SIDES:
                self._synchronized[side] = False
                self._button_down[side] = False
                self._side_ready[side].clear()
        try:
            for role in ("follower", "leader"):
                for side in SIDES:
                    self._robots[role][side] = self._make_robot(role, side)
            for side in SIDES:
                info = self._robots["leader"][side].get_robot_info()
                self._leader_kp[side] = (
                    np.asarray(info["kp"], dtype=np.float64).reshape(ARM_DOF).copy()
                )
                thread = threading.Thread(
                    target=self._teleop_loop,
                    args=(side,),
                    name=f"{self.robot_id}_{side}_teleop",
                    daemon=True,
                )
                self._threads[side] = thread
                thread.start()
            self._connected = True
            self._wait_for_initial_state()
        except Exception:
            self.close()
            raise

    def read_state(self) -> YamRobotState:
        self._raise_if_unhealthy()
        with self._lock:
            missing = [
                side
                for side in SIDES
                if side not in self._follower_state or side not in self._leader_state
            ]
            if missing:
                raise RuntimeError(f"YAM state is not ready for sides: {missing}")
            follower = [self._follower_state[side] for side in SIDES]
            leader = [self._leader_state[side] for side in SIDES]
            applied = [
                self._applied_action.get(side, self._follower_state[side].qpos)
                for side in SIDES
            ]
            return YamRobotState(
                joint_names=self.joint_names,
                stamp_ns=min(state.stamp_ns for state in follower),
                qpos=np.concatenate([state.qpos for state in follower]).astype(
                    np.float32
                ),
                qvel=np.concatenate([state.qvel for state in follower]).astype(
                    np.float32
                ),
                effort=np.concatenate([state.effort for state in follower]).astype(
                    np.float32
                ),
                leader_qpos=np.concatenate([state.qpos for state in leader]).astype(
                    np.float32
                ),
                applied_action=np.concatenate(applied).astype(np.float32),
            )

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        self._raise_if_unhealthy()
        if str(action_space) != "abs_qpos":
            raise ValueError(f"YAM action_space must be one of {YAM_ACTION_SPACES}")
        if self.teleoperation_enabled:
            raise RuntimeError(
                "external YAM actions require teleoperation_enabled=false"
            )
        command = np.asarray(action, dtype=np.float64).reshape(14)
        if not np.all(np.isfinite(command)):
            raise ValueError("YAM action contains NaN or inf values")
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix=f"{self.robot_id}_send"
        ) as executor:
            futures = {
                side: executor.submit(
                    self._external_command_follower,
                    side,
                    command[index * SIDE_DOF : (index + 1) * SIDE_DOF],
                )
                for index, side in enumerate(SIDES)
            }
            for side in SIDES:
                futures[side].result()

    def set_teach_mode(self) -> None:
        with _home_motion(self):
            for side in SIDES:
                self._disengage(side)
                self._robots["follower"][side].enter_gravity_comp_idle()

    def _external_command_follower(self, side: str, action: np.ndarray) -> None:
        with self._motion_locks[side]:
            if self._stop.is_set() or self._home_requested.is_set():
                raise RuntimeError("YAM external action refused during Home/shutdown")
            self._raise_if_unhealthy()
            self._command_follower(side, action)

    def set_position_mode(self) -> None:
        if (
            self.teleoperation_enabled
            and self.require_enable_button
            and not all(self._synchronized.values())
        ):
            raise RuntimeError(
                "press each YAM teaching-handle button before enabling position control"
            )

    def reset_home(self) -> None:
        """Cancel synchronization and return followers with exclusive control."""
        if not self.home_return_enabled:
            raise RuntimeError("YAM home return is disabled")
        target = self._load_home_follower_target()
        with _home_motion(self):
            self._raise_if_unhealthy()
            if not self.is_connected:
                raise RuntimeError("YAM is not connected")
            with self._lock:
                for side in SIDES:
                    self._synchronized[side] = False
                    # Preserve button-down state: a held button must be released
                    # before a fresh press can engage after returning Home.
            start = {side: self._read_follower(side).qpos for side in SIDES}
            self._validate_home_return(start, target)
            for side in SIDES:
                self._robots["leader"][side].enter_gravity_comp_idle()
            _run_home_trajectory(
                self,
                start,
                target,
                self.home_return_duration_s,
                self.control_hz,
                self._command_follower,
            )
            with self._lock:
                for side in SIDES:
                    self._follower_state[side] = self._read_follower(side)
                    self._leader_state[side], self._button_down[side] = (
                        self._read_leader(side)
                    )

    def _load_home_follower_target(self) -> dict[str, np.ndarray]:
        if self.home_path is None:
            raise RuntimeError("YAM Home path is not configured")
        try:
            payload = json.loads(self.home_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"YAM Home file does not exist: {self.home_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"YAM Home file is not valid JSON: {self.home_path}"
            ) from exc
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise RuntimeError("YAM Home file has an unsupported schema")
        interfaces = payload.get("can_interfaces")
        if (
            not isinstance(interfaces, Mapping)
            or interfaces.get("follower") != self.follower_can
        ):
            raise RuntimeError(
                "YAM Home file follower CAN mapping does not match this robot"
            )
        follower = payload.get("follower")
        if not isinstance(follower, Mapping):
            raise RuntimeError("YAM Home file has no follower pose")
        target: dict[str, np.ndarray] = {}
        for side in SIDES:
            values = np.asarray(follower.get(side), dtype=np.float64).reshape(-1)
            if values.shape != (SIDE_DOF,) or not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"YAM Home follower.{side} must contain 7 finite positions"
                )
            target[side] = values
        return target

    def _validate_home_return(
        self, start: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]
    ) -> None:
        for side in SIDES:
            delta = np.abs(target[side] - start[side])
            if np.any(delta[:ARM_DOF] > self.home_return_max_joint_delta_rad):
                raise RuntimeError(
                    f"YAM Home return refused on {side}: joint displacement exceeds "
                    f"{self.home_return_max_joint_delta_rad:.3f} rad"
                )
            if delta[-1] > self.home_return_max_gripper_delta_m:
                raise RuntimeError(
                    f"YAM Home return refused on {side}: gripper displacement exceeds "
                    f"{self.home_return_max_gripper_delta_m:.3f} m"
                )
            info = self._robots["follower"][side].get_robot_info()
            limits = np.asarray(info.get("joint_limits"), dtype=np.float64)
            if (
                limits.shape != (ARM_DOF, 2)
                or np.any(target[side][:ARM_DOF] < limits[:, 0])
                or np.any(target[side][:ARM_DOF] > limits[:, 1])
            ):
                raise RuntimeError(
                    f"YAM Home return refused on {side}: target violates joint limits"
                )

    def drain_actions(self) -> list[dict[str, Any]]:
        """Return and clear clipped follower targets submitted to I2RT."""
        with self._lock:
            items = list(self._pending_actions)
            self._pending_actions.clear()
        return items

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        with self._home_lock:
            errors: list[str] = []
            for thread in self._threads.values():
                if thread is not threading.current_thread():
                    thread.join(timeout=2.0)
            self._threads.clear()
            for role in ("leader", "follower"):
                for side in reversed(SIDES):
                    robot = self._robots[role].pop(side, None)
                    if robot is None:
                        continue
                    try:
                        idle = getattr(robot, "enter_gravity_comp_idle", None)
                        if callable(idle):
                            idle()
                    except Exception as exc:  # noqa: BLE001 - continue releasing the other CAN chains
                        errors.append(f"{role}.{side}.idle: {exc}")
                    try:
                        robot.close()
                    except Exception as exc:  # noqa: BLE001 - continue releasing the other CAN chains
                        errors.append(f"{role}.{side}.close: {exc}")
            self._connected = False
            self._closed = True
            with self._lock:
                for side in SIDES:
                    self._synchronized[side] = False
            if errors:
                raise RuntimeError("YAM cleanup errors: " + "; ".join(errors))

    def status(self) -> dict[str, Any]:
        self._check_runtime_health()
        with self._lock:
            synchronized = dict(self._synchronized)
            errors = dict(self._errors)
            pending_actions = len(self._pending_actions)
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "connected": self.is_connected,
            "ready": self.is_connected
            and all(event.is_set() for event in self._side_ready.values())
            and not errors,
            "ready_for_recording": self.is_connected
            and all(synchronized.values())
            and not self._home_requested.is_set()
            and not errors,
            "action_space": "abs_qpos",
            "action_dim": self.action_dim,
            "supports_reset_home": self.home_return_enabled
            and self.home_path is not None
            and self.home_path.is_file(),
            "home_path": "" if self.home_path is None else str(self.home_path),
            "home_return_duration_s": self.home_return_duration_s,
            "returning_home": self._returning_home,
            "hz": self.hz,
            "control_hz": self.control_hz,
            "teleoperation_enabled": self.teleoperation_enabled,
            "leader_can": dict(self.leader_can),
            "follower_can": dict(self.follower_can),
            "synchronized": synchronized,
            "errors": errors,
            "pending_actions": pending_actions,
        }

    def _make_robot(self, role: str, side: str) -> Any:
        arm = self.leader_arm if role == "leader" else self.follower_arm
        gripper = self.leader_gripper if role == "leader" else self.follower_gripper
        channel = self.leader_can[side] if role == "leader" else self.follower_can[side]
        if self._robot_factory is not None:
            return self._robot_factory(
                role=role,
                side=side,
                channel=channel,
                arm=arm,
                gripper=gripper,
                sim=self.sim,
            )

        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        return get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name(arm),
            gripper_type=GripperType.from_string_name(gripper),
            zero_gravity_mode=True,
            sim=self.sim,
            enable_auto_recovery=self.enable_auto_recovery,
        )

    def _wait_for_initial_state(self) -> None:
        deadline = time.monotonic() + self.connect_timeout_s
        for side in SIDES:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._side_ready[side].wait(remaining):
                self._raise_if_unhealthy()
                raise TimeoutError(
                    f"timed out waiting for YAM {side} leader/follower state"
                )
        self._raise_if_unhealthy()

    def _teleop_loop(self, side: str) -> None:
        period_s = 1.0 / self.control_hz
        initial_deadline = time.monotonic() + self.connect_timeout_s
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                try:
                    with self._motion_locks[side]:
                        if (
                            not self._home_requested.is_set()
                            and not self._stop.is_set()
                        ):
                            self._teleop_step(side)
                except RuntimeError:
                    if (
                        not self._side_ready[side].is_set()
                        and time.monotonic() < initial_deadline
                    ):
                        self._stop.wait(min(period_s, 0.05))
                        continue
                    raise
                delay = period_s - (time.monotonic() - tick)
                if delay > 0:
                    self._stop.wait(delay)
        except Exception as exc:
            with self._lock:
                self._errors[side] = f"{type(exc).__name__}: {exc}"
            self._side_ready[side].set()

    def _teleop_step(self, side: str) -> None:
        follower = self._read_follower(side)
        leader, pressed = self._read_leader(side)
        with self._lock:
            self._follower_state[side] = follower
            self._leader_state[side] = leader
            self._applied_action.setdefault(side, follower.qpos.copy())
        self._side_ready[side].set()
        if not self.teleoperation_enabled:
            return
        rising_edge = pressed and not self._button_down[side]
        self._button_down[side] = pressed
        if self.require_enable_button and rising_edge:
            if self._synchronized[side]:
                self._disengage(side)
            else:
                self._engage(side, follower.qpos, leader.qpos)
        elif not self.require_enable_button and not self._synchronized[side]:
            self._engage(side, follower.qpos, leader.qpos)
        if (
            self._synchronized[side]
            and not self._home_requested.is_set()
            and not self._stop.is_set()
        ):
            self._command_follower(side, leader.qpos)
            self._robots["leader"][side].command_joint_pos(
                follower.qpos[:ARM_DOF].copy()
            )

    def _read_follower(self, side: str) -> _SideState:
        obs = _observations(self._robots["follower"][side], self.state_timeout_s)
        arm_pos = _vector(obs, "joint_pos", ARM_DOF)
        arm_vel = _vector(obs, "joint_vel", ARM_DOF)
        arm_effort = _vector(obs, "joint_eff", ARM_DOF)
        gripper_pos = _scalar(obs, "gripper_pos")
        gripper_vel = _scalar(obs, "gripper_vel", default=np.nan)
        gripper_effort = _scalar(obs, "gripper_eff", default=np.nan)
        return _SideState(
            stamp_ns=int(_scalar(obs, "timestamp") * 1_000_000_000),
            qpos=np.concatenate([arm_pos, [gripper_pos * self.gripper_width_m]]),
            qvel=np.concatenate([arm_vel, [gripper_vel * self.gripper_width_m]]),
            effort=np.concatenate([arm_effort, [gripper_effort]]),
        )

    def _read_leader(self, side: str) -> tuple[_SideState, bool]:
        robot = self._robots["leader"][side]
        obs = _observations(robot, self.state_timeout_s)
        arm_pos = _vector(obs, "joint_pos", ARM_DOF)
        arm_vel = _vector(obs, "joint_vel", ARM_DOF)
        arm_effort = _vector(obs, "joint_eff", ARM_DOF)
        encoder_states = robot.motor_chain.get_same_bus_device_states()
        if not encoder_states:
            raise RuntimeError("teaching-handle encoder state is not ready")
        encoder = encoder_states[0]
        gripper_normalized = float(np.clip(1.0 - float(encoder.position), 0.0, 1.0))
        gripper_velocity = (
            -float(getattr(encoder, "velocity", np.nan)) * self.gripper_width_m
        )
        buttons = np.asarray(
            getattr(encoder, "io_inputs", ()), dtype=np.float64
        ).reshape(-1)
        pressed = bool(buttons.size and buttons[0] > 0.5)
        return (
            _SideState(
                stamp_ns=int(_scalar(obs, "timestamp") * 1_000_000_000),
                qpos=np.concatenate(
                    [arm_pos, [gripper_normalized * self.gripper_width_m]]
                ),
                qvel=np.concatenate([arm_vel, [gripper_velocity]]),
                effort=np.concatenate([arm_effort, [np.nan]]),
            ),
            pressed,
        )

    def _engage(
        self, side: str, follower_qpos: np.ndarray, leader_qpos: np.ndarray
    ) -> None:
        with self._motion_locks[side]:
            self._align_side(side, follower_qpos, leader_qpos)

    def _align_side(
        self, side: str, follower_qpos: np.ndarray, leader_qpos: np.ndarray
    ) -> None:
        if self._home_requested.is_set() or self._stop.is_set():
            return
        leader = self._robots["leader"][side]
        leader.update_kp_kd(
            kp=self._leader_kp[side] * self.bilateral_kp,
            kd=np.zeros(ARM_DOF, dtype=np.float64),
        )
        leader.command_joint_pos(leader_qpos[:ARM_DOF].copy())
        steps = max(1, round(self.alignment_duration_s * self.control_hz))
        for index in range(1, steps + 1):
            if self._stop.is_set() or self._home_requested.is_set():
                return
            self._raise_if_unhealthy()
            alpha = index / steps
            target = follower_qpos * (1.0 - alpha) + leader_qpos * alpha
            self._command_follower(side, target)
            if self.alignment_duration_s > 0:
                self._stop.wait(self.alignment_duration_s / steps)
        with self._lock:
            if not self._stop.is_set() and not self._home_requested.is_set():
                self._synchronized[side] = True

    def _disengage(self, side: str) -> None:
        leader = self._robots["leader"].get(side)
        if leader is not None:
            leader.update_kp_kd(
                kp=np.zeros(ARM_DOF, dtype=np.float64),
                kd=np.zeros(ARM_DOF, dtype=np.float64),
            )
            state = self._follower_state.get(side)
            if state is not None:
                leader.command_joint_pos(state.qpos[:ARM_DOF].copy())
            idle = getattr(leader, "enter_gravity_comp_idle", None)
            if callable(idle):
                idle()
        with self._lock:
            self._synchronized[side] = False

    def _command_follower(self, side: str, physical_qpos: np.ndarray) -> None:
        command = np.asarray(physical_qpos, dtype=np.float64).reshape(SIDE_DOF).copy()
        command[-1] = np.clip(command[-1] / self.gripper_width_m, 0.0, 1.0)
        self._robots["follower"][side].command_joint_pos(command)
        applied = command.copy()
        applied[-1] *= self.gripper_width_m
        with self._lock:
            self._applied_action[side] = applied
            self._action_dirty.add(side)
            active_sides = {name for name in SIDES if self._synchronized[name]}
            required_sides = active_sides or set(SIDES)
            if set(self._applied_action) == set(SIDES) and required_sides.issubset(
                self._action_dirty
            ):
                self._action_sequence += 1
                leader_qpos = (
                    np.concatenate(
                        [self._leader_state[name].qpos for name in SIDES]
                    ).astype(np.float32)
                    if all(name in self._leader_state for name in SIDES)
                    else np.full(14, np.nan, dtype=np.float32)
                )
                self._pending_actions.append(
                    {
                        "action": np.concatenate(
                            [self._applied_action[name] for name in SIDES]
                        ).astype(np.float32),
                        "action_space": "abs_qpos",
                        "hz": self.control_hz,
                        "stamp_ns": time.time_ns(),
                        "metadata": {
                            "source": "yam_leader_follower",
                            "sequence": self._action_sequence,
                            "updated_side": side,
                            "synchronized": dict(self._synchronized),
                            "leader_qpos": leader_qpos.tolist(),
                        },
                    }
                )
                self._action_dirty.clear()

    def _raise_if_unhealthy(self) -> None:
        self._check_runtime_health()
        with self._lock:
            errors = dict(self._errors)
        if errors:
            raise RuntimeError(f"YAM teleoperation failed: {errors}")

    def _check_runtime_health(self) -> None:
        if not self._connected or self._closed:
            return
        with self._lock:
            for role, robots in self._robots.items():
                for side, robot in robots.items():
                    error = _runtime_error(robot, self.state_timeout_s)
                    if error:
                        self._errors[f"{role}.{side}"] = error


class YamDragDriver:
    """Follower-only YAM gravity-compensation driver for manual drag demos.

    The driver deliberately owns only the two follower CAN chains. It never
    sends a position command while a demonstration is being recorded; instead
    it leaves both followers in I2RT gravity-compensation idle mode and records
    the observed follower joint configuration as an ``abs_qpos`` trajectory.
    """

    sides = SIDES
    arm_dof = ARM_DOF
    action_dim = 14

    def __init__(
        self,
        *,
        robot_id: str = "yam_drag",
        follower_left_can: str = "can_follower_l",
        follower_right_can: str = "can_follower_r",
        follower_arm: str = "yam_ultra_2",
        follower_gripper: str = "linear_4310",
        gripper_width_m: float = GRIPPER_WIDTH_M,
        ee_mass: float | None = None,
        hz: float = 120.0,
        connect_timeout_s: float = 15.0,
        state_timeout_s: float = 0.5,
        enable_auto_recovery: bool = False,
        use_coulomb_friction: bool = False,
        coulomb_friction: list[float] | None = None,
        grav_comp_kd: list[float] | None = None,
        gravity_comp_factor: list[float] | None = None,
        gripper_motion_assist_enabled: bool = True,
        gripper_motion_assist_torque_nm: float = 0.05,
        gripper_motion_assist_velocity_threshold_m_s: float = 0.002,
        gripper_motion_assist_direction: int = 1,
        return_to_start_enabled: bool = True,
        return_to_start_duration_s: float = 8.0,
        return_to_start_max_joint_delta_rad: float = 2.0,
        return_to_start_max_gripper_delta_m: float = GRIPPER_WIDTH_M,
        sim: bool = False,
        robot_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not robot_id:
            raise ValueError("YAM drag robot_id must be non-empty")
        if float(hz) <= 0:
            raise ValueError("YAM drag publish rate must be positive")
        if float(gripper_width_m) <= 0:
            raise ValueError("YAM drag gripper_width_m must be positive")
        if ee_mass is not None and (not np.isfinite(ee_mass) or float(ee_mass) <= 0):
            raise ValueError("YAM drag ee_mass must be a positive finite value in kg")
        if float(connect_timeout_s) <= 0:
            raise ValueError("YAM drag connect_timeout_s must be positive")
        if not np.isfinite(state_timeout_s) or float(state_timeout_s) <= 0:
            raise ValueError("YAM drag state_timeout_s must be positive and finite")
        if float(return_to_start_duration_s) <= 0:
            raise ValueError("YAM drag return_to_start_duration_s must be positive")
        if float(return_to_start_max_joint_delta_rad) <= 0:
            raise ValueError(
                "YAM drag return_to_start_max_joint_delta_rad must be positive"
            )
        if float(return_to_start_max_gripper_delta_m) <= 0:
            raise ValueError(
                "YAM drag return_to_start_max_gripper_delta_m must be positive"
            )
        if (
            not np.isfinite(gripper_motion_assist_torque_nm)
            or float(gripper_motion_assist_torque_nm) < 0
            or float(gripper_motion_assist_torque_nm)
            > MAX_GRIPPER_MOTION_ASSIST_TORQUE_NM
        ):
            raise ValueError(
                "YAM drag gripper_motion_assist_torque_nm must be finite and between 0 and "
                f"{MAX_GRIPPER_MOTION_ASSIST_TORQUE_NM:.2f} Nm"
            )
        if (
            not np.isfinite(gripper_motion_assist_velocity_threshold_m_s)
            or float(gripper_motion_assist_velocity_threshold_m_s) <= 0
        ):
            raise ValueError(
                "YAM drag gripper_motion_assist_velocity_threshold_m_s must be positive and finite"
            )
        try:
            gripper_motion_assist_direction = float(gripper_motion_assist_direction)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "YAM drag gripper_motion_assist_direction must be -1 or 1"
            ) from exc
        if not np.isfinite(
            gripper_motion_assist_direction
        ) or gripper_motion_assist_direction not in (-1.0, 1.0):
            raise ValueError("YAM drag gripper_motion_assist_direction must be -1 or 1")

        self.drag_tuning = {
            "use_coulomb_friction": bool(use_coulomb_friction),
            "coulomb_friction": _drag_joint_values(
                coulomb_friction, "coulomb_friction"
            ),
            "grav_comp_kd": _drag_joint_values(grav_comp_kd, "grav_comp_kd"),
            "gravity_comp_factor": _drag_joint_values(
                gravity_comp_factor, "gravity_comp_factor"
            ),
        }

        self.robot_id = str(robot_id)
        self.follower_can = {
            "left": str(follower_left_can),
            "right": str(follower_right_can),
        }
        self.follower_arm = str(follower_arm)
        self.follower_gripper = str(follower_gripper)
        self.gripper_width_m = float(gripper_width_m)
        self.ee_mass = None if ee_mass is None else float(ee_mass)
        self.hz = float(hz)
        self.connect_timeout_s = float(connect_timeout_s)
        self.state_timeout_s = float(state_timeout_s)
        self.enable_auto_recovery = bool(enable_auto_recovery)
        self.gripper_motion_assist_enabled = bool(gripper_motion_assist_enabled)
        self.gripper_motion_assist_torque_nm = float(gripper_motion_assist_torque_nm)
        self.gripper_motion_assist_velocity_threshold_m_s = float(
            gripper_motion_assist_velocity_threshold_m_s
        )
        self.gripper_motion_assist_direction = int(gripper_motion_assist_direction)
        self.return_to_start_enabled = bool(return_to_start_enabled)
        self.return_to_start_duration_s = float(return_to_start_duration_s)
        self.return_to_start_max_joint_delta_rad = float(
            return_to_start_max_joint_delta_rad
        )
        self.return_to_start_max_gripper_delta_m = float(
            return_to_start_max_gripper_delta_m
        )
        self.sim = bool(sim)
        self._robot_factory = robot_factory

        self._followers: dict[str, Any] = {}
        self._follower_state: dict[str, _SideState] = {}
        self._side_ready = {side: threading.Event() for side in SIDES}
        self._threads: dict[str, threading.Thread] = {}
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._home_lock = threading.RLock()
        self._home_requested = threading.Event()
        self._motion_locks = {side: threading.RLock() for side in SIDES}
        self._errors: dict[str, str] = {}
        self._action_sequence = 0
        self._action_dirty: set[str] = set()
        self._pending_actions: deque[dict[str, Any]] = deque(maxlen=4096)
        self._start_pose: dict[str, np.ndarray] = {}
        self._gripper_motion_assist_commanded_torque_nm = {side: 0.0 for side in SIDES}
        self._returning_home = False
        self._connected = False
        self._closed = True

    @property
    def joint_names(self) -> list[str]:
        names: list[str] = []
        for side in SIDES:
            names.extend([f"{side}_joint_{index + 1}" for index in range(ARM_DOF)])
            names.append(f"{side}_gripper")
        return names

    @property
    def is_connected(self) -> bool:
        return (
            self._connected and not self._closed and len(self._followers) == len(SIDES)
        )

    def connect(self) -> None:
        if self.is_connected:
            return
        if self._robot_factory is None:
            _validate_i2rt_api(drag=True)
        self._stop.clear()
        self._closed = False
        with self._lock:
            self._errors.clear()
            self._follower_state.clear()
            self._action_dirty.clear()
            self._pending_actions.clear()
            self._start_pose.clear()
            self._gripper_motion_assist_commanded_torque_nm = {
                side: 0.0 for side in SIDES
            }
            self._returning_home = False
            for side in SIDES:
                self._side_ready[side].clear()
        try:
            for side in SIDES:
                follower = self._make_follower(side)
                self._followers[side] = follower
                self._enter_drag_mode(side)
            for side in SIDES:
                thread = threading.Thread(
                    target=self._drag_loop,
                    args=(side,),
                    name=f"{self.robot_id}_{side}_drag",
                    daemon=True,
                )
                self._threads[side] = thread
                thread.start()
            self._connected = True
            self._wait_for_initial_state()
            if self.return_to_start_enabled:
                self.capture_home()
        except Exception:
            self.close()
            raise

    def read_state(self) -> RobotState:
        self._raise_if_unhealthy()
        with self._lock:
            missing = [side for side in SIDES if side not in self._follower_state]
            if missing:
                raise RuntimeError(
                    f"YAM drag follower state is not ready for sides: {missing}"
                )
            states = [self._follower_state[side] for side in SIDES]
            return RobotState(
                joint_names=self.joint_names,
                stamp_ns=min(state.stamp_ns for state in states),
                qpos=np.concatenate([state.qpos for state in states]).astype(
                    np.float32
                ),
                qvel=np.concatenate([state.qvel for state in states]).astype(
                    np.float32
                ),
                effort=np.concatenate([state.effort for state in states]).astype(
                    np.float32
                ),
            )

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        del action_space, action
        raise RuntimeError(
            "YAM drag mode disables external position commands; manually drag the follower arms instead"
        )

    def set_teach_mode(self) -> None:
        if not self.is_connected:
            raise RuntimeError("YAM drag driver is not connected")
        for side in SIDES:
            self._enter_drag_mode(side)

    def set_position_mode(self) -> None:
        raise RuntimeError("YAM drag mode has no position-control mode")

    def capture_home(self) -> None:
        """Use the current follower poses as the future return-to-start target."""
        if not self.return_to_start_enabled:
            raise RuntimeError("YAM drag return-to-start is disabled")
        self._raise_if_unhealthy()
        with self._lock:
            if not self.is_connected:
                raise RuntimeError("YAM drag is not connected")
            if set(self._follower_state) != set(SIDES):
                raise RuntimeError(
                    "YAM drag follower state is not ready for Home capture"
                )
            self._start_pose = {
                side: self._follower_state[side].qpos.copy() for side in SIDES
            }

    def reset_home(self) -> None:
        """Return to the captured pose and independently restore both followers."""
        if not self.return_to_start_enabled:
            raise RuntimeError("YAM drag return-to-start is disabled")
        with _home_motion(self):
            try:
                self._raise_if_unhealthy()
                if not self.is_connected:
                    raise RuntimeError("YAM drag is not connected")
                start = {side: self._read_follower(side).qpos for side in SIDES}
                with self._lock:
                    target = {
                        side: pose.copy() for side, pose in self._start_pose.items()
                    }
                if set(target) != set(SIDES):
                    raise RuntimeError("YAM drag start pose is not ready for return")
                self._validate_return_to_start(start, target)
                for side in SIDES:
                    self._set_gripper_assist_torque(side, 0.0)
                _run_home_trajectory(
                    self,
                    start,
                    target,
                    self.return_to_start_duration_s,
                    self.hz,
                    self._command_follower_position,
                )
                with self._lock:
                    for side in SIDES:
                        self._follower_state[side] = self._read_follower(side)
            finally:
                errors = []
                for side in SIDES:
                    try:
                        self._enter_drag_mode(side)
                    except Exception as exc:
                        errors.append(f"{side}: {exc}")
                        with self._lock:
                            self._errors[side] = f"failed to restore drag mode: {exc}"
                if errors:
                    raise RuntimeError(
                        "YAM drag mode restoration failed: " + "; ".join(errors)
                    )

    def drain_actions(self) -> list[dict[str, Any]]:
        """Return observed follower poses to store as drag-trajectory labels."""
        with self._lock:
            items = list(self._pending_actions)
            self._pending_actions.clear()
        return items

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        with self._home_lock:
            errors: list[str] = []
            for thread in self._threads.values():
                if thread is not threading.current_thread():
                    thread.join(timeout=2.0)
            self._threads.clear()
            for side in reversed(SIDES):
                follower = self._followers.pop(side, None)
                if follower is None:
                    continue
                try:
                    idle = getattr(follower, "enter_gravity_comp_idle", None)
                    if callable(idle):
                        idle()
                except Exception as exc:  # noqa: BLE001 - release the other CAN chain too
                    errors.append(f"follower.{side}.idle: {exc}")
                try:
                    follower.close()
                except Exception as exc:  # noqa: BLE001 - release the other CAN chain too
                    errors.append(f"follower.{side}.close: {exc}")
            self._connected = False
            self._closed = True
            if errors:
                raise RuntimeError("YAM drag cleanup errors: " + "; ".join(errors))

    def status(self) -> dict[str, Any]:
        self._check_runtime_health()
        with self._lock:
            errors = dict(self._errors)
            pending_actions = len(self._pending_actions)
            gripper_velocity_m_s = {
                side: (
                    float(self._follower_state[side].qvel[-1])
                    if side in self._follower_state
                    else None
                )
                for side in SIDES
            }
            commanded_gripper_torque_nm = dict(
                self._gripper_motion_assist_commanded_torque_nm
            )
        ready = (
            self.is_connected
            and all(event.is_set() for event in self._side_ready.values())
            and not errors
        )
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "connected": self.is_connected,
            "ready": ready,
            "ready_for_recording": ready and not self._home_requested.is_set(),
            "action_space": "abs_qpos",
            "action_dim": self.action_dim,
            "action_semantics": "observed_follower_qpos",
            "drag_mode": "gravity_compensation",
            "ee_mass": self.ee_mass,
            "gripper_motion_assist": {
                "enabled": self.gripper_motion_assist_enabled,
                "torque_nm": self.gripper_motion_assist_torque_nm,
                "velocity_threshold_m_s": self.gripper_motion_assist_velocity_threshold_m_s,
                "direction": self.gripper_motion_assist_direction,
                "max_torque_nm": MAX_GRIPPER_MOTION_ASSIST_TORQUE_NM,
                "active_only_while_moving": True,
                "velocity_m_s": gripper_velocity_m_s,
                "commanded_torque_nm": commanded_gripper_torque_nm,
            },
            # None means the I2RT arm config supplies the value.
            "drag_tuning": dict(self.drag_tuning),
            "supports_reset_home": self.return_to_start_enabled
            and set(self._start_pose) == set(SIDES),
            "auto_return_after_episode": self.return_to_start_enabled
            and set(self._start_pose) == set(SIDES),
            "home_return_label": "this run's startup pose",
            "home_return_duration_s": self.return_to_start_duration_s,
            "returning_home": self._returning_home,
            "hz": self.hz,
            "follower_can": dict(self.follower_can),
            "errors": errors,
            "pending_actions": pending_actions,
        }

    def _make_follower(self, side: str) -> Any:
        channel = self.follower_can[side]
        if self._robot_factory is not None:
            return self._robot_factory(
                role="follower",
                side=side,
                channel=channel,
                arm=self.follower_arm,
                gripper=self.follower_gripper,
                ee_mass=self.ee_mass,
                sim=self.sim,
                **self.drag_tuning,
            )

        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        return get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name(self.follower_arm),
            gripper_type=GripperType.from_string_name(self.follower_gripper),
            ee_mass=self.ee_mass,
            zero_gravity_mode=True,
            sim=self.sim,
            enable_auto_recovery=self.enable_auto_recovery,
            **self.drag_tuning,
        )

    def _enter_drag_mode(self, side: str) -> None:
        idle = getattr(self._followers[side], "enter_gravity_comp_idle", None)
        if not callable(idle):
            raise RuntimeError(
                f"YAM drag follower {side} does not provide enter_gravity_comp_idle()"
            )
        idle()
        self._set_gripper_assist_torque(side, 0.0)

    def _set_gripper_assist_torque(self, side: str, torque_nm: float) -> None:
        """Set, or clear, the direct gripper torque without changing arm mode."""
        assist = getattr(self._followers[side], "set_gripper_assist_torque", None)
        if not callable(assist):
            raise RuntimeError(
                f"YAM drag follower {side} does not provide set_gripper_assist_torque()"
            )
        assist(float(torque_nm))
        with self._lock:
            self._gripper_motion_assist_commanded_torque_nm[side] = float(torque_nm)

    def _update_gripper_motion_assist(
        self, side: str, gripper_velocity_m_s: float
    ) -> None:
        """Assist an operator's already-started gripper motion, never hold it."""
        with self._lock:
            returning_home = self._returning_home
        torque_nm = 0.0
        if (
            self.gripper_motion_assist_enabled
            and not returning_home
            and np.isfinite(gripper_velocity_m_s)
            and abs(gripper_velocity_m_s)
            >= self.gripper_motion_assist_velocity_threshold_m_s
        ):
            torque_nm = float(
                self.gripper_motion_assist_direction
                * np.copysign(
                    self.gripper_motion_assist_torque_nm, gripper_velocity_m_s
                )
            )
        self._set_gripper_assist_torque(side, torque_nm)

    def _command_follower_position(self, side: str, physical_qpos: np.ndarray) -> None:
        command = np.asarray(physical_qpos, dtype=np.float64).reshape(SIDE_DOF).copy()
        command[-1] = np.clip(command[-1] / self.gripper_width_m, 0.0, 1.0)
        self._followers[side].command_joint_pos(command)

    def _validate_return_to_start(
        self, start: Mapping[str, np.ndarray], target: Mapping[str, np.ndarray]
    ) -> None:
        for side in SIDES:
            delta = np.abs(target[side] - start[side])
            if np.any(delta[:ARM_DOF] > self.return_to_start_max_joint_delta_rad):
                raise RuntimeError(
                    f"YAM drag return-to-start refused on {side}: joint displacement exceeds "
                    f"{self.return_to_start_max_joint_delta_rad:.3f} rad"
                )
            if delta[-1] > self.return_to_start_max_gripper_delta_m:
                raise RuntimeError(
                    f"YAM drag return-to-start refused on {side}: gripper displacement exceeds "
                    f"{self.return_to_start_max_gripper_delta_m:.3f} m"
                )
            info = self._followers[side].get_robot_info()
            limits = np.asarray(info.get("joint_limits"), dtype=np.float64)
            if (
                limits.shape != (ARM_DOF, 2)
                or np.any(target[side][:ARM_DOF] < limits[:, 0])
                or np.any(target[side][:ARM_DOF] > limits[:, 1])
            ):
                raise RuntimeError(
                    f"YAM drag return-to-start refused on {side}: target violates joint limits"
                )

    def _wait_for_initial_state(self) -> None:
        deadline = time.monotonic() + self.connect_timeout_s
        for side in SIDES:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._side_ready[side].wait(remaining):
                self._raise_if_unhealthy()
                raise TimeoutError(
                    f"timed out waiting for YAM drag {side} follower state"
                )
        self._raise_if_unhealthy()

    def _drag_loop(self, side: str) -> None:
        period_s = 1.0 / self.hz
        initial_deadline = time.monotonic() + self.connect_timeout_s
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                try:
                    with self._motion_locks[side]:
                        if self._stop.is_set():
                            break
                        follower = self._read_follower(side)
                        self._update_gripper_motion_assist(
                            side, float(follower.qvel[-1])
                        )
                except RuntimeError:
                    if (
                        not self._side_ready[side].is_set()
                        and time.monotonic() < initial_deadline
                    ):
                        self._stop.wait(min(period_s, 0.05))
                        continue
                    raise
                with self._lock:
                    self._follower_state[side] = follower
                    self._action_dirty.add(side)
                    if self._action_dirty == set(SIDES):
                        self._action_sequence += 1
                        self._pending_actions.append(
                            {
                                "action": np.concatenate(
                                    [self._follower_state[name].qpos for name in SIDES]
                                ).astype(np.float32),
                                "action_space": "abs_qpos",
                                "hz": self.hz,
                                "stamp_ns": max(
                                    self._follower_state[name].stamp_ns
                                    for name in SIDES
                                ),
                                "metadata": {
                                    "source": "yam_drag_follower_state",
                                    "action_semantics": "observed_follower_qpos",
                                    "mode": "gravity_compensation",
                                    "sequence": self._action_sequence,
                                    "updated_side": side,
                                    "side_stamp_ns": {
                                        name: self._follower_state[name].stamp_ns
                                        for name in SIDES
                                    },
                                },
                            }
                        )
                        self._action_dirty.clear()
                self._side_ready[side].set()
                delay = period_s - (time.monotonic() - tick)
                if delay > 0:
                    self._stop.wait(delay)
        except Exception as exc:  # noqa: BLE001 - surface a worker failure through status
            with self._lock:
                self._errors[side] = f"{type(exc).__name__}: {exc}"
            self._side_ready[side].set()

    def _read_follower(self, side: str) -> _SideState:
        obs = _observations(self._followers[side], self.state_timeout_s)
        arm_pos = _vector(obs, "joint_pos", ARM_DOF)
        arm_vel = _vector(obs, "joint_vel", ARM_DOF)
        arm_effort = _vector(obs, "joint_eff", ARM_DOF)
        gripper_pos = _scalar(obs, "gripper_pos")
        gripper_vel = _scalar(obs, "gripper_vel", default=np.nan)
        gripper_effort = _scalar(obs, "gripper_eff", default=np.nan)
        return _SideState(
            stamp_ns=int(_scalar(obs, "timestamp") * 1_000_000_000),
            qpos=np.concatenate([arm_pos, [gripper_pos * self.gripper_width_m]]),
            qvel=np.concatenate([arm_vel, [gripper_vel * self.gripper_width_m]]),
            effort=np.concatenate([arm_effort, [gripper_effort]]),
        )

    def _raise_if_unhealthy(self) -> None:
        self._check_runtime_health()
        with self._lock:
            errors = dict(self._errors)
        if errors:
            raise RuntimeError(f"YAM drag failed: {errors}")

    def _check_runtime_health(self) -> None:
        if not self._connected or self._closed:
            return
        with self._lock:
            for side, robot in self._followers.items():
                error = _runtime_error(robot, self.state_timeout_s)
                if error:
                    self._errors[side] = error


def yam_topics(robot_id: str) -> dict[str, str]:
    prefix = f"/prometheus/robots/{robot_id}"
    return {name: f"{prefix}/{suffix}" for name, suffix in ROBOT_TOPIC_SUFFIXES.items()}


def resolve_robot(**robot: Any) -> dict[str, Any]:
    config = dict(robot)
    config["type"] = "yam"
    config.update(runtime_config(config))
    config["sensors"] = sensor_config(config)
    config["streams"] = streams(config)
    return config


def resolve_drag_robot(**robot: Any) -> dict[str, Any]:
    """Resolve the follower-only gravity-compensation YAM collection mode."""
    config = dict(robot)
    config["type"] = "yam_drag"
    config.update(drag_runtime_config(config))
    config["sensors"] = sensor_config(config)
    config["streams"] = streams(config)
    return config


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    robot_id = str(robot["id"])
    robot_cfg = _robot_config(robot)
    leader_can = _side_mapping(robot_cfg, "leader_can")
    follower_can = _side_mapping(robot_cfg, "follower_can")
    topics = yam_topics(robot_id)
    state_publish_hz = float(robot_cfg["state_publish_hz"])
    control_domain_id = int(robot["control_domain_id"])
    action_space = str(robot_cfg.get("action_space", "abs_qpos"))
    if action_space not in YAM_ACTION_SPACES:
        raise ValueError(f"YAM action_space must be one of {YAM_ACTION_SPACES}")
    return {
        "driver": {
            "_target_": "prometheus.nodes.robots.yam.YamBimanualDriver",
            "robot_id": robot_id,
            "leader_left_can": leader_can["left"],
            "leader_right_can": leader_can["right"],
            "follower_left_can": follower_can["left"],
            "follower_right_can": follower_can["right"],
            "leader_arm": robot_cfg.get("leader_arm", "yam"),
            "follower_arm": robot_cfg.get("follower_arm", "yam_ultra_2"),
            "leader_gripper": robot_cfg.get("leader_gripper", "yam_teaching_handle"),
            "follower_gripper": robot_cfg.get("follower_gripper", "linear_4310"),
            "gripper_width_m": float(robot_cfg.get("gripper_width_m", GRIPPER_WIDTH_M)),
            "hz": state_publish_hz,
            "control_hz": float(robot_cfg.get("control_hz", state_publish_hz)),
            "connect_timeout_s": float(robot_cfg.get("connect_timeout_s", 15.0)),
            "state_timeout_s": float(robot_cfg.get("state_timeout_s", 0.5)),
            "alignment_duration_s": float(robot_cfg.get("alignment_duration_s", 3.0)),
            "bilateral_kp": float(robot_cfg.get("bilateral_kp", 0.0)),
            "teleoperation_enabled": bool(robot_cfg.get("teleoperation_enabled", True)),
            "require_enable_button": bool(robot_cfg.get("require_enable_button", True)),
            "enable_auto_recovery": bool(robot_cfg.get("enable_auto_recovery", False)),
            "home_path": robot_cfg.get("home_path"),
            "home_return_enabled": bool(robot_cfg.get("home_return_enabled", False)),
            "home_return_duration_s": float(
                robot_cfg.get("home_return_duration_s", 5.0)
            ),
            "home_return_max_joint_delta_rad": float(
                robot_cfg.get("home_return_max_joint_delta_rad", 2.0)
            ),
            "home_return_max_gripper_delta_m": float(
                robot_cfg.get("home_return_max_gripper_delta_m", GRIPPER_WIDTH_M)
            ),
            "sim": bool(robot_cfg.get("sim", False)),
        },
        "owner": {
            "_target_": "prometheus.nodes.robots.owner.RobotOwnerNode",
            "robot_id": robot_id,
            "topics": {
                "state": topics["state"],
                "control": topics["control"],
                "result": topics["result"],
                "status": topics["status"],
            },
            "hz": state_publish_hz,
            "ros_domain_id": control_domain_id,
            "action_dims": ACTION_DIMS,
        },
        "client": {
            "_target_": "prometheus.nodes.robots.client.RobotControlClient",
            "action_space": action_space,
            "action_dim": 14,
            "control_topic": topics["control"],
            "result_topic": topics["result"],
            "command_source": f"{robot_id}_workflow",
            "ros_domain_id": control_domain_id,
        },
    }


def drag_runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    robot_id = str(robot["id"])
    robot_cfg = _drag_robot_config(robot)
    follower_can = _side_mapping(robot_cfg, "follower_can")
    topics = yam_topics(robot_id)
    state_publish_hz = float(robot_cfg["state_publish_hz"])
    control_domain_id = int(robot["control_domain_id"])
    action_space = str(robot_cfg.get("action_space", "abs_qpos"))
    if action_space not in YAM_ACTION_SPACES:
        raise ValueError(f"YAM drag action_space must be one of {YAM_ACTION_SPACES}")
    return {
        "driver": {
            "_target_": "prometheus.nodes.robots.yam.YamDragDriver",
            "robot_id": robot_id,
            "follower_left_can": follower_can["left"],
            "follower_right_can": follower_can["right"],
            "follower_arm": robot_cfg.get("follower_arm", "yam_ultra_2"),
            "follower_gripper": robot_cfg.get("follower_gripper", "linear_4310"),
            "gripper_width_m": float(robot_cfg.get("gripper_width_m", GRIPPER_WIDTH_M)),
            "ee_mass": robot_cfg.get("ee_mass"),
            "hz": state_publish_hz,
            "connect_timeout_s": float(robot_cfg.get("connect_timeout_s", 15.0)),
            "state_timeout_s": float(robot_cfg.get("state_timeout_s", 0.5)),
            "enable_auto_recovery": bool(robot_cfg.get("enable_auto_recovery", False)),
            "use_coulomb_friction": bool(robot_cfg.get("use_coulomb_friction", False)),
            "coulomb_friction": robot_cfg.get("coulomb_friction"),
            "grav_comp_kd": robot_cfg.get("grav_comp_kd"),
            "gravity_comp_factor": robot_cfg.get("gravity_comp_factor"),
            "gripper_motion_assist_enabled": bool(
                robot_cfg.get("gripper_motion_assist_enabled", True)
            ),
            "gripper_motion_assist_torque_nm": float(
                robot_cfg.get("gripper_motion_assist_torque_nm", 0.05)
            ),
            "gripper_motion_assist_velocity_threshold_m_s": float(
                robot_cfg.get("gripper_motion_assist_velocity_threshold_m_s", 0.002)
            ),
            "gripper_motion_assist_direction": int(
                robot_cfg.get("gripper_motion_assist_direction", 1)
            ),
            "return_to_start_enabled": bool(
                robot_cfg.get("return_to_start_enabled", True)
            ),
            "return_to_start_duration_s": float(
                robot_cfg.get("return_to_start_duration_s", 8.0)
            ),
            "return_to_start_max_joint_delta_rad": float(
                robot_cfg.get("return_to_start_max_joint_delta_rad", 2.0)
            ),
            "return_to_start_max_gripper_delta_m": float(
                robot_cfg.get("return_to_start_max_gripper_delta_m", GRIPPER_WIDTH_M)
            ),
            "sim": bool(robot_cfg.get("sim", False)),
        },
        "owner": {
            "_target_": "prometheus.nodes.robots.owner.RobotOwnerNode",
            "robot_id": robot_id,
            "topics": {
                "state": topics["state"],
                "control": topics["control"],
                "result": topics["result"],
                "status": topics["status"],
            },
            "hz": state_publish_hz,
            "ros_domain_id": control_domain_id,
            "action_dims": ACTION_DIMS,
        },
        "client": {
            "_target_": "prometheus.nodes.robots.client.RobotControlClient",
            "action_space": action_space,
            "action_dim": 14,
            "control_topic": topics["control"],
            "result_topic": topics["result"],
            "command_source": f"{robot_id}_workflow",
            "ros_domain_id": control_domain_id,
        },
    }


def sensor_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    from prometheus.nodes.sensors.realsense import (
        runtime_config as build_sensor_runtime,
    )

    return build_sensor_runtime(robot)


def streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    from prometheus.nodes.sensors.realsense import streams as build_sensor_streams

    items = robot_streams(robot)
    items.update(build_sensor_streams(robot))
    return items


def robot_streams(robot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "robot_state": {
            "topic": yam_topics(str(robot["id"]))["state"],
            "msg_type": "sensor_msgs.msg.JointState",
            "domain": "robot",
            "ros_domain_id": int(robot["control_domain_id"]),
        }
    }


def _robot_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    value = robot.get("robot")
    if not isinstance(value, Mapping):
        raise TypeError("YAM config requires a robot mapping")
    config = dict(value)
    missing = {"state_publish_hz", "leader_can", "follower_can"}.difference(config)
    if missing:
        raise ValueError(f"YAM robot config missing required keys: {sorted(missing)}")
    return config


def _drag_robot_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    value = robot.get("robot")
    if not isinstance(value, Mapping):
        raise TypeError("YAM drag config requires a robot mapping")
    config = dict(value)
    missing = {"state_publish_hz", "follower_can"}.difference(config)
    if missing:
        raise ValueError(f"YAM drag config missing required keys: {sorted(missing)}")
    return config


def _drag_joint_values(value: Any, name: str) -> list[float] | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=np.float64)
    if (
        values.shape != (ARM_DOF,)
        or not np.isfinite(values).all()
        or (values < 0).any()
    ):
        raise ValueError(
            f"YAM drag {name} must contain six finite, non-negative values (J1-J6)"
        )
    return values.tolist()


def _side_mapping(robot: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = robot.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"YAM robot config requires {name}.left and {name}.right")
    missing = set(SIDES).difference(value)
    if missing:
        raise ValueError(f"YAM robot config {name} missing sides: {sorted(missing)}")
    return {side: value[side] for side in SIDES}


def _vector(observations: Mapping[str, Any], name: str, size: int) -> np.ndarray:
    if name not in observations:
        raise RuntimeError(f"I2RT observation missing {name!r}")
    values = np.asarray(observations[name], dtype=np.float64).reshape(-1)
    if values.size != size:
        raise RuntimeError(
            f"I2RT observation {name!r} expected {size} values, got {values.size}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"I2RT observation {name!r} contains NaN or infinity")
    return values.copy()


def _scalar(
    observations: Mapping[str, Any], name: str, *, default: float | None = None
) -> float:
    if name not in observations:
        if default is not None:
            return float(default)
        raise RuntimeError(f"I2RT observation missing {name!r}")
    values = np.asarray(observations[name], dtype=np.float64).reshape(-1)
    if values.size != 1:
        raise RuntimeError(
            f"I2RT observation {name!r} expected one value, got {values.size}"
        )
    if default is None and not np.isfinite(values[0]):
        raise RuntimeError(f"I2RT observation {name!r} contains NaN or infinity")
    return float(values[0])
