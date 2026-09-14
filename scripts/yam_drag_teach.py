#!/usr/bin/env python3
"""Run the two YAM followers in gravity-compensation drag mode only.

This intentionally starts neither ROS, cameras, data recording, pedals, nor
leader arms.  It is for checking how the followers feel before a collection
run.  Ctrl-C closes the motor chains, which releases torque.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

from prometheus.nodes.robots.yam import YamDragDriver


DEFAULT_YAM_DRAG_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "prometheus" / "configs" / "robot" / "yam_drag.yaml"
)
DEFAULT_GRIPPER_MOTION_ASSIST_TORQUE_NM = 0.05
DEFAULT_GRIPPER_MOTION_ASSIST_VELOCITY_THRESHOLD_M_S = 0.002


def _six_values(value: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be six comma-separated numbers") from exc
    if len(values) != 6:
        raise argparse.ArgumentTypeError("must contain exactly six values for J1-J6")
    return values


def _yaml_six_values(value: object, name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} in yam_drag.yaml must contain six numbers or null")
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} in yam_drag.yaml must contain six numbers or null") from exc
    if len(values) != 6:
        raise ValueError(f"{name} in yam_drag.yaml must contain exactly six values for J1-J6")
    return values


def _load_yam_drag_config(path: Path) -> dict[str, object]:
    """Read the shared drag configuration without starting ROS or CAN."""
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # noqa: BLE001 - show the path and parser error to the operator
        raise RuntimeError(f"could not load YAM drag config {path}: {exc}") from exc
    if not isinstance(raw, Mapping) or not isinstance(raw.get("robot"), Mapping):
        raise ValueError(f"YAM drag config {path} must contain a robot mapping")
    return dict(raw["robot"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start YAM Ultra 2 follower-only gravity-compensation drag teach."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_YAM_DRAG_CONFIG_PATH,
        help=f"shared yam_drag YAML to read (default: {DEFAULT_YAM_DRAG_CONFIG_PATH})",
    )
    parser.add_argument("--left-can", default=None, help="temporary override of YAML follower_can.left")
    parser.add_argument("--right-can", default=None, help="temporary override of YAML follower_can.right")
    parser.add_argument(
        "--ee-mass",
        type=float,
        default=None,
        metavar="KG",
        help="temporary YAML ee_mass override in kg (absolute, not added payload)",
    )
    parser.add_argument(
        "--grav-comp-kd",
        type=_six_values,
        default=None,
        metavar="J1,J2,J3,J4,J5,J6",
        help="temporary YAML J1-J6 idle damping override",
    )
    parser.add_argument(
        "--gravity-comp-factor",
        type=_six_values,
        default=None,
        metavar="J1,J2,J3,J4,J5,J6",
        help="optional gravity compensation multiplier for both arms",
    )
    parser.add_argument(
        "--gripper-motion-assist-torque-nm",
        type=float,
        default=None,
        help=(
            "temporary YAML motion-only gripper assist torque override "
            f"(safety cap: 0.20 Nm; YAML fallback: {DEFAULT_GRIPPER_MOTION_ASSIST_TORQUE_NM} Nm)"
        ),
    )
    parser.add_argument(
        "--gripper-motion-assist-velocity-threshold-m-s",
        type=float,
        default=None,
        help=(
            "temporary YAML minimum jaw speed before assist engages "
            f"(YAML fallback: {DEFAULT_GRIPPER_MOTION_ASSIST_VELOCITY_THRESHOLD_M_S} m/s)"
        ),
    )
    parser.add_argument(
        "--disable-gripper-motion-assist",
        action="store_true",
        dest="disable_gripper_motion_assist",
        help="leave the gripper fully passive (rollback switch)",
    )
    parser.add_argument(
        "--assist-log-hz",
        type=float,
        default=4.0,
        help="terminal print rate for gripper speed and assist torque (default: 4 Hz)",
    )
    parser.add_argument("--hz", type=float, default=None, help="temporary YAML polling-frequency override")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.assist_log_hz <= 0:
        raise ValueError("--assist-log-hz must be positive")
    config_path = args.config.resolve()
    robot_config = _load_yam_drag_config(config_path)
    follower_can = robot_config.get("follower_can")
    if not isinstance(follower_can, Mapping) or not {"left", "right"}.issubset(follower_can):
        raise ValueError(f"YAM drag config {config_path} requires follower_can.left and follower_can.right")

    grav_comp_kd = (
        args.grav_comp_kd
        if args.grav_comp_kd is not None
        else _yaml_six_values(robot_config.get("grav_comp_kd"), "grav_comp_kd")
    )
    gravity_comp_factor = (
        args.gravity_comp_factor
        if args.gravity_comp_factor is not None
        else _yaml_six_values(robot_config.get("gravity_comp_factor"), "gravity_comp_factor")
    )
    ee_mass = args.ee_mass if args.ee_mass is not None else robot_config.get("ee_mass")
    if ee_mass is not None:
        ee_mass = float(ee_mass)
    gripper_motion_assist_enabled = bool(
        robot_config.get("gripper_motion_assist_enabled", True)
    ) and not args.disable_gripper_motion_assist
    gripper_motion_assist_torque_nm = (
        args.gripper_motion_assist_torque_nm
        if args.gripper_motion_assist_torque_nm is not None
        else float(
            robot_config.get(
                "gripper_motion_assist_torque_nm", DEFAULT_GRIPPER_MOTION_ASSIST_TORQUE_NM
            )
        )
    )
    gripper_motion_assist_velocity_threshold_m_s = (
        args.gripper_motion_assist_velocity_threshold_m_s
        if args.gripper_motion_assist_velocity_threshold_m_s is not None
        else float(
            robot_config.get(
                "gripper_motion_assist_velocity_threshold_m_s",
                DEFAULT_GRIPPER_MOTION_ASSIST_VELOCITY_THRESHOLD_M_S,
            )
        )
    )
    gripper_motion_assist_direction = int(robot_config.get("gripper_motion_assist_direction", 1))
    driver = YamDragDriver(
        robot_id="yam_drag_teach",
        follower_left_can=args.left_can or str(follower_can["left"]),
        follower_right_can=args.right_can or str(follower_can["right"]),
        follower_arm=str(robot_config.get("follower_arm", "yam_ultra_2")),
        follower_gripper=str(robot_config.get("follower_gripper", "linear_4310")),
        gripper_width_m=float(robot_config.get("gripper_width_m", 0.096)),
        ee_mass=ee_mass,
        hz=args.hz if args.hz is not None else float(robot_config.get("state_publish_hz", 120.0)),
        connect_timeout_s=float(robot_config.get("connect_timeout_s", 15.0)),
        state_timeout_s=float(robot_config.get("state_timeout_s", 0.5)),
        enable_auto_recovery=bool(robot_config.get("enable_auto_recovery", False)),
        use_coulomb_friction=bool(robot_config.get("use_coulomb_friction", False)),
        coulomb_friction=_yaml_six_values(robot_config.get("coulomb_friction"), "coulomb_friction"),
        grav_comp_kd=grav_comp_kd,
        gravity_comp_factor=gravity_comp_factor,
        gripper_motion_assist_enabled=gripper_motion_assist_enabled,
        gripper_motion_assist_torque_nm=gripper_motion_assist_torque_nm,
        gripper_motion_assist_velocity_threshold_m_s=gripper_motion_assist_velocity_threshold_m_s,
        gripper_motion_assist_direction=gripper_motion_assist_direction,
        # No Home target or position-command path is needed for an inspection
        # session; this script is gravity-compensation only.
        return_to_start_enabled=False,
    )
    try:
        driver.connect()
        status = driver.status()
        print("[ready] YAM drag teach is active: follower arms only, gravity compensation.", flush=True)
        print(f"[ready] config: {config_path}", flush=True)
        print(f"[ready] CAN: {status['follower_can']}", flush=True)
        print(f"[ready] J1-J6 idle damping: {grav_comp_kd}", flush=True)
        print(
            "[ready] gripper motion assist: "
            + (
                f"on, {gripper_motion_assist_torque_nm:.3f} Nm after "
                f"{gripper_motion_assist_velocity_threshold_m_s:.3f} m/s "
                f"(motor-direction factor {gripper_motion_assist_direction:+d})"
                if gripper_motion_assist_enabled
                else "off (passive gripper)"
            ),
            flush=True,
        )
        print(
            f"[ready] end-effector body mass: {'model default' if ee_mass is None else f'{ee_mass} kg'}",
            flush=True,
        )
        print("[ready] No ROS, cameras, recording, pedals, or leader arms were started.", flush=True)
        print("[safety] Before Ctrl-C, physically support both arms: closing releases motor torque.", flush=True)
        print(f"[assist] velocity and torque sign relation follows direction={gripper_motion_assist_direction:+d}.", flush=True)
        assist_log_period_s = 1.0 / args.assist_log_hz
        while True:
            status = driver.status()
            if status["errors"]:
                raise RuntimeError(f"YAM drag teach failed: {status['errors']}")
            assist = status["gripper_motion_assist"]
            velocity = assist["velocity_m_s"]
            torque = assist["commanded_torque_nm"]
            def format_side(side: str) -> str:
                speed = velocity[side]
                speed_text = "waiting" if speed is None else f"{speed:+.5f} m/s"
                return f"{side}=v:{speed_text}, assist:{torque[side]:+.3f} Nm"

            print(f"[assist] {format_side('left')} | {format_side('right')}", flush=True)
            time.sleep(assist_log_period_s)
    except KeyboardInterrupt:
        print("\n[teach] stopping drag-teach session.", flush=True)
        return 0
    finally:
        try:
            driver.close()
        except Exception as exc:  # noqa: BLE001 - report every hardware cleanup failure
            print(f"[teach] cleanup error: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
