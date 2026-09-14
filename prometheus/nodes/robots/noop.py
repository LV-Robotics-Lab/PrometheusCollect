from __future__ import annotations

import time
from collections import deque
from typing import Any, Mapping

import numpy as np


def resolve_robot(**robot: Any) -> dict[str, Any]:
    config = dict(robot)
    config["type"] = "noop"
    config.update(runtime_config(config))
    config["sensors"] = sensor_config(config)
    config["streams"] = streams(config)
    return config


def runtime_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    return {}


def sensor_config(robot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(robot.get("sensors", {}))


def streams(robot: Mapping[str, Any]) -> dict[str, Any]:
    return dict(robot.get("streams", {}))


class NoopRobotClient:
    """Robot-control stand-in for hardware-independent collection tests."""

    def __init__(self, *, action_space: str, action_dim: int, history: int = 1024):
        if not action_space:
            raise ValueError("NoopRobotClient requires action_space")
        if int(action_dim) <= 0:
            raise ValueError("NoopRobotClient requires positive action_dim")
        if int(history) <= 0:
            raise ValueError("NoopRobotClient history must be positive")
        self.action_space = str(action_space)
        self.action_dim = int(action_dim)
        self.history: deque[np.ndarray] = deque(maxlen=int(history))
        self.closed = False
        self.released = False
        self.reset_count = 0
        self.teach_count = 0
        self.position_count = 0

    def send_action(self, action: Any, *, timeout: float = 3.0, wait: bool = True) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("NoopRobotClient is closed")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (self.action_dim,):
            raise ValueError(f"noop action must have shape {(self.action_dim,)}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("noop action contains NaN or inf")
        self.history.append(values.copy())
        return {
            "accepted": True,
            "message": "",
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def spin_once(self, timeout_s: float = 0.0) -> None:
        if timeout_s > 0:
            time.sleep(float(timeout_s))

    def release_control(self, *, timeout: float = 3.0, wait: bool = True) -> dict[str, Any]:
        self.released = True
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def reset_home(self, *, timeout: float = 30.0, wait: bool = True) -> dict[str, Any]:
        self.reset_count += 1
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def set_teach_mode(self, *, timeout: float = 5.0, wait: bool = True) -> dict[str, Any]:
        self.teach_count += 1
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def set_position_mode(self, *, timeout: float = 5.0, wait: bool = True) -> dict[str, Any]:
        self.position_count += 1
        return {
            "accepted": True,
            "message": "",
            "timeout": float(timeout),
            "wait": bool(wait),
        }

    def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "sent_actions": len(self.history),
            "released": self.released,
            "closed": self.closed,
            "reset_count": self.reset_count,
            "teach_count": self.teach_count,
            "position_count": self.position_count,
        }
