from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from prometheus.nodes.robots.robot_node import RobotState
from prometheus.nodes.robots.noop import NoopRobotClient as NoopRobotClient






class NoopRobotDriver:
    """RobotDriver test adapter with no ROS or hardware dependency."""

    def __init__(self, *, action_space: str = "noop_action", action_dim: int = 2):
        self.action_space = str(action_space)
        self.action_dim = int(action_dim)
        self.connected = False
        self.closed = False
        self.actions: list[list[float]] = []

    def connect(self) -> None:
        self.connected = True

    def read_state(self) -> RobotState:
        return RobotState(
            joint_names=[f"joint_{index}" for index in range(self.action_dim)],
            qpos=np.zeros(self.action_dim, dtype=np.float32),
        )

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        if action_space != self.action_space:
            raise ValueError(f"expected action_space {self.action_space!r}, got {action_space!r}")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {values.shape}")
        self.actions.append([float(value) for value in values])

    def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "connected": self.connected,
            "closed": self.closed,
            "sent_actions": len(self.actions),
        }


class NoopRobotOwner:
    """Owner runtime test double that wraps a NoopRobotDriver-like object."""

    def __init__(self, *, driver: Any, robot_id: str = "noop"):
        self.driver = driver
        self.robot_id = str(robot_id)
        self.started = False
        self.ready = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.driver.connect()
        self.started = True
        self.ready = True

    def wait_ready(self, timeout_s: float) -> None:
        if not self.ready:
            raise TimeoutError(f"{self.robot_id} owner is not ready")

    def stop(self) -> None:
        self.stopped = True
        self.ready = False

    def close(self) -> None:
        self.driver.close()
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "robot_id": self.robot_id,
            "started": self.started,
            "ready": self.ready,
            "stopped": self.stopped,
            "closed": self.closed,
        }


class NoopRuntimeComponent:
    """Generic lifecycle component for session tests."""

    def __init__(self, *, name: str, data: Any | None = None):
        self.name = str(name)
        self.data = data
        self.started = False
        self.ready = False
        self.stopped = False
        self.closed = False

    def start(self) -> None:
        self.started = True
        self.ready = True

    def wait_ready(self, timeout_s: float) -> None:
        if not self.ready:
            raise TimeoutError(f"{self.name} is not ready")

    def stop(self) -> None:
        self.stopped = True
        self.ready = False

    def close(self) -> None:
        self.closed = True

    def status(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "has_data": self.data is not None,
            "started": self.started,
            "ready": self.ready,
            "stopped": self.stopped,
            "closed": self.closed,
        }


@dataclass
class RecordedChunk:
    """Raw recorder payload; no scheduler or model execution."""

    actions: np.ndarray
    action_space: str
    hz: float
    metadata: dict[str, Any] = field(default_factory=dict)
