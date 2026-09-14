from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from prometheus.nodes.robots.control import RobotTopics, json_dumps
from prometheus.nodes.robots.robot_node import RobotState


class ModeDriver:
    def __init__(self) -> None:
        self.connected = False
        self.modes: list[str] = []

    def connect(self) -> None:
        self.connected = True

    def read_state(self) -> RobotState:
        return RobotState(joint_names=["joint_0"], qpos=np.zeros(1, dtype=np.float32))

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        return None

    def set_teach_mode(self) -> None:
        self.modes.append("teach")

    def set_position_mode(self) -> None:
        self.modes.append("position")

    def close(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return {"modes": list(self.modes)}


class UnsupportedModeDriver:
    def connect(self) -> None:
        return None

    def read_state(self) -> RobotState:
        return RobotState(joint_names=["joint_0"], qpos=np.zeros(1, dtype=np.float32))

    def send_action(self, action_space: str, action: np.ndarray) -> None:
        return None

    def close(self) -> None:
        return None

    def status(self) -> dict[str, object]:
        return {}


class _CapturingPublisher:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def publish(self, msg: object) -> None:
        self.messages.append(json.loads(msg.data))


class _FakeClock:
    def now(self) -> SimpleNamespace:
        return SimpleNamespace(to_msg=lambda: SimpleNamespace(sec=1, nanosec=0))


class _FakeNode:
    context = None

    def __init__(self, name: str, *, context: object | None = None) -> None:
        self.name = name
        self.context = context
        self.result_pub = _CapturingPublisher()

    def create_publisher(self, msg_type: object, topic: str, qos: int) -> _CapturingPublisher | object:
        if topic.endswith("control_result"):
            return self.result_pub
        return _CapturingPublisher()

    def create_subscription(self, msg_type: object, topic: str, callback: object, qos: int) -> None:
        return None

    def create_timer(self, period: float, callback: object) -> None:
        return None

    def get_clock(self) -> _FakeClock:
        return _FakeClock()

    def destroy_node(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _ros_stubs(monkeypatch):
    fake_rclpy = ModuleType("rclpy")
    fake_executors = ModuleType("rclpy.executors")
    fake_node_mod = ModuleType("rclpy.node")
    fake_sensor_msgs = ModuleType("sensor_msgs.msg")
    fake_std_msgs = ModuleType("std_msgs.msg")

    class SingleThreadedExecutor:
        def __init__(self, context: object | None = None) -> None:
            self.context = context

        def add_node(self, node: object) -> None:
            return None

        def spin_once(self, timeout_sec: float = 0.0) -> None:
            return None

        def remove_node(self, node: object) -> None:
            return None

        def shutdown(self) -> None:
            return None

    class JointState:
        pass

    class String:
        def __init__(self) -> None:
            self.data = ""

    fake_executors.SingleThreadedExecutor = SingleThreadedExecutor
    fake_node_mod.Node = _FakeNode
    fake_sensor_msgs.JointState = JointState
    fake_std_msgs.String = String
    fake_rclpy.Context = lambda: SimpleNamespace(ok=lambda: True)
    fake_rclpy.init = lambda **kwargs: None
    fake_rclpy.ok = lambda: True
    fake_rclpy.shutdown = lambda **kwargs: None

    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.executors", fake_executors)
    monkeypatch.setitem(sys.modules, "rclpy.node", fake_node_mod)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", fake_sensor_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", fake_std_msgs)


def test_owner_set_teach_mode_invokes_driver():
    owner, driver = _owner_with_driver(ModeDriver())
    result = _dispatch(owner, command="set_teach_mode", command_source="workflow")

    assert result["accepted"] is True
    assert result["status"] == "teach_mode_set"
    assert driver.modes == ["teach"]


def test_owner_set_position_mode_invokes_driver():
    owner, driver = _owner_with_driver(ModeDriver())
    result = _dispatch(owner, command="set_position_mode", command_source="workflow")

    assert result["accepted"] is True
    assert result["status"] == "position_mode_set"
    assert driver.modes == ["position"]


def test_owner_mode_command_rejects_missing_driver_method():
    owner, _driver = _owner_with_driver(UnsupportedModeDriver())

    result = _dispatch(owner, command="set_teach_mode", command_source="workflow")

    assert result["accepted"] is False
    assert result["status"] == "error"
    assert "does not support set_teach_mode" in result["message"]


def test_owner_mode_command_rejects_wrong_command_source():
    owner, driver = _owner_with_driver(ModeDriver())
    owner.active_command_source = "other_client"
    owner.active_command_source_stamp_ns = owner._now_ns()

    result = _dispatch(owner, command="set_teach_mode", command_source="workflow")

    assert result["accepted"] is False
    assert result["status"] == "error"
    assert "active command_source" in result["message"]
    assert driver.modes == []


def _owner_with_driver(driver: object):
    from prometheus.nodes.robots.owner import RobotOwnerNode

    fake_node = _FakeNode("test_owner")
    owner = RobotOwnerNode(
        driver=driver,
        robot_id="test_robot",
        topics=RobotTopics(
            state="/robot/state",
            control="/robot/control",
            result="/robot/control_result",
            status="/robot/status",
        ),
        hz=30.0,
        action_dims={"joint": 1},
        node=fake_node,
    )
    owner.driver.connect()
    owner._started = True
    return owner, driver


def _dispatch(owner, *, command: str, command_source: str) -> dict[str, object]:
    msg = owner.String()
    msg.data = json_dumps(
        {
            "request_id": f"{command_source}:1",
            "client_stamp_ns": owner._now_ns(),
            "command": command,
            "command_source": command_source,
        }
    )
    owner._control_cb(msg)
    messages = owner.result_pub.messages
    assert messages
    return messages[-1]
