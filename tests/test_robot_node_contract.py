from __future__ import annotations

import pytest

from prometheus.nodes.robots.control import (
    RobotTopics,
    checked_action_array,
    robot_topics_from_mapping,
    validate_control_payload,
)
from prometheus.nodes.robots.robot_node import (
    RobotDriver,
    RobotState,
)


def test_robot_driver_protocol_stays_small():
    public_methods = {
        name
        for name, value in RobotDriver.__dict__.items()
        if callable(value) and not name.startswith("_")
    }

    assert public_methods == {"connect", "read_state", "send_action", "close", "status"}


def test_robot_driver_accepts_minimal_adapter():
    class Driver:
        def connect(self):
            pass

        def read_state(self):
            return RobotState(joint_names=[], qpos=[])

        def send_action(self, action_space, action):
            return None

        def close(self):
            pass

        def status(self):
            return {"ready": True}

    assert isinstance(Driver(), RobotDriver)


def test_robot_topics_from_mapping_requires_explicit_topics():
    topics = robot_topics_from_mapping(
        {
            "state": "/robot/state",
            "control": "/robot/control",
            "result": "/robot/control_result",
            "status": "/robot/status",
        }
    )

    assert topics == RobotTopics(
        state="/robot/state",
        control="/robot/control",
        result="/robot/control_result",
        status="/robot/status",
    )


def test_validate_control_payload_is_strict():
    payload = validate_control_payload(
        {
            "request_id": "client:1",
            "client_stamp_ns": 1,
            "command": "send_action",
            "command_source": "client",
            "action_space": "joint",
            "action": [0.0, 1.0],
        }
    )

    assert payload["command"] == "send_action"

    with pytest.raises(ValueError, match="unknown fields"):
        validate_control_payload(
            {
                "request_id": "client:1",
                "client_stamp_ns": 1,
                "command": "release_control",
                "command_source": "client",
                "extra": True,
            }
        )

    reset_payload = validate_control_payload(
        {
            "request_id": "client:2",
            "client_stamp_ns": 2,
            "command": "reset_home",
            "command_source": "client",
        }
    )
    assert reset_payload["command"] == "reset_home"

    teach_payload = validate_control_payload(
        {
            "request_id": "client:3",
            "client_stamp_ns": 3,
            "command": "set_teach_mode",
            "command_source": "client",
        }
    )
    assert teach_payload["command"] == "set_teach_mode"

    position_payload = validate_control_payload(
        {
            "request_id": "client:4",
            "client_stamp_ns": 4,
            "command": "set_position_mode",
            "command_source": "client",
        }
    )
    assert position_payload["command"] == "set_position_mode"


def test_checked_action_array_validates_shape_and_finite_values():
    action = checked_action_array(
        [0.0, 1.0],
        action_space="joint",
        action_dims={"joint": 2},
    )

    assert action.shape == (2,)

    with pytest.raises(ValueError, match="2 values"):
        checked_action_array([0.0], action_space="joint", action_dims={"joint": 2})

    with pytest.raises(ValueError, match="NaN or inf"):
        checked_action_array([float("nan"), 0.0], action_space="joint")
