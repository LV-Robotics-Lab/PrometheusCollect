from __future__ import annotations

import pytest

from prometheus.nodes.data_bridge import RosDataBridge, _streams_by_domain, import_message_type
from prometheus.sessions.data import DataSession


def test_ros_data_bridge_groups_topics_by_ros_domain():
    data = DataSession(
        {
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "prometheus.nodes.robots.robot_node.RobotState",
                "domain": "robot",
                "ros_domain_id": 21,
            },
            "camera": {
                "topic": "/camera/image",
                "msg_type": "prometheus.nodes.robots.robot_node.RobotState",
                "domain": "sensor",
                "ros_domain_id": 22,
            },
        },
        history=2,
    )

    bridge = RosDataBridge(data=data)

    assert bridge.status()["domains"] == {
        "21": ["robot_state"],
        "22": ["camera"],
    }
    assert _streams_by_domain(data.streams.values())[21][0].name == "robot_state"
    assert import_message_type("prometheus.nodes.robots.robot_node.RobotState").__name__ == "RobotState"


def test_ros_data_bridge_requires_topics():
    data = DataSession({}, history=2)
    bridge = RosDataBridge(data=data)

    with pytest.raises(ValueError, match="requires at least one data stream"):
        bridge.start()
