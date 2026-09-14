from __future__ import annotations

import numpy as np
import pytest

from prometheus.nodes.sensors.realsense import (
    RealsenseSensorNode,
    _image_message,
    camera_serials,
    camera_stream_names,
    camera_streams,
    normalize_resize,
    normalize_camera_map,
    parse_camera_map,
    runtime_config,
)


class _Header:
    stamp = None
    frame_id = ""


class _Image:
    def __init__(self):
        self.header = _Header()
        self.height = 0
        self.width = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.step = 0
        self.data = b""


def test_image_message_does_not_require_cv_bridge():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    stamp = object()

    msg = _image_message(image, "bgr8", stamp, "camera_color_frame", _Image)

    assert (msg.height, msg.width, msg.step) == (2, 3, 9)
    assert msg.encoding == "bgr8"
    assert msg.header.stamp is stamp
    assert msg.header.frame_id == "camera_color_frame"
    assert msg.data == image.tobytes()


def test_camera_stream_names_follow_official_wrapper_streams():
    assert camera_stream_names("d405") == {
        "color": "d405_color",
        "color_info": "d405_color_info",
        "infra1": "d405_infra1",
        "infra1_info": "d405_infra1_info",
        "infra2": "d405_infra2",
        "infra2_info": "d405_infra2_info",
    }
    assert camera_stream_names("d405", enable_infra=False) == {
        "color": "d405_color",
        "color_info": "d405_color_info",
    }
    assert camera_stream_names("d405", camera_info=False) == {
        "color": "d405_color",
        "infra1": "d405_infra1",
        "infra2": "d405_infra2",
    }


def test_camera_map_validation_is_strict():
    assert parse_camera_map('{"d405":"260422273830"}') == {"d405": "260422273830"}

    with pytest.raises(ValueError, match="ROS-safe"):
        normalize_camera_map({"d405-rgb": "260422273830"})

    with pytest.raises(ValueError, match="is used by"):
        normalize_camera_map({"left": "1", "right": "1"})


def test_sensor_node_uses_pyrealsense2_runtime_options():
    node = RealsenseSensorNode(
        cameras={"d405": "260422273830"},
        ros_domain_id=22,
        publish_hz=15,
        resize=[320, 240],
    )

    status = node.status()

    assert status["cameras"] == {"d405": "260422273830"}
    assert status["fps"] == 30
    assert status["publish_hz"] == 15.0
    assert status["enable_infra"] is True
    assert status["resize"] == {"scale": None, "width": 320, "height": 240}
    assert status["qos"] == "sensor"
    assert status["camera_info"] is True


def test_sensor_node_can_disable_infra_streams():
    node = RealsenseSensorNode(
        cameras={"d405": "260422273830"},
        enable_infra=False,
        ros_domain_id=22,
    )

    assert node.status()["enable_infra"] is False


def test_camera_streams_match_official_wrapper_topics():
    topics = camera_streams("d405", 22)

    assert topics["d405_color"] == {
        "topic": "/prometheus/sensors/cameras/d405/color/image_raw",
        "msg_type": "sensor_msgs.msg.Image",
        "domain": "sensor",
        "ros_domain_id": 22,
    }
    assert topics["d405_color_info"]["topic"] == "/prometheus/sensors/cameras/d405/color/camera_info"
    assert topics["d405_infra1"]["topic"] == "/prometheus/sensors/cameras/d405/infra1/image_rect_raw"
    assert topics["d405_infra2_info"]["topic"] == "/prometheus/sensors/cameras/d405/infra2/camera_info"


def test_camera_streams_can_disable_infra_streams():
    topics = camera_streams("d405", 22, enable_infra=False)

    assert set(topics) == {"d405_color", "d405_color_info"}


def test_camera_streams_can_disable_camera_info_streams():
    topics = camera_streams("d405", 22, camera_info=False)

    assert set(topics) == {"d405_color", "d405_infra1", "d405_infra2"}


def test_runtime_config_accepts_ir_and_resize_fields():
    cfg = runtime_config(
        {
            "sensor_domain_id": 12,
            "cameras": {
                "capture_hz": 30,
                "capture_width": 640,
                "capture_height": 480,
                "publish_hz": 20,
                "ir": False,
                "prepub_resize": {"scale": 0.5},
                "qos": "reliable",
                "camera_info": False,
                "serial_no": {
                    "base_0": "260522274589",
                },
            },
        }
    )

    sensor = cfg["realsense"]
    assert sensor["cameras"] == {"base_0": "260522274589"}
    assert sensor["width"] == 640
    assert sensor["height"] == 480
    assert sensor["enable_infra"] is False
    assert sensor["publish_hz"] == 20.0
    assert sensor["resize"] == {"scale": 0.5}
    assert sensor["qos"] == "reliable"
    assert sensor["camera_info"] is False


def test_camera_serials_accept_serial_no_group():
    assert camera_serials(
        {
            "hz": 30,
            "capture_width": 640,
            "serial_no": {
                "base_0": "260522274589",
            },
        }
    ) == {"base_0": "260522274589"}


def test_resize_can_be_scale_or_width_height():
    scale = normalize_resize(0.5)
    fixed = normalize_resize([320, 240])

    assert scale.scale == 0.5
    assert fixed.width == 320
    assert fixed.height == 240
