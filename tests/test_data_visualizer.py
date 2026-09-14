import numpy as np

from prometheus.data.visualizer import encode_png
from prometheus.sessions.data import DataSession


def test_data_visualizer_entrypoint_and_png_encoding():
    data = DataSession(
        {"camera_color": {"topic": "/camera", "msg_type": "sensor_msgs.msg.Image"}},
        bridge={"enabled": False},
    )

    viewer = data.visualize(port=8765, refresh_ms=100)
    payload = encode_png(np.zeros((4, 6, 3), dtype=np.uint8))

    assert viewer.url == "http://127.0.0.1:8765"
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
