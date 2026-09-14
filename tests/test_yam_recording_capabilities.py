from types import SimpleNamespace

import numpy as np
import pytest

from prometheus.data.raw_episode import EPISODE_DIRS, RawEpisodeWriter
from prometheus.data.types import DataSample


@pytest.mark.parametrize("requires_eef", [False, True])
def test_joint_only_recording_plots_and_declared_completeness(tmp_path, requires_eef):
    for name in EPISODE_DIRS:
        (tmp_path / name).mkdir()
    streams = {
        "robot_state": {"topic": "/yam/state", "msg_type": "sensor_msgs.msg.JointState"}
    }
    if requires_eef:
        streams["left_eef"] = {
            "topic": "/yam/left_eef",
            "msg_type": "geometry_msgs.msg.PoseStamped",
        }
    writer = RawEpisodeWriter(tmp_path, streams, fps=30, enable_plots=True)
    for stamp in (100_000_000, 200_000_000):
        writer.write_sample(
            DataSample(
                name="robot_state",
                stamp_ns=stamp,
                recv_ns=stamp,
                stream=writer.streams["robot_state"],
                msg=SimpleNamespace(
                    position=np.zeros(14), velocity=np.zeros(14), effort=np.zeros(14)
                ),
            )
        )
    writer.finalize()
    assert writer.robot.incomplete_counts() == {
        "left": 2 if requires_eef else 0,
        "right": 0,
    }
    assert writer.robot.plot_error == ""
    assert len(writer.robot.plot_outputs) == 1
    from pathlib import Path

    assert Path(writer.robot.plot_outputs[0]).is_file()
