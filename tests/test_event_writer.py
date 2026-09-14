from __future__ import annotations

import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prometheus.data.raw_episode import EventWriter, RawEpisodeWriter, carry_forward_event_values
from prometheus.data.types import DataSample, DataStream
from prometheus.sessions.data import DataSession

_ROBOT_STREAM = DataStream(
    name="robot_state",
    topic="/robot/state",
    msg_type="sensor_msgs.msg.JointState",
)


def test_carry_forward_event_values_basic_toggle():
    values = carry_forward_event_values(
        [100, 200, 300, 400],
        [(200, 1)],
        default_value=0,
    )
    assert values == [0, 1, 1, 1]


def test_carry_forward_event_values_without_toggles():
    values = carry_forward_event_values([100, 200], [], default_value=0)
    assert values == [0, 0]


def test_carry_forward_event_values_applies_toggle_before_first_robot_frame():
    values = carry_forward_event_values([200, 300], [(100, 1)], default_value=0)
    assert values == [1, 1]


def test_carry_forward_event_values_multiple_toggles():
    values = carry_forward_event_values(
        [100, 200, 300, 400],
        [(200, 1), (300, 0)],
        default_value=0,
    )
    assert values == [0, 1, 0, 0]


def test_event_writer_skips_densify_when_default_value_is_none(tmp_path):
    writer = EventWriter(tmp_path, default_value=None)
    writer.write({"name": "record_start", "value": True}, stamp_ns=100_000_000, metadata={})
    writer.ensure_default_from_robot({"left": {"timestamps": [100, 200]}, "right": {"timestamps": []}})

    assert writer.data["names"] == ["record_start"]
    assert writer.data["data"] == [True]
    assert writer.data["timestamps"] == [100]


def test_event_writer_densifies_with_carry_forward(tmp_path):
    writer = EventWriter(tmp_path, default_value=0)
    writer.write({"name": "event_label", "value": 1}, stamp_ns=200_000_000, metadata={"phase": "correcting"})
    writer.ensure_default_from_robot(
        {
            "left": {"timestamps": [100, 200, 300]},
            "right": {"timestamps": [100, 200, 300]},
        }
    )

    assert writer.data["timestamps"] == [100, 200, 300]
    assert writer.data["data"] == [0, 1, 1]
    assert writer.data["names"] == ["event_label", "event_label", "event_label"]
    assert writer.data["metadata"] == [
        {"source": "carry_forward"},
        {"source": "explicit_toggle"},
        {"source": "carry_forward"},
    ]


def test_raw_episode_finalize_writes_dense_event_timeline(tmp_path):
    episode_dir = tmp_path / "0001"
    for name in ("camera", "tactile", "robot", "event", "manifests", "action"):
        (episode_dir / name).mkdir(parents=True)

    writer = RawEpisodeWriter(
        episode_dir,
        {
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            }
        },
        fps=30.0,
        enable_plots=False,
        default_event_value=0,
    )
    for stamp_ms in (100, 200, 300, 400):
        writer.write_sample(_robot_sample(stamp_ms * 1_000_000))
    writer.write_event({"name": "event_label", "value": 1}, stamp_ns=200_000_000, metadata={})
    writer.finalize()

    events = _read_pickle(episode_dir / "event" / "event_dict.pkl")
    assert events["timestamps"] == [100, 200, 300, 400]
    assert events["data"] == [0, 1, 1, 1]


def test_data_session_event_carry_forward_integration(tmp_path):
    data = DataSession(
        {
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            },
        },
        history=8,
        bridge={"enabled": False},
        recording={
            "enabled": True,
            "output_dir": tmp_path,
            "queue_size": 16,
            "fps": 30.0,
            "default_event_value": 0,
            "enable_plots": False,
        },
    )
    data.start()
    data.start_episode("0001")
    data.ingest("robot_state", _joint_state(100_000_000))
    data.ingest("robot_state", _joint_state(200_000_000))
    data.ingest("robot_state", _joint_state(300_000_000))
    data.record_event("event_label", value=1, stamp_ns=200_000_000)
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)
    _wait_until(lambda: data.recorder.details()["queues"].get("event", 0) == 0, timeout_s=1.0)
    data.stop_episode()
    data.close()

    events = _read_pickle(tmp_path / "0001" / "event" / "event_dict.pkl")
    assert events["timestamps"] == [100, 200, 300]
    assert events["data"] == [0, 1, 1]


def _robot_sample(stamp_ns: int) -> DataSample:
    return DataSample(
        name="robot_state",
        stamp_ns=stamp_ns,
        msg=_joint_state(stamp_ns),
        recv_ns=stamp_ns,
        stream=_ROBOT_STREAM,
        metadata={},
    )


def _stamp(stamp_ns: int):
    sec, nanosec = divmod(int(stamp_ns), 1_000_000_000)
    return SimpleNamespace(sec=sec, nanosec=nanosec)


def _header(stamp_ns: int):
    return SimpleNamespace(stamp=_stamp(stamp_ns))


def _joint_state(stamp_ns: int):
    values = np.arange(14, dtype=np.float32)
    return SimpleNamespace(
        header=_header(stamp_ns),
        position=values.tolist(),
        velocity=(values + 100).tolist(),
        effort=(values + 200).tolist(),
    )


def _read_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def _wait_until(predicate, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")
