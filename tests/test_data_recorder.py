from __future__ import annotations

import pickle
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from prometheus.data.types import DataSample, DataStream
from prometheus.sessions.data import DataSession

_ROBOT_STREAM = DataStream(
    name="robot_state",
    topic="/robot/state",
    msg_type="sensor_msgs.msg.JointState",
)


def test_recorder_pause_blocks_append_and_disk_writes(tmp_path):
    data = _recording_data_session(tmp_path)
    data.start()
    data.start_episode("0001")

    assert data.recorder.record_sample(_sample(100_000_000))
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)
    data.recorder.pause()
    assert data.recorder.recording
    assert data.recorder.paused
    assert not data.recorder.record_sample(_sample(200_000_000))
    assert not data.recorder.record_event("event_label", value=1)

    data.stop_episode()
    data.close()

    robot = _read_pickle(tmp_path / "0001" / "robot" / "robot_state_dict.pkl")
    assert robot["left"]["timestamps"] == [100]


def test_recorder_resume_restores_disk_writes(tmp_path):
    data = _recording_data_session(tmp_path)
    data.start()
    data.start_episode("0001")

    assert data.recorder.record_sample(_sample(100_000_000))
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)
    data.recorder.pause()
    assert not data.recorder.record_sample(_sample(200_000_000))
    data.recorder.resume()
    assert not data.recorder.paused
    assert data.recorder.record_sample(_sample(300_000_000))
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)

    data.stop_episode()
    data.close()

    robot = _read_pickle(tmp_path / "0001" / "robot" / "robot_state_dict.pkl")
    assert robot["left"]["timestamps"] == [100, 300]


def test_recorder_pause_without_active_episode_raises(tmp_path):
    data = _recording_data_session(tmp_path)
    data.start()

    with pytest.raises(RuntimeError, match="no active episode"):
        data.recorder.pause()
    with pytest.raises(RuntimeError, match="no active episode"):
        data.recorder.resume()

    data.close()


def test_recorder_start_episode_resets_paused_state(tmp_path):
    data = _recording_data_session(tmp_path)
    data.start()
    data.start_episode("0001")
    data.recorder.pause()
    data.stop_episode()

    data.start_episode("0002")
    assert not data.recorder.paused
    assert data.recorder.record_sample(_sample(100_000_000))
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)

    data.stop_episode()
    data.close()

    robot = _read_pickle(tmp_path / "0002" / "robot" / "robot_state_dict.pkl")
    assert robot["left"]["timestamps"] == [100]


def test_data_session_buffer_grows_while_recorder_paused(tmp_path):
    data = _recording_data_session(tmp_path)
    data.start()
    data.start_episode("0001")

    data.ingest("robot_state", _joint_state(100_000_000))
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)
    counts_before_pause = data.counts()["robot_state"]
    data.recorder.pause()
    data.ingest("robot_state", _joint_state(200_000_000))
    data.ingest("robot_state", _joint_state(300_000_000))
    counts_after_pause = data.counts()["robot_state"]

    data.stop_episode()
    data.close()

    assert counts_after_pause > counts_before_pause
    robot = _read_pickle(tmp_path / "0001" / "robot" / "robot_state_dict.pkl")
    assert robot["left"]["timestamps"] == [100]


def test_recorder_write_item_skips_queued_samples_after_pause(tmp_path, monkeypatch):
    first_sample_started = threading.Event()
    allow_first_sample_finish = threading.Event()

    import prometheus.data.raw_episode as raw_episode

    original_write_sample = raw_episode.RawEpisodeWriter.write_sample

    def gated_write_sample(self, sample):
        if int(sample.stamp_ns) == 100_000_000:
            first_sample_started.set()
            assert allow_first_sample_finish.wait(timeout=1.0)
        return original_write_sample(self, sample)

    monkeypatch.setattr(raw_episode.RawEpisodeWriter, "write_sample", gated_write_sample)

    data = _recording_data_session(tmp_path)
    data.start()
    data.start_episode("0001")

    assert data.recorder.record_sample(_sample(100_000_000))
    assert first_sample_started.wait(timeout=1.0)
    assert data.recorder.record_sample(_sample(200_000_000))
    data.recorder.pause()
    allow_first_sample_finish.set()
    _wait_until(lambda: data.recorder.details()["queues"].get("robot", 0) == 0, timeout_s=1.0)
    assert not data.recorder.record_sample(_sample(300_000_000))

    data.stop_episode()
    data.close()

    robot = _read_pickle(tmp_path / "0001" / "robot" / "robot_state_dict.pkl")
    assert robot["left"]["timestamps"] == [100]


def _recording_data_session(tmp_path):
    return DataSession(
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
        },
    )


def _sample(stamp_ns: int) -> DataSample:
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


def _read_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def _wait_until(predicate, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for condition")
