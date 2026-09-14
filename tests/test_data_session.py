from __future__ import annotations

import json
import pickle
import time
from types import SimpleNamespace

import numpy as np

import prometheus.data.raw_episode as raw_episode
from tests.noop import RecordedChunk
from prometheus.sessions.data import DataSession


def test_camera_video_writer_uses_software_hevc_by_default(tmp_path, monkeypatch):
    created = {}

    class CaptureWriter:
        def __init__(self, path, **kwargs):
            created["path"] = path
            created.update(kwargs)

    monkeypatch.setattr(raw_episode, "FFmpegPipeWriter", CaptureWriter)
    _writer, metadata = raw_episode.open_camera_video_writer(
        "base_0_rgb",
        tmp_path / "base_0_rgb.mp4",
        np.zeros((4, 6, 3), dtype=np.uint8),
        "rgb8",
        fps=30.0,
    )

    assert created["output_args"][:2] == ["-c:v", "libx265"]
    assert "pools=4:log-level=error" in created["output_args"]
    assert metadata["encoder"] == "libx265"
    assert metadata["quality"] == {"mode": "crf", "value": 14}


def test_camera_video_writer_uses_supported_nvenc_tune(tmp_path, monkeypatch):
    created = {}

    class CaptureWriter:
        def __init__(self, path, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(raw_episode, "FFmpegPipeWriter", CaptureWriter)
    _writer, metadata = raw_episode.open_camera_video_writer(
        "base_0_rgb",
        tmp_path / "base_0_rgb.mp4",
        np.zeros((4, 6, 3), dtype=np.uint8),
        "bgr8",
        fps=30.0,
        encoder="hevc_nvenc",
    )

    tune_index = created["output_args"].index("-tune")
    assert created["output_args"][tune_index + 1] == "hq"
    assert "uhq" not in created["output_args"]
    assert metadata["encoder"] == "hevc_nvenc"


def test_data_session_writes_raw_episode_layout(tmp_path):
    data = _recording_data_session(
        tmp_path,
        {
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            },
            "left_eef": {
                "topic": "/robot/left_eef",
                "msg_type": "geometry_msgs.msg.PoseStamped",
            },
            "right_eef": {
                "topic": "/robot/right_eef",
                "msg_type": "geometry_msgs.msg.PoseStamped",
            },
        },
    )
    data.start()
    episode_dir = data.start_episode("0001", metadata={"task": "unit_test"})

    data.ingest("robot_state", _joint_state(100_000_000))
    data.ingest("left_eef", _pose(100_000_000, x=1.0))
    data.ingest("right_eef", _pose(100_000_000, x=2.0))
    data.record_action_chunk(
        RecordedChunk(
            np.zeros((2, 3), dtype=np.float32),
            action_space="abs_qpos",
            hz=20.0,
        ),
        metadata={"chunk_index": 0},
    )
    data.record_event("event_mark", metadata={"source": "test"})
    data.stop_episode()
    data.close()

    robot = _read_pickle(episode_dir / "robot" / "robot_state_dict.pkl")
    actions = _read_pickle(episode_dir / "action" / "action_chunk_dict.pkl")
    events = _read_pickle(episode_dir / "event" / "event_dict.pkl")
    robot_manifest = _read_json(episode_dir / "manifests" / "robot.json")
    action_manifest = _read_json(episode_dir / "manifests" / "action.json")
    event_manifest = _read_json(episode_dir / "manifests" / "event.json")

    np.testing.assert_array_equal(robot["left"]["joint"][0], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(robot["right"]["joint"][0], np.arange(7, 14, dtype=np.float32))
    np.testing.assert_allclose(robot["left"]["eef"][0], np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_allclose(robot["right"]["eef"][0], np.array([2, 0, 0, 1, 0, 0, 0], dtype=np.float32))
    assert robot["left"]["timestamps"] == [100]
    assert actions["chunks"][0]["action_space"] == "abs_qpos"
    np.testing.assert_array_equal(actions["chunks"][0]["actions"], np.zeros((2, 3), dtype=np.float32))
    assert events["names"] == ["event_mark"]
    assert events["data"] == [True]
    assert robot_manifest["prometheus_robot_recorder"]["status"] == "saved"
    assert action_manifest["prometheus_action_recorder"]["policy_action_chunks"] == 1
    assert action_manifest["prometheus_action_recorder"]["executed_actions"] == 0
    assert event_manifest["prometheus_event_recorder"]["event_samples"] == 1


def test_camera_recording_commits_complete_physical_bundle(tmp_path, monkeypatch):
    created_writers = {}

    def fake_open(name, path, image, encoding, *, fps, encoder):
        writer = FakeVideoWriter()
        created_writers[name] = writer
        return writer, {
            "video": f"camera/{path.name}",
            "container": "fake",
            "codec": "fake",
            "pixel_format": encoding,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "lossless": True,
        }

    monkeypatch.setattr(raw_episode, "open_camera_video_writer", fake_open)
    data = _recording_data_session(
        tmp_path,
        {
            "d405_color": {
                "topic": "/camera/d405/color/image_raw",
                "msg_type": "sensor_msgs.msg.Image",
            },
            "d405_infra1": {
                "topic": "/camera/d405/infra1/image_rect_raw",
                "msg_type": "sensor_msgs.msg.Image",
            },
            "d405_infra2": {
                "topic": "/camera/d405/infra2/image_rect_raw",
                "msg_type": "sensor_msgs.msg.Image",
            },
        },
    )
    data.start()
    episode_dir = data.start_episode("0002")

    data.ingest("d405_color", _image(200_000_000, encoding="rgb8", shape=(2, 3, 3)))
    data.ingest("d405_infra1", _image(200_000_000, encoding="mono8", shape=(2, 3)))
    data.ingest("d405_infra2", _image(200_000_000, encoding="mono8", shape=(2, 3)))
    data.stop_episode()
    data.close()

    timestamps = _read_pickle(episode_dir / "camera" / "timestamp.pkl")
    metadata = _read_json(episode_dir / "camera" / "metadata.json")
    manifest = _read_json(episode_dir / "manifests" / "camera.json")

    assert timestamps == {"d405": [200]}
    assert set(metadata) == {"d405_rgb", "d405_ir_left", "d405_ir_right"}
    assert metadata["d405_rgb"]["topic"] == "/camera/d405/color/image_raw"
    assert metadata["d405_rgb"]["timestamps"] == "camera/timestamp.pkl::d405"
    assert all(writer.frames == 1 and writer.released for writer in created_writers.values())
    assert manifest["prometheus_camera_d405_recorder"]["complete_sensor_bundles"] == 1


def test_camera_recording_keeps_timestamps_when_video_finalize_fails(tmp_path, monkeypatch):
    def fake_open(name, path, image, encoding, *, fps, encoder):
        return FakeVideoWriter(fail_on_release=name == "d405_rgb"), {
            "video": f"camera/{path.name}",
            "container": "fake",
            "codec": "fake",
            "pixel_format": encoding,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "lossless": True,
        }

    monkeypatch.setattr(raw_episode, "open_camera_video_writer", fake_open)
    data = _recording_data_session(
        tmp_path,
        {
            "d405_color": {
                "topic": "/camera/d405/color/image_raw",
                "msg_type": "sensor_msgs.msg.Image",
            },
        },
    )
    data.start()
    episode_dir = data.start_episode("0002")

    data.ingest("d405_color", _image(200_000_000, encoding="rgb8", shape=(2, 3, 3)))
    data.stop_episode()
    data.close()

    timestamps = _read_pickle(episode_dir / "camera" / "timestamp.pkl")
    metadata = _read_json(episode_dir / "camera" / "metadata.json")
    manifest = _read_json(episode_dir / "manifests" / "camera.json")

    assert timestamps == {"d405": [200]}
    assert metadata["d405_rgb"]["timestamps"] == "camera/timestamp.pkl::d405"
    assert manifest["prometheus_camera_d405_recorder"]["status"] == "failed"
    assert "release failed" in manifest["prometheus_camera_d405_recorder"]["failure"]


def test_data_session_recv_stamp_aligns_offset_streams():
    data = DataSession(
        {
            "camera": {
                "topic": "/camera/image",
                "msg_type": "sensor_msgs.msg.Image",
            },
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            },
        },
        history=16,
        bridge={"enabled": False, "stamp_source": "recv"},
    )
    data.start()
    recv_base = 1_000_000_000
    for idx in range(8):
        recv_ns = recv_base + idx * 33_000_000
        data.ingest("camera", _image(recv_ns + 400_000_000, encoding="rgb8", shape=(2, 2, 3)), recv_ns=recv_ns)
        data.ingest("robot_state", _joint_state(recv_ns + 122_000_000), recv_ns=recv_ns + 1_000_000)

    frames = data.window(
        anchor="camera",
        names=("camera", "robot_state"),
        count=8,
        stride=1,
        slop_ms=50.0,
    )
    assert len(frames) == 8
    for frame in frames:
        assert abs(frame.skew_ms["robot_state"]) <= 50.0


def test_data_session_async_numpy_image_buffer_decodes_rgb():
    data = DataSession(
        {
            "camera": {
                "topic": "/camera/image",
                "msg_type": "sensor_msgs.msg.Image",
            },
        },
        history=4,
        bridge={"enabled": False},
        numpy={"enabled": True, "history": 4},
    )
    data.start()
    msg = _image(100_000_000, encoding="bgr8", shape=(2, 2, 3))
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(2, 2, 3)

    data.ingest("camera", msg)
    decoded = _wait_numpy_latest(data, "camera")
    data.close()

    assert decoded.encoding == "rgb8"
    np.testing.assert_array_equal(decoded.data, raw[:, :, ::-1])


def test_data_session_numpy_buffer_decodes_robot_and_records_tmp_npz(tmp_path):
    data = DataSession(
        {
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            },
            "left_eef": {
                "topic": "/robot/left_eef",
                "msg_type": "geometry_msgs.msg.PoseStamped",
            },
        },
        history=4,
        bridge={"enabled": False},
        numpy={
            "enabled": True,
            "history": 4,
            "recording": {"enabled": True, "output_dir": tmp_path, "run_id": "numpy_test"},
        },
    )
    data.start()
    data.ingest("robot_state", _joint_state(100_000_000))
    data.ingest("left_eef", _pose(100_000_000, x=1.0))

    robot = _wait_numpy_latest(data, "robot_state")
    eef = _wait_numpy_latest(data, "left_eef")
    data.close()

    np.testing.assert_array_equal(robot.data["position"], np.arange(14, dtype=np.float32))
    np.testing.assert_allclose(eef.data, np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32))
    root = tmp_path / "numpy_test"
    assert (root / "robot_state" / "000000.npz").is_file()
    assert (root / "left_eef" / "000000.npz").is_file()
    assert (root / "timestamps.jsonl").is_file()


def test_data_session_numpy_recording_can_write_raw_episode_layout(tmp_path, monkeypatch):
    created_writers = {}

    def fake_open(name, path, image, encoding, *, fps, encoder):
        writer = FakeVideoWriter()
        created_writers[name] = writer
        return writer, {
            "video": f"camera/{path.name}",
            "container": "fake",
            "codec": "fake",
            "pixel_format": encoding,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "lossless": True,
        }

    monkeypatch.setattr(raw_episode, "open_camera_video_writer", fake_open)
    data = DataSession(
        {
            "base_0_color": {
                "topic": "/camera/base_0/color/image_raw",
                "msg_type": "sensor_msgs.msg.Image",
            },
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            },
            "left_eef": {
                "topic": "/robot/left_eef",
                "msg_type": "geometry_msgs.msg.PoseStamped",
            },
            "right_eef": {
                "topic": "/robot/right_eef",
                "msg_type": "geometry_msgs.msg.PoseStamped",
            },
        },
        history=4,
        bridge={"enabled": False},
        numpy={
            "enabled": True,
            "history": 4,
            "recording": {
                "enabled": True,
                "format": "raw_episode",
                "output_dir": tmp_path,
                "run_id": "numpy_raw_episode",
                "fps": 30.0,
                "default_event_value": 0,
                "enable_plots": False,
            },
        },
    )
    data.start()
    data.ingest("base_0_color", _image(100_000_000, encoding="bgr8", shape=(2, 3, 3)))
    data.ingest("robot_state", _joint_state(100_000_000))
    data.ingest("left_eef", _pose(100_000_000, x=1.0))
    data.ingest("right_eef", _pose(100_000_000, x=2.0))
    _wait_numpy_latest(data, "base_0_color")
    _wait_numpy_latest(data, "right_eef")
    action_chunk = RecordedChunk(
        np.ones((2, 14), dtype=np.float32),
        action_space="abs_eef",
        hz=20.0,
        metadata={"timestamp": {"frame": np.asarray([90_000_000, 100_000_000], dtype=np.int64)}},
    )
    assert data.record_policy_action_chunk(action_chunk, metadata={"chunk_index": 0})
    data.close()

    root = tmp_path / "numpy_raw_episode"
    timestamps = _read_pickle(root / "camera" / "timestamp.pkl")
    metadata = _read_json(root / "camera" / "metadata.json")
    manifest = _read_json(root / "manifest.json")
    robot = _read_pickle(root / "robot" / "robot_state_dict.pkl")
    events = _read_pickle(root / "event" / "event_dict.pkl")
    action_by_input = _read_pickle(root / "action" / "action_chunk_by_input_timestamp.pkl")
    action_metadata = _read_pickle(root / "action" / "action_chunk_metadata.pkl")

    assert timestamps == {"base_0": [100]}
    assert metadata["base_0_rgb"]["topic"] == "/camera/base_0/color/image_raw"
    assert metadata["base_0_rgb"]["timestamps"] == "camera/timestamp.pkl::base_0"
    assert manifest["status"] == "complete"
    assert manifest["format"] == "prometheus_raw_episode_v1"
    assert robot["left"]["timestamps"] == [100]
    assert events["timestamps"] == [100]
    assert events["data"] == [0]
    np.testing.assert_array_equal(action_by_input[100_000_000], np.ones((2, 14), dtype=np.float32))
    assert action_metadata[100_000_000]["metadata"]["chunk_index"] == 0
    assert created_writers["base_0_rgb"].frames == 1
    assert created_writers["base_0_rgb"].released


def test_window_numpy_can_wait_for_latest_raw_anchor():
    data = DataSession(
        {
            "camera": {
                "topic": "/camera/image",
                "msg_type": "sensor_msgs.msg.Image",
            },
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            },
        },
        history=8,
        bridge={"enabled": False, "stamp_source": "header"},
        numpy={"enabled": True, "history": 8},
    )
    data.start()
    try:
        for idx in range(8):
            stamp_ns = (idx + 1) * 1_000_000
            data.ingest("camera", _image(stamp_ns, encoding="rgb8", shape=(2, 2, 3)))
            data.ingest("robot_state", _joint_state(stamp_ns))
        _wait_numpy_latest(data, "camera")
        stamp_ns = 9_000_000
        data.ingest("camera", _image(stamp_ns, encoding="rgb8", shape=(2, 2, 3)))
        data.ingest("robot_state", _joint_state(stamp_ns))

        frames = data.window_numpy(
            anchor="camera",
            names=("camera", "robot_state"),
            count=8,
            slop_ms=1.0,
            wait_latest=True,
            timeout_ms=100.0,
        )
    finally:
        data.close()

    assert frames[-1].stamp_ns == stamp_ns


def test_data_session_records_samples_without_sample_sink(tmp_path):
    data = _recording_data_session(
        tmp_path,
        {
            "robot_state": {
                "topic": "/robot/state",
                "msg_type": "sensor_msgs.msg.JointState",
            }
        },
    )
    data.start()
    episode_dir = data.start_episode("0003")

    data.ingest("robot_state", _joint_state(300_000_000))
    data.stop_episode()
    data.close()

    robot = _read_pickle(episode_dir / "robot" / "robot_state_dict.pkl")
    assert robot["left"]["timestamps"] == [300]


class FakeVideoWriter:
    def __init__(self, *, fail_on_release: bool = False):
        self.frames = 0
        self.released = False
        self.fail_on_release = bool(fail_on_release)

    def write(self, image):
        assert image.dtype == np.uint8
        self.frames += 1

    def release(self):
        self.released = True
        if self.fail_on_release:
            raise RuntimeError("release failed")


def _recording_data_session(tmp_path, streams):
    return DataSession(
        streams,
        history=4,
        bridge={"enabled": False},
        recording={"enabled": True, "output_dir": tmp_path, "queue_size": 16, "fps": 30.0},
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


def _pose(stamp_ns: int, *, x: float):
    return SimpleNamespace(
        header=_header(stamp_ns),
        pose=SimpleNamespace(
            position=SimpleNamespace(x=x, y=0.0, z=0.0),
            orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
        ),
    )


def _image(stamp_ns: int, *, encoding: str, shape: tuple[int, ...]):
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    height, width = shape[:2]
    channels = 1 if len(shape) == 2 else shape[2]
    return SimpleNamespace(
        header=_header(stamp_ns),
        height=height,
        width=width,
        encoding=encoding,
        step=width * channels,
        data=data.tobytes(),
    )


def _wait_numpy_latest(data: DataSession, name: str):
    deadline = time.monotonic() + 1.0
    last_error = None
    while time.monotonic() < deadline:
        try:
            return data.latest_numpy([name])[name]
        except Exception as exc:
            last_error = exc
            time.sleep(0.01)
    raise AssertionError(f"timed out waiting for numpy sample: {last_error}")


def _read_pickle(path):
    with path.open("rb") as file:
        return pickle.load(file)


def _read_json(path):
    return json.loads(path.read_text())
