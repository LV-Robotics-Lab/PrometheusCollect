import json
import pickle

from prometheus.data.episode import RawEpisodeReader
from prometheus.sessions.data import DataSession


def test_raw_episode_reader_exposes_timeline(tmp_path):
    episode_dir = tmp_path / "episode_0001"
    (episode_dir / "camera").mkdir(parents=True)
    (episode_dir / "robot").mkdir()
    (episode_dir / "event").mkdir()
    (episode_dir / "action").mkdir()
    (episode_dir / "manifests").mkdir()
    _write_json(
        episode_dir / "camera" / "metadata.json",
        {
            "d405_rgb": {
                "topic": "/camera/d405/color/image_raw",
                "video": "camera/d405_rgb.mp4",
                "frames": 2,
                "timestamps": "camera/timestamp.pkl::d405",
                "container": "mp4",
                "codec": "hevc",
            }
        },
    )
    _write_pickle(episode_dir / "camera" / "timestamp.pkl", {"d405": [100, 133]})
    _write_pickle(episode_dir / "event" / "event_dict.pkl", {"timestamps": [120], "names": ["record_start"], "data": [True]})
    _write_json(episode_dir / "manifests" / "camera.json", {"prometheus_camera_d405_recorder": {"status": "saved"}})

    reader = RawEpisodeReader(episode_dir)
    via_session = DataSession.open_episode(episode_dir)

    assert reader.counts()["d405_rgb"] == 2
    assert reader.timeline()["timestamps_ms"] == [100, 120, 133]
    assert reader.visual_streams(t_ms=121)[0]["index"] == 1
    assert reader.visual_streams(t_ms=121)[0]["stamp_ms"] == 133
    assert via_session.status()["details"]["episode_dir"] == str(episode_dir)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_pickle(path, value):
    with path.open("wb") as file:
        pickle.dump(value, file)
