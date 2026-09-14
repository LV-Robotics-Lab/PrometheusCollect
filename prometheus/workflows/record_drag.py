from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from prometheus.sessions.event import semantics_by_id, semantics_by_name
from prometheus.utils.workflow import (
    build_data_session,
    build_event_session,
    build_hardware_session,
    config_dict,
    prepare_run,
    section,
    write_json,
)
from prometheus.visualization.preview import (
    close_preview,
    poll_preview,
    set_preview_status,
    start_collection_preview,
)


def run_from_config(cfg: Any) -> int:
    data = config_dict(cfg, "record_drag")
    _apply_ros_config(data.get("ros", {}))
    run_cfg = section(data, "run")
    robot_cfg = section(data, "robot")
    task_cfg = section(data, "task")
    hardware_cfg = section(data, "hardware")
    data_cfg = section(data, "data")
    event_cfg = section(data, "event")
    workflow_cfg = section(data, "workflow")
    run = prepare_run(run_cfg)

    hardware = build_hardware_session(run=run, hardware_cfg=hardware_cfg, robot_cfg=robot_cfg)
    data_session = build_data_session(data_cfg=data_cfg, hardware=hardware)
    event_session = build_event_session(event_cfg)
    event_semantics = semantics_by_id(event_cfg)

    if not data_session.recording_enabled:
        raise ValueError("record_drag requires data.recording.enabled=true")

    wait_ready_timeout_s = float(run_cfg["wait_ready_timeout_s"])
    event_wait_timeout_s = float(workflow_cfg["event_wait_timeout_s"])
    max_episodes = int(workflow_cfg["max_episodes"])
    scripted_events = _scripted_events(workflow_cfg["events"], event_cfg)
    if max_episodes <= 0:
        raise ValueError("workflow.max_episodes must be positive")
    if event_wait_timeout_s <= 0:
        raise ValueError("workflow.event_wait_timeout_s must be positive")
    if not scripted_events and not event_cfg["input_source"]:
        raise ValueError("record_drag needs event.input_source or workflow.events")
    if scripted_events and not event_cfg["input_source"]:
        stop_id = semantics_by_name(event_cfg)["record_stop"]
        stops = sum(1 for event_id, _value in scripted_events if event_id == stop_id)
        if stops < max_episodes:
            raise ValueError("scripted record_drag needs one record_stop per requested episode")

    status = "running"
    stop_reason = "not_started"
    next_episode_index = _next_episode_index(data_cfg)
    started = 0
    saved = 0
    rejected = 0
    ignored = 0
    current_episode_id = ""
    current_episode_dir = None
    preview = None
    cleanup_errors: list[str] = []

    try:
        hardware.start()
        data_session.start()
        event_session.start()
        hardware.wait_ready(wait_ready_timeout_s)
        data_session.wait_ready(wait_ready_timeout_s)
        event_session.wait_ready(wait_ready_timeout_s)
        _quiet_hardware_logs(workflow_cfg)
        preview = start_collection_preview(data_session, workflow_cfg)
        _print_ready_message(run, task_cfg, workflow_cfg, hardware, data_session, event_cfg)
        _emit_scripted_events(event_session, scripted_events)

        while saved + rejected < max_episodes:
            _check_collection_health(hardware, data_session)
            _drain_robot_actions(hardware, data_session)
            poll_preview(preview)
            event = event_session.wait(timeout_s=event_wait_timeout_s)
            if event is None:
                continue
            _check_collection_health(hardware, data_session)
            poll_preview(preview)
            event_id, value = event
            name = event_semantics[event_id]

            if name == "capture_home":
                home_path = _capture_home_snapshot(hardware, workflow_cfg)
                print(f"[home] captured current YAM pose to {home_path}", flush=True)
                continue

            if name == "return_home":
                if current_episode_id:
                    ignored += 1
                    print("[home] return ignored while an episode is recording; stop/save it first.", flush=True)
                    continue
                _return_robot_home(hardware, workflow_cfg, trigger="on_manual_return")
                continue

            if name == "quit":
                print("[record] quit requested by input; saving before safe shutdown.", flush=True)
                raise KeyboardInterrupt

            if name == "record_start":
                if current_episode_id:
                    ignored += 1
                    continue
                _drain_robot_actions(hardware, data_session)
                if not _robot_ready_for_recording(hardware):
                    ignored += 1
                    print(
                        "[record] start ignored: robot is not ready for recording; "
                        "for YAM, press both teaching-handle buttons to synchronize both sides.",
                        flush=True,
                    )
                    continue
                current_episode_id = f"{next_episode_index:04d}"
                next_episode_index += 1
                started += 1
                current_episode_dir = data_session.start_episode(
                    current_episode_id,
                    metadata={"workflow": "record_drag", "run_id": run.run_id, "task": task_cfg},
                )
                set_preview_status(preview, "running", f"episode={current_episode_id}")
                data_session.record_event("record_start", value, metadata={"episode_id": current_episode_id})
                print(f"[record] started episode={current_episode_id} dir={current_episode_dir}", flush=True)
                continue

            if name == "record_stop":
                if not current_episode_id:
                    ignored += 1
                    continue
                _drain_robot_actions(hardware, data_session)
                data_session.record_event("record_stop", value, metadata={"episode_id": current_episode_id})
                accept = bool(value)
                _stop_episode_checked(data_session, accept=accept)
                set_preview_status(preview, "saved" if accept else "rejected", f"episode={current_episode_id}")
                print(
                    f"[record] {'saved' if accept else 'rejected'} episode={current_episode_id} dir={current_episode_dir}",
                    flush=True,
                )
                episode_id = current_episode_id
                saved += int(accept)
                rejected += int(not accept)
                current_episode_id = ""
                current_episode_dir = None
                if accept and _auto_return_after_saved_drag_episode(hardware, workflow_cfg):
                    pass
                elif _has_home_return_lifecycle(workflow_cfg):
                    _return_robot_home(hardware, workflow_cfg, trigger="on_episode_stop")
                else:
                    _reset_robot_after_episode(hardware, workflow_cfg, episode_id)
                _check_collection_health(hardware, data_session)
                stop_reason = "max_episodes" if saved + rejected >= max_episodes else "record_stop"
                continue

            if not current_episode_id:
                ignored += 1
                continue
            data_session.record_event(name, value, metadata={"episode_id": current_episode_id})
        else:
            stop_reason = "max_episodes"

        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        stop_reason = "keyboard_interrupt"
        print("[record] Ctrl-C received; saving any active episode before shutdown.", flush=True)
    except Exception as exc:
        status = "error"
        stop_reason = "exception"
        data_session.recorder.fail_episode(str(exc))
        raise
    finally:
        def cleanup(label: str, action: Any) -> Any:
            try:
                return action()
            except Exception as exc:
                cleanup_errors.append(f"{label}: {exc}")
                print(f"[cleanup] {label}: {exc}", flush=True)
                return None

        if current_episode_id:
            cleanup("drain_actions", lambda: _drain_robot_actions(hardware, data_session))
            cleanup("record_stop", lambda: data_session.record_event("record_stop", metadata={"reason": stop_reason}))
            accept = status in {"completed", "interrupted"} and not cleanup_errors
            if cleanup_errors:
                data_session.recorder.fail_episode("; ".join(cleanup_errors))
            final_status = cleanup("stop_episode", lambda: _stop_episode_checked(data_session, accept=accept))
            if final_status == "saved":
                saved += 1
                cleanup("preview_status", lambda: set_preview_status(preview, "saved", f"episode={current_episode_id}"))
                print(f"[record] saved episode={current_episode_id} dir={current_episode_dir}", flush=True)
                if not _has_home_return_lifecycle(workflow_cfg):
                    cleanup("reset_after_episode", lambda: _reset_robot_after_episode(hardware, workflow_cfg, current_episode_id))
            else:
                rejected += 1
        if status == "interrupted" and not cleanup_errors and _has_home_return_lifecycle(workflow_cfg):
            cleanup("home_return", lambda: _return_robot_home(hardware, workflow_cfg, trigger="on_shutdown"))
        cleanup("preview.close", lambda: close_preview(preview))
        for name, session in (("event", event_session), ("data", data_session), ("hardware", hardware)):
            cleanup(f"{name}.stop", session.stop)
            cleanup(f"{name}.close", session.close)
            snapshot = cleanup(f"{name}.status", session.status)
            details = {} if snapshot is None else snapshot.details
            for error in details.get("cleanup_errors", []):
                cleanup_errors.append(f"{name}: {error}")
        if cleanup_errors:
            status = "error"
            stop_reason = "cleanup_error" if stop_reason != "exception" else stop_reason
        write_json(
            run.manifest_path,
            {
                "run_id": run.run_id,
                "status": status,
                "stop_reason": stop_reason,
                "episodes_started": started,
                "episodes_saved": saved,
                "episodes_rejected": rejected,
                "events_ignored": ignored,
                "cleanup_errors": cleanup_errors,
                "time_ns": time.time_ns(),
                "output_dir": str(run.output_dir),
                "runtime_dir": str(run.runtime_dir),
                "task": task_cfg,
                "hardware": hardware.status().as_dict(),
                "data": data_session.status().as_dict(),
                "event": event_session.status().as_dict(),
            },
        )
    return {"completed": 0, "interrupted": 130}.get(status, 1)


def _stop_episode_checked(data_session: Any, *, accept: bool) -> str:
    result = data_session.stop_episode(accept=accept)
    if result == "failed":
        raise RuntimeError(f"episode save failed: {data_session.recorder.details().get('failure', '')}")
    return result


def _check_collection_health(hardware: Any, data_session: Any) -> None:
    hardware.check_health()
    failure = data_session.recorder.details().get("failure")
    if failure:
        raise RuntimeError(f"data recorder failed: {failure}")
    status = data_session.status()
    if status.state.value == "failed":
        raise RuntimeError(f"data session failed: {status.message}")


def _emit_scripted_events(event_session: Any, events: list[tuple[int, Any]]) -> None:
    for event_id, value in events:
        event_session.emit(event_id, value)


def _robot_ready_for_recording(hardware: Any) -> bool:
    status = getattr(hardware, "robot_driver_status", None)
    if not callable(status):
        return True
    payload = status()
    if not isinstance(payload, dict):
        raise TypeError("hardware.robot_driver_status() must return a mapping")
    return payload.get("ready_for_recording") is not False


def _drain_robot_actions(hardware: Any, data_session: Any) -> int:
    drain = getattr(hardware, "drain_robot_actions", None)
    if not callable(drain):
        return 0
    items = drain()
    if not isinstance(items, list):
        raise TypeError("hardware.drain_robot_actions() must return a list")
    if not data_session.recorder.recording:
        return 0

    recorded = 0
    for item in items:
        if not isinstance(item, dict):
            raise TypeError("robot action entries must be mappings")
        missing = {"action", "action_space", "hz", "stamp_ns"}.difference(item)
        if missing:
            raise ValueError(f"robot action entry missing keys: {sorted(missing)}")
        ok = data_session.record_action(
            item["action"],
            action_space=str(item["action_space"]),
            hz=float(item["hz"]),
            stamp_ns=int(item["stamp_ns"]),
            metadata=dict(item.get("metadata", {})),
        )
        if not ok:
            failure = str(data_session.recorder.details().get("failure", "")).strip()
            if failure:
                raise RuntimeError(f"data recorder failed: {failure}")
            raise RuntimeError("failed to enqueue executed robot action")
        recorded += 1
    return recorded


def _scripted_events(raw_events: Any, event_cfg: dict[str, Any]) -> list[tuple[int, Any]]:
    if not isinstance(raw_events, list):
        raise ValueError("workflow.events must be a list")
    name_to_id = semantics_by_name(event_cfg)
    events: list[tuple[int, Any]] = []
    for item in raw_events:
        if not isinstance(item, dict):
            raise ValueError("workflow.events items must be mappings with name and value")
        if "name" not in item or "value" not in item:
            raise ValueError("workflow.events items require name and value")
        name = str(item["name"])
        if name not in name_to_id:
            raise ValueError(f"workflow.events name {name!r} is not defined in event.semantics")
        events.append((name_to_id[name], item["value"]))
    return events


def _reset_robot_after_episode(hardware: Any, workflow_cfg: dict[str, Any], episode_id: str) -> None:
    reset_cfg = workflow_cfg.get("reset_after_episode", True)
    if isinstance(reset_cfg, dict):
        enabled = bool(reset_cfg.get("enabled", True))
        timeout_s = float(reset_cfg.get("timeout_s", 30.0))
    else:
        enabled = bool(reset_cfg)
        timeout_s = 30.0
    if not enabled:
        return

    driver_status = getattr(hardware, "robot_driver_status", None)
    if callable(driver_status):
        payload = driver_status()
        if isinstance(payload, dict) and payload.get("supports_reset_home") is False:
            print(
                f"[robot] episode={episode_id} recorded; reset skipped because this robot has no safe home routine.",
                flush=True,
            )
            return

    client = getattr(hardware, "robot_client", None)
    reset_home = getattr(client, "reset_home", None)
    if not callable(reset_home):
        print("[robot] reset_after_episode skipped: robot client has no reset_home()", flush=True)
        return

    print(f"[robot] episode={episode_id} recorded; resetting robot home.", flush=True)
    result = reset_home(timeout=timeout_s, wait=True)
    if isinstance(result, dict) and not bool(result.get("accepted", False)):
        raise RuntimeError(f"robot reset_home rejected: {result}")


def _has_home_return_lifecycle(workflow_cfg: dict[str, Any]) -> bool:
    return isinstance(workflow_cfg.get("home_return"), dict)


def _auto_return_after_saved_drag_episode(hardware: Any, workflow_cfg: dict[str, Any]) -> bool:
    """Run YAM-drag's explicit startup-pose return without affecting YAM teleop."""
    driver_status = getattr(hardware, "robot_driver_status", None)
    payload = driver_status() if callable(driver_status) else {}
    if not isinstance(payload, dict) or not bool(payload.get("auto_return_after_episode", False)):
        return False
    try:
        _return_robot_home(hardware, workflow_cfg, trigger="on_drag_episode_stop", force=True)
    except Exception as exc:
        # A refused displacement or a transient control failure must not end
        # the collection process and release the followers' holding torque.
        print(f"[robot] automatic return-to-start failed; staying in drag mode: {exc}", flush=True)
    return True


def _return_robot_home(
    hardware: Any, workflow_cfg: dict[str, Any], *, trigger: str, force: bool = False
) -> None:
    home_cfg = workflow_cfg.get("home_return")
    if not isinstance(home_cfg, dict):
        raise TypeError("workflow.home_return must be a mapping")
    if not force and not bool(home_cfg.get(trigger, False)):
        return
    timeout_s = float(home_cfg.get("timeout_s", 30.0))
    if timeout_s <= 0:
        raise ValueError("workflow.home_return.timeout_s must be positive")

    driver_status = getattr(hardware, "robot_driver_status", None)
    payload = driver_status() if callable(driver_status) else {}
    if isinstance(payload, dict) and payload.get("supports_reset_home") is False:
        print("[robot] Home return skipped: no validated YAM Home routine is available.", flush=True)
        return

    client = getattr(hardware, "robot_client", None)
    reset_home = getattr(client, "reset_home", None)
    if not callable(reset_home):
        print("[robot] Home return skipped: robot client has no reset_home().", flush=True)
        return
    label = str(payload.get("home_return_label", "the captured Home pose"))
    print(f"[robot] returning follower arms to {label}.", flush=True)
    result = reset_home(timeout=timeout_s, wait=True)
    if isinstance(result, dict) and not bool(result.get("accepted", False)):
        # A validated refusal is an operator event, not a reason to release
        # torque by exiting the collection process. Actual driver faults are
        # checked again on the next collection tick.
        print(f"[robot] Home return rejected; collection remains active: {result}", flush=True)
        return
    if trigger in {"on_episode_stop", "on_manual_return", "on_drag_episode_stop"}:
        if isinstance(payload, dict) and payload.get("drag_mode"):
            print("[robot] return-to-start complete; followers are back in gravity-compensation drag mode.", flush=True)
        else:
            print("[robot] Home return complete; press both teaching-handle buttons to re-sync before recording again.", flush=True)


def _next_episode_index(data_cfg: dict[str, Any]) -> int:
    recording_cfg = data_cfg.get("recording", {})
    if not isinstance(recording_cfg, dict):
        return 0
    output_dir = Path(str(recording_cfg.get("output_dir", ""))).expanduser()
    if not output_dir.exists():
        return 0
    indices = [
        int(path.name)
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    return max(indices, default=-1) + 1


def _capture_home_snapshot(hardware: Any, workflow_cfg: dict[str, Any]) -> Path:
    """Persist the current YAM state without issuing a robot command."""
    capture_cfg = workflow_cfg.get("home_capture", {})
    if not isinstance(capture_cfg, dict) or not bool(capture_cfg.get("enabled", False)):
        raise RuntimeError("home capture is disabled; set workflow.home_capture.enabled=true")
    path_value = str(capture_cfg.get("path", "")).strip()
    if not path_value:
        raise ValueError("workflow.home_capture.path must be set")

    driver = hardware.robot_driver
    capture_home = getattr(driver, "capture_home", None)
    if callable(capture_home):
        capture_home()
    read_state = getattr(driver, "read_state", None)
    if not callable(read_state):
        raise RuntimeError(f"{type(driver).__name__} cannot provide a home snapshot")
    state = read_state()
    follower = np.asarray(getattr(state, "qpos", None), dtype=np.float64).reshape(-1)
    leader_value = getattr(state, "leader_qpos", None)
    leader = None if leader_value is None else np.asarray(leader_value, dtype=np.float64).reshape(-1)
    if follower.shape != (14,) or (leader is not None and leader.shape != (14,)):
        raise ValueError(
            "home capture requires 14 follower positions and, when available, 14 leader positions; "
            f"got leader={None if leader is None else leader.shape}, follower={follower.shape}"
        )
    if not np.all(np.isfinite(follower)) or (leader is not None and not np.all(np.isfinite(leader))):
        raise ValueError("home capture refused: robot state contains NaN or infinity")

    status = hardware.robot_driver_status()
    payload = {
        "schema_version": 1,
        "robot_id": str(status.get("robot_id", "yam")),
        "captured_at_unix_ns": time.time_ns(),
        "units": {"arm_joints": "rad", "gripper": "m"},
        "joint_order_per_side": [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"
        ],
        "follower": {"left": follower[:7].tolist(), "right": follower[7:].tolist()},
        "can_interfaces": {
            "follower": status.get("follower_can", {}),
        },
    }
    if leader is not None:
        payload["leader"] = {"left": leader[:7].tolist(), "right": leader[7:].tolist()}
        payload["can_interfaces"]["leader"] = status.get("leader_can", {})
    path = Path(path_value).expanduser()
    write_json(path, payload)
    return path


def _quiet_hardware_logs(workflow_cfg: dict[str, Any]) -> None:
    """Hide routine i2rt diagnostics after all hardware has reported ready.

    Recording status uses direct, prefixed terminal messages, so lowering the
    logging threshold here keeps start/stop messages readable while retaining
    hardware warnings and errors.
    """
    if bool(workflow_cfg.get("quiet_hardware_logs", False)):
        logging.getLogger().setLevel(logging.WARNING)


def _print_ready_message(
    run: Any,
    task_cfg: dict[str, Any],
    workflow_cfg: dict[str, Any],
    hardware: Any,
    data_session: Any,
    event_cfg: dict[str, Any],
) -> None:
    preview_cfg = workflow_cfg.get("preview", {})
    preview_mode = preview_cfg.get("mode", "local") if isinstance(preview_cfg, dict) else "disabled"
    print("", flush=True)
    print("[ready] Prometheus data collection is ready.", flush=True)
    print(f"[ready] task={task_cfg.get('id')} run_id={run.run_id}", flush=True)
    print(f"[ready] recording_root={data_session.recorder.config.get('output_dir', '')}", flush=True)
    print(f"[ready] preview={preview_mode}", flush=True)
    print(f"[ready] ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '')}", flush=True)
    home_return = workflow_cfg.get("home_return", {})
    driver_status = getattr(hardware, "robot_driver_status", None)
    driver_payload = driver_status() if callable(driver_status) else {}
    supports_home_return = not (
        isinstance(driver_payload, dict)
        and driver_payload.get("supports_reset_home") is False
    )
    if isinstance(home_return, dict) and bool(home_return.get("on_shutdown", False)) and supports_home_return:
        print("[ready] Ctrl-C saves first, then returns follower arms to Home before torque release.", flush=True)
    if isinstance(home_return, dict) and bool(home_return.get("on_episode_stop", False)) and supports_home_return:
        print("[ready] b saves first, then returns follower arms to Home; re-sync both teaching handles before the next episode.", flush=True)
    if isinstance(home_return, dict) and bool(home_return.get("on_manual_return", False)) and supports_home_return:
        print("[ready] r returns idle follower arms to the captured Home pose; re-sync both teaching handles afterward.", flush=True)
    if isinstance(driver_payload, dict) and bool(driver_payload.get("auto_return_after_episode", False)):
        print("[ready] each saved YAM drag episode returns followers to this run's startup pose, then restores gravity-compensation.", flush=True)
    if not supports_home_return:
        print("[ready] follower drag mode does not auto-return Home; move both arms to a safe pose manually.", flush=True)
    if bool(workflow_cfg.get("quiet_hardware_logs", False)):
        print("[ready] routine hardware diagnostics are hidden; warnings and errors remain visible.", flush=True)
    input_sources = {
        name.strip()
        for name in str(event_cfg.get("input_source", "")).split(",")
        if name.strip()
    }
    if "tty" in input_sources:
        tty_cfg = event_cfg.get("tty", {})
        confirm_presses = int(tty_cfg.get("confirm_presses", 2)) if isinstance(tty_cfg, dict) else 2
        repeat = max(1, confirm_presses)
        print(
            "[ready] controls: "
            f"{'a' * repeat}=start recording, "
            f"{'b' * repeat}=stop/save, "
            f"{'c' * repeat}=event mark, "
            f"{'h' * repeat}=capture YAM Home pose, "
            f"{'r' * repeat}=return idle followers to captured Home, "
            f"{'q' * repeat}=save and safely quit, "
            "Ctrl-C=save active episode and exit",
            flush=True,
        )
        print("[ready] keyboard input is read from this terminal; keep the terminal focused.", flush=True)
    else:
        print(
            f"[ready] controls: input_source={','.join(sorted(input_sources))}, "
            "Ctrl-C=save active episode and exit",
            flush=True,
        )
    print("", flush=True)


def _apply_ros_config(raw_cfg: Any) -> None:
    if raw_cfg is None:
        return
    if not isinstance(raw_cfg, dict):
        raise ValueError("ros must be a mapping")
    if "localhost_only" in raw_cfg:
        enabled = bool(raw_cfg["localhost_only"])
        os.environ["ROS_LOCALHOST_ONLY"] = "1" if enabled else "0"
        if enabled:
            os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../configs", config_name="collect")
    def _main(cfg: DictConfig) -> None:
        raise SystemExit(run_from_config(cfg))

    _main()


if __name__ == "__main__":
    main()
