#!/usr/bin/env python3
"""Diagnose ROS stream rate and timestamp alignment for FM rollout.

Simulates DataSession window alignment:
  - stamp_source=header: match on message header.stamp
  - stamp_source=recv: match on local receive time (rollout_fm_sync default)
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import deque
from dataclasses import dataclass, field


def _stamp_ns(msg) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


@dataclass
class StreamStats:
    name: str
    header_stamps_ns: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    recv_stamps_ns: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    inter_recv_ms: list[float] = field(default_factory=list)

    def record(self, msg, recv_ns: int) -> None:
        header_ns = _stamp_ns(msg)
        if self.recv_stamps_ns:
            delta_ms = (recv_ns - self.recv_stamps_ns[-1]) / 1_000_000.0
            if delta_ms > 0:
                self.inter_recv_ms.append(delta_ms)
        self.header_stamps_ns.append(header_ns)
        self.recv_stamps_ns.append(recv_ns)


def _effective_hz(intervals_ms: list[float]) -> float | None:
    if not intervals_ms:
        return None
    avg_ms = statistics.mean(intervals_ms)
    if avg_ms <= 0:
        return None
    return 1000.0 / avg_ms


def _nearest_skew_ms(target_ns: int, candidates_ns: deque[int]) -> float | None:
    if not candidates_ns:
        return None
    nearest = min(candidates_ns, key=lambda item: abs(item - target_ns))
    return (nearest - target_ns) / 1_000_000.0


def _simulate_window_skew(
    *,
    anchor: StreamStats,
    others: dict[str, StreamStats],
    count: int,
    stamp_key: str,
    slop_ms: float,
) -> tuple[float, str, int]:
    anchor_series = anchor.recv_stamps_ns if stamp_key == "recv" else anchor.header_stamps_ns
    if len(anchor_series) < count:
        return float("nan"), "", 0

    selected = list(anchor_series)[-count:]
    worst_skew = 0.0
    worst_pair = ""
    failures = 0
    for anchor_ns in selected:
        for name, stats in others.items():
            series = stats.recv_stamps_ns if stamp_key == "recv" else stats.header_stamps_ns
            skew_ms = _nearest_skew_ms(anchor_ns, series)
            if skew_ms is None:
                failures += 1
                continue
            abs_skew = abs(skew_ms)
            if abs_skew > worst_skew:
                worst_skew = abs_skew
                worst_pair = f"{name}@{skew_ms:+.1f}ms"
            if abs_skew > slop_ms:
                failures += 1
    return worst_skew, worst_pair, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", default="x5_1")
    parser.add_argument("--control-domain", type=int, default=21)
    parser.add_argument("--sensor-domain", type=int, default=22)
    parser.add_argument("--anchor", default="base_0_color")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--slop-ms", type=float, default=100.0)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--expected-hz", type=float, default=30.0)
    args = parser.parse_args()

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState

    camera_names = ("base_0", "left_wrist_0", "right_wrist_0")
    stream_topics = {
        "robot_state": f"/prometheus/robots/{args.robot_id}/state",
        "left_eef": f"/prometheus/robots/{args.robot_id}/left_eef_pose",
        "right_eef": f"/prometheus/robots/{args.robot_id}/right_eef_pose",
    }
    for camera in camera_names:
        stream_topics[f"{camera}_color"] = (
            f"/prometheus/sensors/cameras/{camera}/color/image_raw"
        )

    robot_streams = {"robot_state", "left_eef", "right_eef"}
    sensor_streams = set(stream_topics) - robot_streams

    ctx_robot = rclpy.Context()
    ctx_sensor = rclpy.Context()
    rclpy.init(context=ctx_robot, domain_id=args.control_domain)
    rclpy.init(context=ctx_sensor, domain_id=args.sensor_domain)

    node_robot = Node("sync_check_robot", context=ctx_robot)
    node_sensor = Node("sync_check_sensor", context=ctx_sensor)
    exec_robot = SingleThreadedExecutor(context=ctx_robot)
    exec_sensor = SingleThreadedExecutor(context=ctx_sensor)
    exec_robot.add_node(node_robot)
    exec_sensor.add_node(node_sensor)

    stats: dict[str, StreamStats] = {name: StreamStats(name) for name in stream_topics}

    def make_cb(name: str):
        def _cb(msg) -> None:
            stats[name].record(msg, time.time_ns())

        return _cb

    node_robot.create_subscription(
        JointState, stream_topics["robot_state"], make_cb("robot_state"), 10
    )
    node_robot.create_subscription(
        PoseStamped, stream_topics["left_eef"], make_cb("left_eef"), 10
    )
    node_robot.create_subscription(
        PoseStamped, stream_topics["right_eef"], make_cb("right_eef"), 10
    )
    for name in sensor_streams:
        node_sensor.create_subscription(Image, stream_topics[name], make_cb(name), 10)

    print(f"robot domain : {args.control_domain}")
    print(f"sensor domain: {args.sensor_domain}")
    print(f"anchor={args.anchor!r} window={args.window_size} slop_ms={args.slop_ms}")
    print(f"collecting for {args.duration_s:.1f}s ...")

    deadline = time.monotonic() + args.duration_s
    while time.monotonic() < deadline:
        exec_robot.spin_once(timeout_sec=0.01)
        exec_sensor.spin_once(timeout_sec=0.01)

    print("\n=== effective rate (recv inter-arrival) ===")
    hz_low = []
    for name, item in stats.items():
        hz = _effective_hz(item.inter_recv_ms)
        n = len(item.recv_stamps_ns)
        if hz is None:
            print(f"{name:24s} n={n:4d}  hz= n/a")
            continue
        flag = " LOW" if hz < args.expected_hz * 0.8 else ""
        if flag:
            hz_low.append(name)
        interval_p95 = (
            statistics.quantiles(item.inter_recv_ms, n=20)[18]
            if len(item.inter_recv_ms) >= 20
            else (max(item.inter_recv_ms) if item.inter_recv_ms else float("nan"))
        )
        print(
            f"{name:24s} n={n:4d}  hz={hz:5.1f}{flag:4s}  "
            f"p95_interval_ms={interval_p95:6.1f}"
        )

    anchor_stats = stats.get(args.anchor)
    if anchor_stats is None:
        print(f"\nunknown anchor stream {args.anchor!r}")
        return 1

    others = {name: item for name, item in stats.items() if name != args.anchor}
    print("\n=== latest-frame skew vs anchor (ms) ===")
    if anchor_stats.recv_stamps_ns:
        anchor_recv = anchor_stats.recv_stamps_ns[-1]
        anchor_header = anchor_stats.header_stamps_ns[-1]
        for name, item in sorted(others.items()):
            recv_skew = _nearest_skew_ms(anchor_recv, item.recv_stamps_ns)
            header_skew = _nearest_skew_ms(anchor_header, item.header_stamps_ns)
            print(
                f"{name:24s}  recv={recv_skew:+.1f}  header={header_skew:+.1f}"
                if recv_skew is not None and header_skew is not None
                else f"{name:24s}  no samples"
            )

    print("\n=== window alignment simulation ===")
    for stamp_key in ("recv", "header"):
        worst, pair, failures = _simulate_window_skew(
            anchor=anchor_stats,
            others=others,
            count=args.window_size,
            stamp_key=stamp_key,
            slop_ms=args.slop_ms,
        )
        ok = failures == 0
        print(
            f"stamp_source={stamp_key:6s}  worst={worst:6.1f}ms ({pair})  "
            f"failures={failures}  {'OK' if ok else 'FAIL'}"
        )

    print("\n=== hints ===")
    if hz_low:
        print(f"- low frame rate on: {', '.join(hz_low)} (USB/CPU load/dropped frames)")
    header_worst, _, _ = _simulate_window_skew(
        anchor=anchor_stats,
        others=others,
        count=args.window_size,
        stamp_key="header",
        slop_ms=args.slop_ms,
    )
    recv_worst, _, recv_fail = _simulate_window_skew(
        anchor=anchor_stats,
        others=others,
        count=args.window_size,
        stamp_key="recv",
        slop_ms=args.slop_ms,
    )
    if header_worst > args.slop_ms and recv_worst <= args.slop_ms:
        print("- header stamps diverge across domains -> keep rollout data.bridge.stamp_source=recv")
    if recv_worst > args.slop_ms:
        needed = max(args.slop_ms, recv_worst * 1.2)
        print(
            f"- recv alignment still exceeds slop: try policy.inputs.slop_ms>={needed:.0f} "
            f"(30Hz frame period ~33ms; 107ms ~= 3 missed frames)"
        )
    if recv_fail:
        print("- run this script while infer_fm is NOT loading the model, then again during infer")
        print("  if skew jumps only during infer -> GPU/CPU starvation on ROS spin threads")

    return 0 if recv_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
