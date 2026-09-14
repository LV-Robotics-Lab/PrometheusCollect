from __future__ import annotations

import os
import time

import pytest

from prometheus.sessions.event import EventSession, PEDAL_DEVICE

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_EVENT_COUNT = 1


def collect_pedal_events(timeout_s: float, expected_events: int) -> list[tuple[int, object]]:
    session = EventSession(input_source="pedal")
    events: list[tuple[int, object]] = []
    session.start()
    try:
        deadline = time.monotonic() + timeout_s
        while len(events) < expected_events and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            event = session.wait(timeout_s=min(0.5, remaining))
            if event is not None:
                events.append(event)
                print(f"{time.strftime('%H:%M:%S')} {event}")
    finally:
        session.close()
    return events


def test_live_pedal_input():
    if os.environ.get("PROMETHEUS_LIVE_PEDAL") != "1":
        pytest.skip("set PROMETHEUS_LIVE_PEDAL=1 to read the physical pedal")

    timeout_s = float(os.environ.get("PROMETHEUS_PEDAL_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    expected_events = int(os.environ.get("PROMETHEUS_PEDAL_EVENTS", DEFAULT_EVENT_COUNT))
    events = collect_pedal_events(timeout_s=timeout_s, expected_events=expected_events)
    assert len(events) >= expected_events


def main() -> int:
    timeout_s = float(os.environ.get("PROMETHEUS_PEDAL_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    expected_events = int(os.environ.get("PROMETHEUS_PEDAL_EVENTS", DEFAULT_EVENT_COUNT))
    print(f"device: {PEDAL_DEVICE}")
    print(f"waiting for {expected_events} event(s), timeout {timeout_s:.1f}s")
    print("default pedal logic requires two presses of the same key within 1s")
    events = collect_pedal_events(timeout_s=timeout_s, expected_events=expected_events)
    if not events:
        print("no pedal events received")
        return 1
    print(f"received {len(events)} event(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
