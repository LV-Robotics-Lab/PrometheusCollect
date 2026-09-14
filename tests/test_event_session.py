from __future__ import annotations

import threading
import time
import struct

import pytest

from prometheus.sessions.event import (
    DEFAULT_PEDAL_MAP,
    EventSession,
    PedalInput,
    TtyInput,
    event_name,
    semantics_by_id,
    semantics_by_name,
)
from prometheus.sessions.session import SessionState, SessionStateError

RECORD_DRAG_SEMANTICS = {
    0: "record_start",
    1: "record_stop",
    2: "event_mark",
}


def test_event_session_polls_fifo_events():
    session = EventSession()
    session.start()
    assert session.wait_ready(1.0).state == SessionState.READY

    session.emit(0)
    session.emit(2, 3)

    assert session.poll() == (0, True)
    assert session.poll() == (2, 3)
    assert session.poll() is None

    session.close()


def test_event_session_waits_for_event():
    session = EventSession()
    session.start()

    thread = threading.Thread(
        target=lambda: (time.sleep(0.02), session.emit(1)),
        daemon=True,
    )
    thread.start()

    assert session.wait(timeout_s=1.0) == (1, True)
    assert session.wait(timeout_s=0.01) is None
    thread.join(timeout=1.0)

    session.close()


def test_event_session_lifecycle_errors():
    session = EventSession()

    with pytest.raises(SessionStateError):
        session.emit(0)

    session.start()
    session.close()

    with pytest.raises(SessionStateError):
        session.poll()


def test_event_session_can_own_pedal_input():
    session = EventSession(input_source="pedal")

    pedal = session.status().details["input_source"]
    assert pedal["name"] == "pedal"
    assert pedal["type"] == "PedalInput"
    assert pedal["device"] == "/dev/input/prometheus-pedal"
    assert pedal["pedal_map"] == DEFAULT_PEDAL_MAP


def test_event_session_can_own_composite_inputs():
    session = EventSession(input_source="pedal,tty")

    source = session.status().details["input_source"]
    assert source["name"] == "pedal,tty"
    assert source["type"] == "CompositeInput"
    assert [item["name"] for item in source["sources"]] == ["pedal", "tty"]


def test_event_session_rejects_unknown_input_source():
    with pytest.raises(ValueError, match="unsupported input source"):
        EventSession(input_source="keyboard")


def test_event_session_uses_custom_pedal_map_from_config():
    session = EventSession(
        input_source="pedal",
        event_cfg={"pedal": {"pedal_map": {99: 7}, "confirm_presses": 1}},
    )
    pedal = session.status().details["input_source"]
    assert pedal["pedal_map"] == {99: 7}
    assert pedal["confirm_presses"] == 1


def test_pedal_input_confirms_and_locks_key_edges():
    session = EventSession()
    session.start()
    pedal = PedalInput(session, confirm_presses=2, confirm_window_s=1.0, debounce_s=0.25)

    assert pedal._process_event_edge(0, "down", now=0.0) is None
    assert pedal._process_event_edge(0, "down", now=0.1) == (0, True)
    assert session.poll() == (0, True)

    assert pedal._process_event_edge(0, "down", now=0.2) is None
    assert pedal._process_event_edge(0, "up", now=0.3) is None
    assert pedal._process_event_edge(0, "down", now=0.4) is None
    assert pedal._process_event_edge(0, "down", now=0.5) == (0, True)

    session.close()


def test_tty_input_confirms_key_events():
    session = EventSession()
    session.start()
    tty_input = TtyInput(session, confirm_presses=2, confirm_window_s=1.0, debounce_s=0.25)

    assert tty_input._process_key_event(1, now=0.0) is None
    assert tty_input._process_key_event(1, now=0.1) == (1, True)
    assert session.poll() == (1, True)

    session.close()


def test_pedal_input_resets_expired_confirmation_window():
    session = EventSession()
    session.start()
    pedal = PedalInput(session, confirm_presses=2, confirm_window_s=1.0)

    assert pedal._process_event_edge(1, "down", now=0.0) is None
    assert pedal._process_event_edge(1, "down", now=2.0) is None
    assert pedal._process_event_edge(1, "down", now=2.1) == (1, True)
    assert session.poll() == (1, True)

    session.close()


def test_pedal_input_can_emit_on_single_press():
    session = EventSession()
    session.start()
    pedal = PedalInput(session, confirm_presses=1)

    assert pedal._process_event_edge(2, "down", now=0.0) == (2, True)
    assert session.poll() == (2, True)

    session.close()


def test_pedal_input_debounces_after_release():
    session = EventSession()
    session.start()
    pedal = PedalInput(session, confirm_presses=1, debounce_s=0.25)

    assert pedal._process_event_edge(0, "down", now=0.0) == (0, True)
    assert pedal._process_event_edge(0, "up", now=0.01) is None
    assert pedal._process_event_edge(0, "down", now=0.1) is None
    assert pedal._process_event_edge(0, "down", now=0.3) == (0, True)

    session.close()


def test_pedal_input_parses_linux_evdev_events():
    session = EventSession()
    session.start()
    pedal = PedalInput(session, pedal_map={30: 0}, confirm_presses=1)
    evdev_event = struct.Struct("llHHI")

    emitted = pedal._process_evdev_bytes(evdev_event.pack(0, 0, 1, 30, 1))

    assert emitted == [(0, True)]
    assert session.poll() == (0, True)

    session.close()


def test_semantics_helpers_roundtrip():
    event_cfg = {"semantics": RECORD_DRAG_SEMANTICS}
    assert semantics_by_id(event_cfg) == RECORD_DRAG_SEMANTICS
    assert semantics_by_name(event_cfg) == {
        "record_start": 0,
        "record_stop": 1,
        "event_mark": 2,
    }
    assert event_name(event_cfg, 1) == "record_stop"
