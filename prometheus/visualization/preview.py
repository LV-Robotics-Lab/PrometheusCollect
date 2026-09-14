from __future__ import annotations

from typing import Any


def _preview_cfg(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return dict(raw)








def start_collection_preview(data_session: Any, workflow_cfg: dict[str, Any]) -> Any | None:
    preview_cfg = _preview_cfg(workflow_cfg.get("preview"))
    if not bool(preview_cfg.get("enabled", False)):
        return None
    mode = str(preview_cfg.get("mode", "local")).strip().lower()
    if mode == "web":
        viewer = data_session.visualize(
            host=str(preview_cfg.get("host", "127.0.0.1")),
            port=int(preview_cfg.get("port", 7860)),
            refresh_ms=int(preview_cfg.get("refresh_ms", 100)),
        )
    else:
        from prometheus.visualization.local_preview import LocalDataPreview

        viewer = LocalDataPreview(
            data_session,
            window=str(preview_cfg.get("window", "Prometheus Collection Preview")),
            width=int(preview_cfg.get("width", 1280)),
            height=int(preview_cfg.get("height", 720)),
            fps=float(preview_cfg.get("fps", 20.0)),
            stale_timeout_s=float(preview_cfg.get("stale_timeout_s", 1.0)),
            always_on_top=bool(preview_cfg.get("always_on_top", False)),
            ui_mode="collection",
        )
    viewer.start()
    print(f"[preview] {viewer.url}", flush=True)
    return viewer


def set_preview_status(preview: Any | None, status: str, detail: str = "") -> None:
    if preview is None:
        return
    set_status = getattr(preview, "set_status", None)
    if callable(set_status):
        set_status(status, detail)


def poll_preview(preview: Any | None) -> None:
    if preview is None:
        return
    try:
        poll = getattr(preview, "poll", None)
        if callable(poll):
            poll()
    except Exception as exc:
        print(f"[preview] poll error: {exc}", flush=True)


def close_preview(preview: Any | None) -> None:
    if preview is None:
        return
    try:
        close = getattr(preview, "close", None)
        if callable(close):
            close()
    except Exception as exc:
        print(f"[preview] close error: {exc}", flush=True)






