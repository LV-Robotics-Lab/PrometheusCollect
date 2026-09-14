"""Upload a saved local dataset to ModelScope without starting hardware."""

from __future__ import annotations

import fnmatch
import getpass
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def scan_files(root: Path, ignore_patterns: list[str]) -> list[dict[str, Any]]:
    """Inventory regular files; never follow links into other datasets."""
    files = []

    def ignored(relative: str) -> bool:
        return any(fnmatch.fnmatchcase(relative, pattern) for pattern in ignore_patterns)

    def walk_error(error: OSError) -> None:
        raise error

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
        for name in list(dirs):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if ignored(relative) or ignored(relative + "/"):
                dirs.remove(name)
            elif path.is_symlink():
                raise ValueError(f"目录包含符号链接，请移除或明确排除：{relative}")
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if ignored(relative):
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"不支持符号链接或特殊文件：{relative}")
            stat = path.stat()
            files.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return sorted(files, key=lambda item: item["path"])


def check_saved_episodes(root: Path, files: list[dict[str, Any]]) -> int:
    """Reject unfinished raw episodes based on recorder manifests, not pickle data."""
    episodes: set[Path] = set()
    for item in files:
        relative = PurePosixPath(item["path"])
        if relative.parent.name == "manifests" and relative.suffix == ".json":
            episodes.add(Path(relative.parent.parent))
        elif (
            relative.name in {"robot_state_dict.pkl", "executed_action_dict.pkl"}
            or (relative.parent.name == "camera" and relative.suffix in {".mp4", ".mkv"})
        ):
            episodes.add(Path(relative.parent.parent))
    for episode in sorted(episodes):
        manifest_dir = root / episode / "manifests"
        manifests = sorted(manifest_dir.glob("*.json"))
        required = {"camera.json", "tactile.json", "robot.json", "event.json"}
        if (root / episode / "action" / "executed_action_dict.pkl").exists():
            required.add("action.json")
        missing = required - {path.name for path in manifests}
        if missing:
            raise ValueError(f"episode {episode} 缺少保存清单 {sorted(missing)}，请先完成录制和保存")
        records = 0
        for path in manifests:
            if path.is_symlink():
                raise ValueError(f"保存清单不能是符号链接：{path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"无效的 episode 清单：{path}")
            # Disabled streams (for example tactile) legitimately write {}.
            for record in payload.values():
                records += 1
                if not isinstance(record, dict) or record.get("status") != "saved" or record.get("failure"):
                    raise ValueError(f"episode {episode} 尚未正常保存或已被丢弃：{path.name}")
        if not records:
            raise ValueError(f"episode {episode} 没有已保存的数据流记录")
    return len(episodes)


def _exact_pattern(path: str) -> str:
    # SDK allow_patterns uses fnmatch; escape filename metacharacters so the
    # uploaded file set is exactly the set shown in our preview.
    return "".join({"[": "[[]", "*": "[*]", "?": "[?]"}.get(char, char) for char in path)


def _resolve_paths(config: DictConfig) -> tuple[Path, Path, str]:
    if not config.local_dir:
        raise ValueError("请设置 local_dir，例如 local_dir=data/romoya-egg-stage2-0911")
    root = Path(str(config.local_dir)).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("local_dir 必须是数据集目录")
    project = Path(__file__).resolve().parents[2]
    if root in {Path("/"), Path.home(), project}:
        raise ValueError("请选择具体数据集目录，不能上传系统目录、用户目录或整个项目")
    output = Path(str(config.run.output_dir)).expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError("上传日志目录不能位于待上传数据集内部")
    prefix = str(config.path_in_repo)
    if prefix.startswith("/") or "\\" in prefix or ".." in prefix.split("/"):
        raise ValueError("path_in_repo 必须是仓库内相对路径，例如 raw/egg-stage2")
    prefix = PurePosixPath(prefix).as_posix() if prefix else ""
    if prefix == ".":
        prefix = ""
    if config.repo_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", str(config.repo_id)):
        raise ValueError("repo_id 格式应为 用户名或组织名/数据集名")
    if not config.dry_run and not config.repo_id:
        raise ValueError("正式上传必须设置 repo_id=你的用户名/数据集名")
    if int(config.max_workers) < 1 or int(config.max_workers) > 32:
        raise ValueError("max_workers 必须为 1–32")
    if config.visibility not in {"private", "public"}:
        raise ValueError("visibility 必须为 private 或 public")
    return root, output, prefix


def run_upload(config: DictConfig) -> dict[str, Any]:
    root, output, prefix = _resolve_paths(config)
    patterns = list(config.ignore_patterns)
    files = scan_files(root, patterns)
    if not files:
        raise ValueError("数据集为空，或者所有文件都被排除了")
    episodes = check_saved_episodes(root, files) if config.check_saved_episodes else None
    total_size = sum(item["size"] for item in files)
    output.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://modelscope.cn/datasets/{config.repo_id}" if config.repo_id else None
    result = {
        "status": "preview",
        "local_dir": str(root),
        "repo_id": config.repo_id,
        "repo_type": "dataset",
        "repo_url": repo_url,
        "path_in_repo": prefix,
        "revision": config.revision,
        "file_count": len(files),
        "total_bytes": total_size,
        "saved_episodes": episodes,
    }
    _write_json(output / "files.json", files)
    _write_json(output / "result.json", result)
    print(f"[upload] 本地目录：{root}", flush=True)
    print(f"[upload] 目标仓库：{repo_url or '尚未填写 repo_id'} / {prefix or '(根目录)'}", flush=True)
    print(f"[upload] 共 {len(files)} 个文件，{total_size / 1024**3:.3f} GiB；已保存 episode：{episodes}", flush=True)
    for item in files[:8]:
        print(f"  {item['path']} ({item['size']} bytes)", flush=True)
    print(f"[upload] 完整文件清单与结果：{output}", flush=True)
    if config.dry_run:
        print("[preview] 未登录或上传。确认后加 dry_run=false 执行上传。", flush=True)
        return result

    token = ""
    try:
        from modelscope_hub import HubApi
        from modelscope_hub.errors import NotExistError

        token = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
        if not token:
            if not sys.stdin.isatty():
                raise ValueError("请在交互终端输入令牌，或通过 MODELSCOPE_API_TOKEN 环境变量提供")
            token = getpass.getpass("ModelScope 访问令牌（需写权限，输入不显示）：").strip()
        if not token:
            raise ValueError("ModelScope 访问令牌不能为空")
        api = HubApi(token=token, endpoint="https://modelscope.cn")
        api.whoami()
        try:
            info = api.get_repo(str(config.repo_id), repo_type="dataset")
        except NotExistError:
            if not config.create_repo:
                raise ValueError("目标仓库不存在或当前令牌不可见。请核对权限或先在网页创建；也可设 create_repo=true（默认创建私有仓库）") from None
            info = api.create_repo(
                str(config.repo_id), repo_type="dataset", visibility=str(config.visibility),
                license=config.license,
            )
        print(f"[upload] 仓库可见性：{info.visibility}；同路径文件会更新。", flush=True)
        if scan_files(root, patterns) != files:
            raise RuntimeError("预览后本地数据已变化，请停止修改数据后重新运行")
        result["status"] = "uploading"
        _write_json(output / "result.json", result)
        commit = api.upload_folder(
            repo_id=str(config.repo_id),
            repo_type="dataset",
            folder_path=str(root),
            path_in_repo=prefix,
            revision=str(config.revision),
            commit_message=str(config.commit_message),
            allow_patterns=[_exact_pattern(item["path"]) for item in files],
            ignore_patterns=patterns,
            max_workers=int(config.max_workers),
            use_cache=True,
            sync_remote_repo=False,
        )
        if scan_files(root, patterns) != files:
            raise RuntimeError("上传期间本地数据发生变化，远端可能包含不一致数据；请停止录制后重新上传")
        result["status"] = "completed"
        # Keep only commit identities, not arbitrary server responses in logs.
        commits = commit if isinstance(commit, list) else [commit]
        result["commits"] = [
            {key: item[key] for key in ("oid", "commit_id", "commit_url") if key in item}
            for item in commits if isinstance(item, dict)
        ]
        print(f"[done] SDK 上传完成（包含已缓存且无需重传的文件）：{repo_url}", flush=True)
        return result
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        print("\n[upload] 上传中断；重新运行相同命令可复用 SDK 上传缓存。", flush=True)
        raise
    except Exception as exc:
        result["status"] = "failed"
        message = str(exc).replace(token, "[REDACTED]") if token else str(exc)
        result["error"] = f"{type(exc).__name__}: {message}"
        raise RuntimeError(result["error"]) from None
    finally:
        _write_json(output / "result.json", result)


@hydra.main(version_base=None, config_path="../configs", config_name="upload_dataset")
def main(config: DictConfig) -> None:
    # Resolve before authentication. No token belongs in Hydra's saved config.
    OmegaConf.resolve(config)
    run_upload(config)


if __name__ == "__main__":
    main()
