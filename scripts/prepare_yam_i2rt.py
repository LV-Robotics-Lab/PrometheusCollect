#!/usr/bin/env python3
"""Export pinned I2RT source and apply the reviewed Prometheus compatibility patch."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


RELEASE = Path(__file__).resolve().parents[1] / "third_party" / "i2rt_release"


@contextmanager
def source_repository(source: Path | None, lock: dict):
    if source is not None:
        yield source.resolve()
        return
    with tempfile.TemporaryDirectory(prefix="yam-i2rt-upstream-") as temporary:
        repository = Path(temporary)
        subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
        subprocess.run(
            ["git", "-C", str(repository), "fetch", "--depth=1", lock["repository"], lock["base_commit"]],
            check=True,
        )
        yield repository


def prepare(source: Path | None, output: Path, *, release: Path = RELEASE) -> Path:
    lock = json.loads((release / "lock.json").read_text())
    patch = release / "compat.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["patch_sha256"]:
        raise ValueError("I2RT compatibility patch checksum does not match lock.json")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"output already exists: {output}; use a new output directory"
        )
    with source_repository(source, lock) as repository:
        return export_source(repository, output, patch=patch, lock=lock)


def export_source(source: Path, output: Path, *, patch: Path, lock: dict) -> Path:
    # git archive reads the pinned commit, never the developer's working tree.
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "cat-file",
            "-e",
            f"{lock['base_commit']}^{{commit}}",
        ],
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yam-i2rt-") as temporary:
        archive = Path(temporary) / "source.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "archive",
                "--format=tar",
                "--output",
                str(archive),
                lock["base_commit"],
            ],
            check=True,
        )
        tree = Path(temporary) / "source"
        tree.mkdir()
        with tarfile.open(archive) as handle:
            handle.extractall(tree, filter="data")
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=tree, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=tree, check=True)
        (tree / "prometheus_i2rt_release.json").write_text(
            json.dumps(lock, indent=2) + "\n"
        )
        import shutil

        shutil.copytree(tree, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="optional local I2RT git repository; otherwise fetch the pinned upstream commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory for reproducible source",
    )
    args = parser.parse_args()
    output = prepare(args.source, args.output)
    print(f"[ready] prepared pinned YAM dependency: {output}")
    print(
        "[ready] install this directory with uv pip install --python <collection-python> -e <directory>"
    )


if __name__ == "__main__":
    main()
