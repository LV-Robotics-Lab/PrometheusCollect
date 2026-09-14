from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from scripts.prepare_yam_i2rt import prepare


@pytest.mark.parametrize("use_local_source", [True, False])
def test_prepare_pinned_source_ignores_dirty_tree_and_supports_fetch(tmp_path, use_local_source):
    repository = tmp_path / "upstream"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / "value.txt").write_text("base\n")
    subprocess.run(["git", "-C", str(repository), "add", "value.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "commit", "--quiet", "-m", "fixture"], check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    (repository / "value.txt").write_text("uncommitted local change\n")
    patch = b"diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-base\n+patched\n"
    lock = {
        "repository": repository.as_uri(),
        "base_commit": commit,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
    }
    release = tmp_path / "release"
    release.mkdir()
    (release / "lock.json").write_text(json.dumps(lock))
    (release / "compat.patch").write_bytes(patch)
    output = tmp_path / "prepared"
    prepare(repository if use_local_source else None, output, release=release)
    assert (output / "value.txt").read_text() == "patched\n"
    assert (repository / "value.txt").read_text() == "uncommitted local change\n"
    assert json.loads((output / "prometheus_i2rt_release.json").read_text()) == lock
    with pytest.raises(FileExistsError):
        prepare(None, output, release=release)
