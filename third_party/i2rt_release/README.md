This directory is the reproducible I2RT dependency for YAM collection.
`lock.json` identifies upstream source and the SHA-256 of `compat.patch`.

The patch carries the required drag tuning, gripper torque primitive, clean CAN
shutdown, NumPy-compatible dependency metadata, and runtime feedback health API.
It also carries hardware-free tests for these additions. It excludes local rig
examples, model changes, and unrelated working-tree files.

From the PrometheusCollect root, fetch and prepare the pinned source:

```bash
.venv/bin/python scripts/prepare_yam_i2rt.py \
  --output outputs/dependencies/i2rt-yam
```

For offline preparation, add `--source /path/to/i2rt` with a local Git repository
containing the pinned commit. Neither the original Prometheus repository nor a
sibling I2RT checkout is required to run collection. The command uses
`git archive`, verifies the patch hash and `git apply --check`, then applies the
patch in a temporary directory before copying the result to the requested new
output directory. See the root README for installation into the ROS environment.

When updating the dependency, review the patch and update its hash and base
commit together; prepare and test a new output directory before installing it.
The source identity is also copied to `prometheus_i2rt_release.json` in that tree.
