# PrometheusCollect

This is an independent data collection project. It does not depend on another
Prometheus checkout. Keep training, model serving, policy execution, rollout,
and DAgger outside this project.

- Workflows compose HardwareSession, DataSession and EventSession.
- HardwareSession owns robot and sensor lifecycles. Do not open hardware from
  collection workflows directly.
- DataSession owns stream subscriptions, buffers, alignment and raw recording.
- Keep Hydra configuration and readiness, error, stop and cleanup contracts
  explicit. Hardware failures must invalidate the active episode.
- Preserve raw episode schemas and the YAM release fixes when refactoring.
- Treat third_party as external code; edit only for an explicitly required
  dependency change. Update the I2RT patch hash with any patch change.
- Tests run without physical hardware by default. Do not enable live tests or
  send CAN commands unless the user explicitly requests a hardware run.
- Validate package installation outside the source directory when changing
  entry points or configuration packaging.
