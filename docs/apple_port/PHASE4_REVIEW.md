# Hesiod Phase 4 Review

## Delivered

- Added a per-update Metal graph bridge with one `DeviceSession` per graph
  evaluation.
- Integrated resident Thermal and ADD Blend execution into the real
  `SpectralEqualizer.hsd` branch/convergence graph.
- Preserved host/OpenCL fallback for unsupported nodes and parameters.
- Added transfer, node backend, Metal command, synchronization, GPU-time, and
  parity diagnostics to the Phase 4 benchmark CLI.
- Pinned Hesiod to the HighMap fork commit containing the resident border and
  ridge support required for graph parity.
- Documented graph execution, node eligibility, residency, workload results,
  and benchmark evidence in this directory.

## Evidence and limitations

Fallback and resident outputs pass the `1e-2` comparison gate at 512²,
1024², 2048², and 4096². The largest observed difference is `0.00505316` at
512² and decreases with resolution in the measured runs. The terminal output
has one readback; preview and export remain host boundaries.

Coverage is intentionally narrow. CoherentNoise, SpectralEqualizer, non-ADD
Blend methods, masked/deposition/scaled Thermal, and non-identity post-process
configurations use the established host path. There is no universal scheduler,
persistent cross-update device cache, renderer rewrite, or HSD format change.

The Apple Metal path was built and exercised on Apple M3. The repository's
CTest configuration currently reports no registered tests, so the executable
test results and real-graph runs are the authoritative checks for this phase.
The no-Metal configuration is built and exercised separately before the branch
is published.

The current HighMap Metal-enabled executable suite ran 349 tests: 348 passed
and the pre-existing `PathSplines.PreservePathShape` test failed at
`0.15608564 > 0.15`. The same failure and value reproduce in the pre-Phase-4
baseline. The current no-Metal executable suite also ran 349 tests with the
same single baseline failure; Metal-specific tests are skipped there.

Phase 5 has not been started.
