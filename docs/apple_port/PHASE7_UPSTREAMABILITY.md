# Phase 7 Upstreamability

## HighMap

Ready for an upstream review as a focused backend change:

* one `morphological_gradient` Metal kernel with the same clamped disk-window
  semantics as the existing OpenCL implementation;
* one resident `DeviceSession` operation and synchronous wrapper;
* matching no-Metal stubs;
* direct OpenCL parity coverage, boundary/extreme-value coverage, and a
  three-backend benchmark;
* existing shader-source embedding and release metallib packaging paths remain
  the only resource mechanisms.

The HighMap patch is independent of Hesiod and Qt. An upstream review should
check naming and API placement against the current `dev`, then run native
Linux/Windows CI and an Apple release configuration that has `metal` and
`metallib` available.

## Hesiod

Ready for a separate integration review:

* explicit `MorphologicalGradient` eligibility for single logical arrays;
* exact fallback diagnostics for unavailable Metal, multi-tile halo, missing
  ports, and closed sessions;
* the session lifecycle guard shared by existing resident candidates;
* Graph D qualification and regression documentation.

The Phase 7 cache-soak CLI and persistent cache remain experimental support
for this branch. They should either be split into a separately reviewed
diagnostics change or retained behind the existing environment flags. The
deprecated node policy should also be confirmed by maintainers before making
the eligibility bridge a long-term public behavior.

## Remaining review work

Before the first upstream pull request:

1. Re-run the latest upstream fetch and final history/authorship audits.
2. Run Linux and Windows CI, plus an Apple Release build with precompiled
   metallib generation and a real application bundle/package test.
3. Review the known unrelated `PathSplines.PreservePathShape` failure as its
   own issue; it is unchanged by Phase 7.
4. Decide whether the cache/soak harness belongs in the initial integration
   patch or in follow-up diagnostics work.
5. Submit HighMap's reusable backend change first, followed by Hesiod's
   optional graph integration after the API is accepted.

No upstream pull request is opened by this task.
