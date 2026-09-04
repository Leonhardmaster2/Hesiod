# Hesiod Phase 5 Review

Date: 2026-08-28 historical measurements; review refreshed 2026-09-04.
HighMap feature revision at the current published baseline:
`d335ec051f7c7d5cb241e80baa5e4139cf4361de`. The Hesiod feature branch was
rebased onto upstream `dev` at `b44dfe8203ab827c9f76b84dacd18a4f25922d19`;
the pre-refresh implementation baseline was `ac11110a`.

The historical measurements and conclusions below are retained. A fresh
post-sync 512²/1024²/2048²/4096² qualification reproduced resident execution
with four resident nodes, zero uploads, one terminal readback, and parity PASS
at every size; see `APPLE_METAL_CURRENT_STATE.md` and `PHASE8_REVIEW.md` for
the exact current values.

## Decision gate

1. **Why was the Phase 4 resident path slower at 4096²?** The single sample
   mixed a resident Metal branch with host/OpenCL CoherentNoise and the
   bandwidth-heavy OpenCL SpectralEqualizer. Repeated runs showed strong
   order/system-state dependence, not a stable Metal allocation limit.
2. **Was the regression reproducible?** The broad 21–24 s versus 11–13 s
   order-dependent bands were reproducible across five samples per order; the
   exact original single-run delta was not stable.
3. **Was unified-memory pressure involved?** Resource contention is plausible
   and supported by variable RSS/order sensitivity, but the Metal working-set
   limit was not approached. A standalone held-DeviceArray A/B harness was not
   completed, so this is not isolated as the sole cause.
4. **Was CPU/GPU overlap involved?** The old graph mixed OpenCL host-side work
   and Metal execution. The new graph is serialized through two command
   buffers, with one scalar reduction synchronization, and removes that mixed
   heavy branch. Direct overlap was not separately traced.
5. **Was thermal throttling involved?** It was not separately instrumented;
   the data cannot attribute a percentage to throttling.
6. **Was resource allocation involved?** No material allocation bottleneck was
   found: Phase 4 peak Metal bytes were ~320 MiB, allocation time was small,
   and allocations were reused.
7. **Was benchmark ordering involved?** Yes. It is the clearest measured
   confounder and is why both execution orders were collected.
8. **Did CoherentNoise become fully resident?** Yes, for compatible FBM,
   supported noise, no-envelope, single-tile configurations; generated noise
   uses zero host upload bytes.
9. **Did SpectralEqualizer become fully resident?** Yes, for the compatible
   single-tile, no-mask configuration; its blur pyramid and accumulation stay
   in the DeviceSession.
10. **Which generic post-process operations became resident?** The active
    automatic range remap used by the selected FBM node became resident via a
    scalar min/max reduction and pointwise normalize. Identity settings remain
    resident by eligibility.
11. **Is the primary graph now fully GPU-resident until final readback?** Yes:
    CoherentNoise, SpectralEqualizer, Thermal, and ADD Blend are resident, with
    zero host compute nodes and one terminal readback.
12. **How many uploads remain?** Zero for the selected graph.
13. **How many readbacks remain?** One terminal output readback; preview/export
    remain explicit host boundaries.
14. **What is the graph speedup at 512²?** Final fresh-run mean: 4.45×,
    94.925 ms fallback versus 21.308 ms resident.
15. **At 1024²?** 2.05×, 250.236 ms versus 122.364 ms.
16. **At 2048²?** 1.49×, 1,270.759 ms versus 854.844 ms.
17. **At 4096²?** 1.32× by median, 8,707.482 ms versus 6,593.112 ms.
18. **What is peak resident memory at each resolution?** 512²:
    16,777,228 bytes; 1024²: 67,108,876; 2048²: 268,435,468; 4096²:
    1,073,741,836.
19. **Which optimization produced the largest improvement?** Removing the
    host/OpenCL SpectralEqualizer boundary while keeping its blur stages
    resident produced the largest graph-level effect; resident FBM generation
    removed the source upload and completed the chain.
20. **Which attempted optimization did not help?** The old partial-residency
    path was not a reliable 4096² optimization when measured in isolation;
    it exposed order-sensitive contention. No speculative fused generic
    post-process kernel was kept without a semantic/performance case.
21. **Which nodes remain major residency breakers across the graph corpus?**
    GaborWaveFbm by frequency, NoiseFbm's non-FBM families and their
    downstream Vorolines/Rifts/cloud nodes, hydraulic downstream nodes, and
    visual Colorize/MixTexture branches.
22. **Should more node families be ported next?** Yes, but by corpus and
    boundary evidence: first audit GaborWaveFbm and the repeated NoiseFbm
    consumers, then hydraulic/visual boundaries separately.
23. **Should persistent DeviceArray caching across updates be implemented?**
    Not yet. The safe one-session lifetime is established; persistent caching
    needs dirty-lifetime and bounded-memory measurements first.
24. **Is one DeviceSession per evaluation still appropriate?** Yes. It keeps
    ownership, reuse, and synchronization bounded and clear for the current
    explicit policy.
25. **Is a residency-aware graph planner now justified?** No. The explicit
    resident-if-compatible policy handles this graph; corpus evidence calls
    for targeted ports, not a universal cost model yet.
26. **Is 4096² practical on the tested Apple Silicon machine?** Yes for this
    compatible single-tile resident graph: about 6.5–6.6 s in the fresh runs,
    ~1.0 GiB peak Metal bytes, and parity pass. Other graphs/configurations
    remain unqualified.
27. **Are CPU/OpenCL fallback paths still clean?** Yes for compute: Hesiod
    no-Metal graph parity passes, all six bootstraps batch-load, and the HighMap
    no-Metal suite retains its baseline result. A normal Qt file launch under
    the headless `offscreen` platform reproduces the pre-existing Phase 4
    segmentation fault in both Metal and no-Metal builds, so it is not a new
    fallback-path failure.
28. **Is HighMap still independently usable?** Yes. Synchronous wrappers,
    resident APIs, focused parity tests, Metal suite, full suite, and no-Metal
    build all remain available.
29. **Are the HighMap and Hesiod repositories/submodules clean and
    reproducible?** The tracked histories and submodule pin are clean after
    the final docs commit; pre-existing untracked build artifacts are
    preserved outside the commits. HighMap is pinned to the published
    `702a3acb1` feature revision.
30. **What should Phase 6 do?** Audit and benchmark GaborWaveFbm plus the
    highest-scoring NoiseFbm consumers, add broader tiled/overlap contracts,
    and study bounded persistent DeviceArray caching. Keep renderer work and a
    universal scheduler as separate decisions.

## Scope stop

Phase 5 does not port QTerrainRenderer/OpenGL, introduce a universal graph
scheduler, persist DeviceArrays across updates, or start Phase 6.
