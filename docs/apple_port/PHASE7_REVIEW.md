# Phase 7 Review and Decision Gate

1. **What were the final Hesiod and HighMap upstream SHAs?** Hesiod
   `324c3ed51202ec6aa43d89730da39c913060d392`; HighMap
   `e0d1279fced75cf7556de233128abfb5650e25b5`.
2. **What were the final feature SHAs?** The Phase 7 implementation commits
   are Hesiod `58f8dc627e1f5d7ae4462557c1f1c5db81b2811a` and HighMap
   `a0c416dc54d3c57143a589cbf2118b6ceb5bb489`; the final branch heads are
   reported from `git rev-parse` after publication.
3. **Which remaining residency breaker ranked highest?**
   `MorphologicalGradient` after excluding already-qualified families.
4. **Why was it selected?** Four real graphs use it; it is a bounded regular
   neighborhood operation with direct CPU/OpenCL semantics and a small Metal
   surface.
5. **What was the runner-up?** `HydraulicStreamLog`, deferred because its
   iterative flow, multiple outputs, remaps, and deposition tree exceed the
   one-family scope.
6. **Was the selected family ported to Metal?** Yes, in HighMap.
7. **Was a resident DeviceArray path implemented?** Yes; a
   `DeviceSession::morphological_gradient` operation keeps the input/output on
   the Metal session for the qualified case.
8. **Is the synchronous API still correct?** Yes; the synchronous wrapper is
   parity-tested against the existing OpenCL implementation.
9. **Which parameters/configurations are supported?** A connected float
   input/output, non-negative pixel radius, one logical tile, clamped disk
   neighborhood, and identity post-process; active range remap uses the
   existing resident normalize operation.
10. **Which remain fallback-only?** Metal-unavailable builds, closed sessions,
    missing ports, multi-tile arrays/halo execution, and unsupported resident
    input/output state use the original CPU/OpenCL path. Legacy inverse,
    non-identity gain/gamma, smoothing, saturation, and mix remain host-only
    post-processing boundaries.
11. **Is the family tile-local, halo-required, global-reduction, or
    global-nonlocal?** Halo-required.
12. **Which tiling configurations were qualified?** Single logical 1×1 was
    qualified for Metal. 2×1, 1×2, 2×2, and 4×4 were qualified as explicit
    fallback configurations with whole-output and seam checks.
13. **What was maximum seam error?** `0.00000000` for the selected family’s
    tiled Graph D checks; the retained broader Graph B/C maximum is
    `0.00000024` with off-seam maximum `0.00000030`.
14. **Which Graph D was selected?** `data/bootstraps/texturing_101.hsd`,
    `MorphologicalGradient` id 26, with real terrain/soil input and
    color/texture/advection/presentation downstream.
15. **How many host/device boundaries existed before?** In the host-only
    pre-Phase-7 baseline, the selected family had no Metal transfers and was a
    host fallback island. In a resident comparison, that is the selected
    host-stage boundary rather than a resident DeviceArray chain.
16. **How many remain after Phase 7?** The measured Graph D resident path has
    one host-to-device input upload and one terminal device-to-host readback;
    the selected core itself adds no intermediate transfer. Host-only
    downstream nodes remain explicit boundaries.
17. **What is Graph D speedup at 512²?** 1.31× median: 227.725 ms fallback
    versus 174.080 ms resident.
18. **At 1024²?** 1.06× median: 1170.064 ms fallback versus 1103.133 ms
    resident.
19. **At 2048²?** A full Graph D resident speedup was not claimed: the
    host-heavy graph’s fallback spot run took 13,554.375 ms and the resident
    run was outside the practical measurement window. The isolated selected
    HighMap operation measured 5.62× CPU/Metal at 2048².
20. **At 4096²?** A full Graph D speedup was not claimed because that graph is
    not practical at this size on the validation host. The isolated selected
    operation measured 5.85× CPU/Metal at 4096².
21. **Did primary Graph A performance regress?** No. The final single-tile
    resident run was faster than the Phase 6 comparison point, with unchanged
    four-node residency and one terminal readback.
22. **What is Graph A’s final 4096² runtime?** 6085.190 ms resident; the
    same-run host fallback was 8393.292 ms.
23. **Did primary Graph A parity remain valid?** Yes: max absolute difference
    `0.00117421`, PASS.
24. **What is peak memory for Graph D?** Largest measured full-graph Graph D
    point (1024²) used 12,582,912 resident bytes and reached 545,112,064
    process RSS. The isolated 4096² operation peaked at 134,217,728 resident
    bytes.
25. **Does the new family benefit from persistent caching?** No measured
    benefit in the selected Graph D, because its host-produced input and
    host-heavy downstream force a terminal readback. A resident downstream
    chain could reuse it, but that is not claimed here.
26. **Did the long-lived cache soak reveal stale values?** No. Both budgets
    produced 102 finite output checks, zero invalid outputs, and PASS parity.
27. **Any use-after-free/resource lifetime issue?** No. The 100-cycle run
    completed with graph destruction/recreation and final resident bytes zero.
28. **Any cross-project cache contamination?** No. The alternate-project and
    reloaded-project fingerprint check passed with zero collisions.
29. **Did cache memory remain within budget?** Yes: 4 MiB maximum under the
    128 MiB budget, and exactly 1 MiB maximum under the 1 MiB pressure budget.
30. **How many hits/misses/evictions occurred under pressure?** 72 hits, 76
    misses, and 105 evictions at 1 MiB.
31. **Is persistent cache ready for default-on?** No.
32. **If not, why not?** It is only bounded and soak-validated behind an
    opt-in flag; selected Graph D does not need it, and release-wide cache
    policy has not been established.
33. **Does cache-off still remain fully supported?** Yes; it is the default
    and all Phase 6/7 graph paths were tested with it.
34. **Did normal application editing remain stable?** The incremental CLI
    edit matrix and 100-cycle lifecycle run remained stable with no node errors.
    A GUI editing soak was not run in this headless validation.
35. **Are Metal release resources/metallibs packaged correctly?** The CMake
    path is structurally correct: release builds with `metal`, `metallib`, and
    `xxd` compile and embed the library; the current Command Line Tools image
    lacks those utilities, so a signed precompiled bundle was not produced
    here.
36. **Is first-use pipeline creation acceptable?** Yes. The focused new
    kernel test initialized the library/pipeline successfully; warm samples
    reused the cached pipeline.
37. **Are failure/fallback diagnostics adequate?** Yes. Node diagnostics now
    distinguish unavailable Metal, closed session, missing ports, and
    multi-tile halo fallback.
38. **Do all Metal-focused HighMap tests pass?** Yes, 32/32.
39. **Does the full suite introduce any new failure?** No. The full suite had
    321 passes and the same pre-existing `PathSplines.PreservePathShape`
    failure.
40. **Does no-Metal remain correct?** Yes: 321 passes, 32 Metal tests skipped,
    and the same pre-existing spline failure.
41. **Are Windows/Linux paths still structurally clean?** Yes: Metal is
    capability-gated with a no-Metal stub and no Qt/Hesiod dependency in the
    HighMap backend. Native Windows/Linux CI remains required before upstream.
42. **Is HighMap still independent of Hesiod/Qt?** Yes.
43. **Which HighMap changes are ready for an upstream PR?** The kernel,
    resident/synchronous API, no-Metal stub, parity test, and benchmark are
    ready for focused upstream review.
44. **Which Hesiod changes are ready for an upstream PR?** The explicit
    eligibility bridge, lifecycle guard, precise fallback diagnostics, and
    integration documentation are ready for separate review.
45. **Which pieces remain experimental?** The persistent cache, soak CLI,
    single-tile-only policy, and compatibility support for the deprecated node.
46. **Is a universal graph scheduler now justified?** No.
47. **Is renderer migration now justified?** No; QTerrainRenderer remains
    untouched and outside this compute boundary.
48. **Should Phase 8 be another performance phase?** No.
49. **Or should Phase 8 be upstream/cleanup/release preparation?** Upstream,
    cleanup, CI, and release preparation should come first.
50. **What exact work remains before opening the first upstream PR?** Re-fetch
    and audit final histories, run native Linux/Windows CI, run an Apple
    Release build with precompiled metallib tools and bundle packaging, review
    the unrelated spline failure, decide whether cache/soak code is split from
    the initial patch, and submit HighMap before the dependent Hesiod
    integration. No pull request is opened by this task, and Phase 8 is not
    started.

## Decision

Phase 7 resolves one additional real residency boundary with a bounded,
parity-tested Metal implementation while preserving CPU/OpenCL/no-Metal
fallbacks, cache opt-in, clean histories, and renderer separation. The work is
ready to stop here for upstream/cleanup review.
