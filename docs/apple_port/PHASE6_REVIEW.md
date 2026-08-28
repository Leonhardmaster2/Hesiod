# Hesiod Apple Metal Phase 6 Review

Phase 6 answers the residency question with a bounded yes: Hesiod now extends
resident execution to a corpus-selected GaborWaveFbm boundary, qualifies the
whole-logical-array path across real tiled graphs, and safely reuses completed
DeviceArrays for repeated edits when the opt-in cache remains within budget.
The fallback path remains the default compatibility contract.

## Forty-two questions

1. **What exact upstream and feature SHAs were used?** Hesiod used upstream
   `dev` `4c0ee2156b987ec221e92a229e81d4d9ec9c6f2b`, with feature baseline
   `dfbf63ec0d953e9e07e1bf3f7b77edd9c919b088`; the Phase 6 source checkpoints
   are `93c27db0` (HighMap pointer), `716d5d82` (Gabor), `32ae6db7` (cache),
   and `55d3d598` (diagnostics). HighMap used upstream `dev`
   `e0d1279fced75cf7556de233128abfb5650e25b5` and feature
   `f8b91e1f12c7e1ab9cb796c8aae822c880525734`. The final Hesiod branch tip is
   reported by the publication audit because this review is itself a feature
   commit.

2. **Which additional real graphs were selected?** Built-in
   `data/examples/MakePeriodic.hsd` (Graph B) and
   `data/examples/MakePeriodicStitching.hsd` (Graph C). The permanent Graph A
   remains `data/examples/SpectralEqualizer.hsd`.

3. **Which node family ranked highest in the corpus?** GaborWaveFbm: 178 graph
   node occurrences across 158 HSD files, including 154 example files.

4. **Was GaborWaveFbm made resident?** Yes. HighMap implements the Metal
   primitive and Hesiod selects it for compatible parameters.

5. **Which additional NoiseFbm/coherent-noise families became resident?** No
   new NoiseFbm family was added. The Phase 5 supported CoherentNoise FBM
   family remains resident; Phase 6's one major family was GaborWaveFbm.

6. **Which candidates remained CPU/OpenCL and why?** Value cubic and Worley
   noise, Ridged/IQ/Jordan/Parberry/PingPong/Swiss groups, standalone
   NoiseFbm, envelope/post-process variants, and host boundary nodes remain on
   fallback paths because their semantics or parity evidence are not covered
   by the staged evaluator.

7. **How many more built-in graphs now contain useful resident islands?** Two
   additional graphs were measured and qualified as useful resident islands:
   Graph B and Graph C. The corpus count is a potential-opportunity count, not
   a claim that all 154 Gabor-containing examples meet the eligibility rules.

8. **How many are fully resident for compatible configurations?** One of the
   three qualification graphs, Graph A, is fully resident for its compatible
   configuration. Graphs B and C intentionally retain their downstream
   periodic/stitching host boundary.

9. **What tiling layouts were qualified?** 1×1, 2×1, 1×2, 2×2, and 4×4 at
   512², plus 4×4 at 1024².

10. **What overlap configurations were qualified?** Overlap `0.25` for every
    multi-tile Graph C run. The 1×1 control used overlap `0`.

11. **Which operations require halos?** Thermal's neighborhood update,
    smoothing, gradient/advection-style neighborhood operations, and other
    tile-local algorithms that read adjacent cells require a halo. Phase 6 did
    not claim a generic resident halo scheduler.

12. **Which operations require global reductions?** Normalize/min-max range
    remapping requires a whole-logical-array reduction. SpectralEqualizer also
    has global-frequency semantics. The resident API keeps those semantics
    explicit rather than independently normalizing tiles.

13. **Which remain single-tile only?** SpectralEqualizer remains independently
    tiled-ineligible and is resident only in its existing single-tile compatible
    configuration. Global nonlocal flooding/path families remain fallback-only.
    Gabor's multi-tile qualification is whole-array gather/compute/scatter, not
    independent per-tile execution.

14. **Were any tile seam errors discovered?** No. The seam checks found no
    introduced numerical or visible seam.

15. **What was the maximum seam-specific parity error?** `3.0e-7` across the
    qualified Graph C runs; the maximum off-seam error was `4.2e-7`.

16. **How does tiled resident performance compare to fallback?** At 512²,
    Graph C resident wall time was 74.744–76.661 ms versus 87.245–95.975 ms
    fallback across the tested layouts. At 1024² 4×4 it was 744.549 ms versus
    758.313 ms. These are qualification spot samples, with parity and transfer
    behavior as the primary gate.

17. **How much memory does tiled resident execution use?** Graph C peak Metal
    resident bytes were 2,097,168 at 512² and 8,388,624 at 1024² 4×4. The
    resident Gabor source used zero host uploads and one terminal readback.

18. **Was persistent DeviceArray caching implemented?** Yes, behind the
    opt-in `HESIOD_METAL_PERSISTENT_CACHE=1` switch.

19. **If yes, what HighMap ownership change enabled it?** HighMap added
    `DeviceSession::adopt_completed(const DeviceArray&)`, allowing a completed
    DeviceArray to be adopted by a new session without exposing a raw
    `MTLBuffer`. Finished sessions release their scratch pools.

20. **Are only completed resources reused across evaluations?** Yes. Cache
    insertion occurs after `DeviceSession::finish()`, and adoption also
    validates/finishes the source resource.

21. **What is the cache budget policy?** The default is one quarter of the
    reported Metal recommended working set. `HESIOD_METAL_CACHE_MB` provides an
    explicit positive experimental budget.

22. **What eviction policy is used?** A simple least-recently-used timestamp;
    oldest entries are evicted until the new completed resource fits. Entries
    larger than the budget are rejected.

23. **What are the cache hit/miss/eviction counts in the benchmark?** At the
    default budget, 512²/1024²/2048²/4096² Blend edits each reported 2 hits,
    0 misses, and 0 evictions. The 4096² 128 MiB pressure run reported 1 hit,
    1 miss, and 1 eviction, with 1 upload and persistent bytes capped at 128
    MiB.

24. **Does a SpectralEqualizer edit still require uploads?** Cache off: 2
    uploads. Cache on: 0 uploads and 1 readback.

25. **Does a Thermal edit still require uploads?** Cache off: 2 uploads. Cache
    on: 0 uploads and 1 readback.

26. **Does a Blend edit still require uploads?** Cache off: 2 uploads. Cache
    on: 0 uploads and 1 readback when the parent resources fit the budget.

27. **What are the incremental edit speedups at 512²?** SpectralEqualizer
    1.59×, Thermal 1.44×, and Blend 1.81× in the matched spot samples.

28. **At 1024²?** SpectralEqualizer 1.28×, Thermal 1.34×, and Blend 2.56×.

29. **At 2048²?** SpectralEqualizer 1.13×, Thermal 1.21×, and Blend 2.70×.

30. **What happens at 4096²?** SpectralEqualizer improved 3,123.509→2,936.288
    ms, Thermal 2,040.691→1,825.564 ms, and Blend 284.597→102.937 ms, all
    with zero uploads on cache-on. CoherentNoise remained a full recompute at
    approximately 6.3 s, as expected.

31. **What is peak Metal memory with the cache enabled?** The 4096² primary
    evaluation peaked at 1,073,741,836 bytes. With a 512 MiB cache budget,
    persistent cached bytes were 201,326,592 (192 MiB), below the budget.

32. **Does cache-off still reproduce Phase 5 behavior?** Yes. Cache-off is the
    default, the 4096² primary parity is still `0.00117421`, and the measured
    resident wall time remains in the Phase 5 6.5–6.6 s range.

33. **Does the primary SpectralEqualizer graph still pass through 4096²?** Yes,
    with four resident nodes, zero uploads, one terminal readback, and parity
    `0.00117421`.

34. **Did Phase 6 regress the 6.5–6.6 s 4096² resident performance materially?**
    No. The current cache-off spot was 6,458.723 ms against the Phase 5
    repeated median of 6,593.112 ms; the cache-on spot was 6,519.256 ms.

35. **Are CPU/OpenCL/no-Metal paths still correct?** Yes. HighMap's no-Metal
    suite compiles and retains its stubs; Hesiod's primary and additional
    graphs pass no-Metal fallback parity, including 2×2 tiling.

36. **Do all HighMap Metal tests pass?** Yes: 31 of 31 Metal-focused tests,
    including Gabor parity and completed-resource adoption.

37. **Is the pre-existing PathSplines failure still the only known full-suite
    failure?** Yes. Metal-enabled HighMap is 352 passed plus that one existing
    `PathSplines.PreservePathShape` failure; no-Metal is 321 passed, 31 skipped,
    plus the same one failure.

38. **Are both Git histories linear and clean?** The feature-only ranges are
    linear with no merge commits; final status and merge-base audits are part
    of the publication check. Existing untracked build/output artifacts remain
    uncommitted by design.

39. **Are AI-authored/co-authored commits still zero?** Yes. The final audit
    scans full commit bodies case-insensitively for all requested AI names and
    trailers; expected counts are zero.

40. **Is a universal graph scheduler now justified?** No. The evidence supports
    explicit node eligibility, established dirty propagation, and an optional
    cache—not a heterogeneous scheduler or graph compiler.

41. **Is QTerrainRenderer Metal migration now justified?** No. The renderer
    remains out of scope and untouched; compute residency is independently
    measurable without coupling it to rendering.

42. **What should Phase 7 actually do?** Only after a separately approved
    phase, use corpus evidence to select the next single residency breaker or
    halo-safe tiled family, then repeat parity, memory, and no-Metal gates. Keep
    the cache opt-in until longer-lived application evidence exists; do not
    start a universal scheduler or QTerrainRenderer migration automatically.

## Gate decision

Gate A passes for one corpus-selected major family: GaborWaveFbm is resident,
two additional real graphs have useful resident islands, and unsupported noise
families remain explicit fallback paths.

Gate B passes for the qualified whole-array tiled contract: all requested
layouts used by Graph C pass seam-specific parity with no implicit correction.

Gate C passes as an experimental opt-in: completed-resource adoption is safe,
bounded, invalidation-aware, and removes uploads for unchanged downstream
inputs. The default cache-off path remains available.

Phase 6 stops here. No Phase 7 work is started by this branch.
