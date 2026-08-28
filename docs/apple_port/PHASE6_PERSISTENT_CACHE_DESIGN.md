# Phase 6 Persistent DeviceArray Cache Design

## Policy

The cache is optional and off by default. It is enabled only with
`HESIOD_METAL_PERSISTENT_CACHE=1`; the Phase 6 CLI benchmark enables that switch
for its dedicated experiment. This prevents normal graph updates from keeping
GPU resources alive unexpectedly.

The cache is scoped to one `GraphNode`, keyed by the owning
`VirtualArray*`, and stores geometry metadata: logical shape, tile shape, and
halo. A recomputed output is invalidated before its node runs. This reuses
unchanged upstream outputs during a targeted dirty update while preserving the
existing GNode dirty propagation policy.

## Lifetime and ownership

HighMap exposes `DeviceSession::adopt_completed(const DeviceArray&)`. The
operation finishes the source session, then adopts the completed resource into
the new session. Hesiod sees only `DeviceArray`; it never stores or passes an
`MTLBuffer`. Finished sessions release their scratch pools, so the bounded
cache retains completed result resources rather than an entire old session's
temporary allocation pool.

Cache insertion occurs after the graph session finishes. Cache hits adopt only
completed resources. An entry is rejected on empty data, geometry mismatch,
disabled mode, or a resource larger than the budget.

## Budget and invalidation

The default budget is one quarter of Metal's reported recommended working set.
`HESIOD_METAL_CACHE_MB` can set an explicit positive budget for experiments.
Entries use a simple least-recently-used timestamp and are evicted until the
new completed resource fits. Recomputed outputs are invalidated by identity;
geometry changes also invalidate an entry. Graph destruction releases the
graph-scoped cache.

The cache does not alter scheduler, renderer, HSD serialization, or
VirtualArray tile storage. It is not a disk cache and does not survive process
restart. Host previews and exports remain explicit readback boundaries.

## Acceptance rule

The cache is accepted only if targeted dirty updates show cache hits, no
behavioral parity change, and a reduction in host uploads without unbounded
resident memory. Otherwise the switch remains experimental and the default
path stays unchanged. Phase 6 benchmark results are recorded in
`PHASE6_CACHE_BENCHMARKS.md`.
