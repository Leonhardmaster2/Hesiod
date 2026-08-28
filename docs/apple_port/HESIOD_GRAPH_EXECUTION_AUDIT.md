# Hesiod Phase 4 Graph Execution Audit

## Execution path

The existing runtime path is:

`GraphManager::load_from_file` → `GraphNode::json_from` → node factory and
`Graph::new_link` → `GraphNode::update` → GNode topological update →
`BaseNode::compute` → the node's established `compute_fct`.

Phase 4 adds a scoped `MetalGraphExecution` around each `GraphNode::update`.
`BaseNode::compute` gives the scope a chance to prepare a node, while Thermal
and Blend explicitly opt into the resident path. Every other node remains on
its existing CPU/OpenCL implementation.

## Data ownership and boundaries

GNode links share one `Data<T>` object. For `hmap::VirtualArray` ports, linked
input and output ports therefore resolve to the same VirtualArray address. The
resident bridge uses that address as the lifetime-scoped key for its
`DeviceArray` map.

`DataPreview` converts a VirtualArray with `VirtualArray::to_array`, so preview
is an explicit host boundary. `GraphManager::export_flatten` likewise consumes
host VirtualArrays; Phase 4 leaves that export contract unchanged. HSD
serialization contains graph configuration and node parameters, not Metal
objects or device allocations.

## Observed real graph

`Hesiod/data/examples/SpectralEqualizer.hsd` is the Phase 4 workload. Its
connected topology is:

```text
8 CoherentNoise ──┬─> 9 SpectralEqualizer ──┐
                  └─> 11 Thermal ───────────┴─> 10 Blend (final)
```

The observed update order is `8, 9, 11, 10` (with GNode's runtime counters
including repeated evaluations caused by graph setup). Thermal and Blend form
the resident branch/convergence opportunity; CoherentNoise and
SpectralEqualizer stay on the host path.

The dirty update `graph->update("11")` recomputes the Thermal node and its
Blend consumer while leaving the unrelated SpectralEqualizer branch cached.
The benchmark reports this as two resident nodes and zero host nodes for the
edited update.

## Deliberate scope

There is no universal scheduler, persistent device cache, renderer rewrite, or
new HSD semantic. Residency lasts for one graph update and is released or
materialized at the update boundary. This keeps the existing graph semantics,
incremental update behavior, preview behavior, and export behavior intact.
