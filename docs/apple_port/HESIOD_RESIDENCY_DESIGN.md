# Hesiod Phase 4 Residency Design

## Lifetime

One `MetalGraphExecution` and one shared-storage `DeviceSession` are created
per `GraphNode::update` call. The scope is not serialized and does not survive
the update. The environment switch `HESIOD_METAL_RESIDENT=0` (also `false` or
`off`) disables the bridge and leaves the established implementation active.

## VirtualArray mapping

The bridge maps `const hmap::VirtualArray*` to `hmap::gpu::metal::DeviceArray`.
Because GNode links share their `Data<T>`, a downstream node sees the same key
as the upstream output. A host VirtualArray is uploaded only when a resident
node first consumes it. A resident output is bound to the linked VirtualArray
without a host readback.

The bridge tracks whether a mapped value was modified on the device. This is
important for fallback: an uploaded, unmodified input is already host-valid,
whereas a resident-produced value must be materialized before an unsupported
node consumes it. This avoids unnecessary readbacks while preserving host
semantics.

## Branches, convergence, and terminal outputs

Multiple consumers can retrieve the same mapped device value. At a convergence
point, a resident consumer can continue without a transfer; a host consumer
materializes only modified resident inputs. Terminal outputs are marked
`host_required` and are read back during `flush`, which also finishes the Metal
queue. Preview and export therefore retain their existing host-facing
contracts.

The mapping lifetime bounds temporary device memory to one graph update. No
global scheduler, cross-update cache, device ownership in `.hsd`, or renderer
specific path is introduced.
