# Phase 7 Persistent Cache Soak

The soak is opt-in and runs through the Phase 4 CLI benchmark with
`HESIOD_PHASE7_CACHE_SOAK=1`. It performs 100 deterministic graph updates,
edits four different resident-capable node types, changes shape/tiling at
cycles 24, 49, and 74, destroys and recreates the graph at cycle 50, switches
to `MakePeriodic.hsd`, and then reloads the original project. Every update
checks that the terminal output exists, has a non-empty shape, and contains
only finite values. The project switch compares fingerprints of the alternate
and reloaded projects to detect accidental cache reuse.

## 128 MiB budget

Command configuration: 512², 1×1, overlap 0, `HESIOD_METAL_CACHE_MB=128`.

```text
cycles=100 updates=100 project_switches=2
hits=146 misses=2 evictions=0
max_cache_bytes=4194304 cache_budget=134217728
max_peak_resident_bytes=20971520 max_peak_rss_bytes=418152448
output_checks=102 invalid_outputs=0
project_switch_checks=1 project_switch_collisions=0
final_cache_bytes=3145728 final_resident_bytes=0
```

The companion Graph A parity line was `0.00901616 PASS`.

## 1 MiB pressure budget

The same deterministic run with `HESIOD_METAL_CACHE_MB=1` produced:

```text
cycles=100 updates=100 project_switches=2
hits=72 misses=76 evictions=105
max_cache_bytes=1048576 cache_budget=1048576
max_peak_resident_bytes=20971520 max_peak_rss_bytes=492437504
output_checks=102 invalid_outputs=0
project_switch_checks=1 project_switch_collisions=0
final_cache_bytes=1048576 final_resident_bytes=0
```

The companion Graph A parity line was again `0.00901616 PASS`. No stale output,
invalid output, use-after-free symptom, or cross-project fingerprint collision
was observed. Cache bytes stayed at or below the configured budget in both
runs. The cache is still disabled unless
`HESIOD_METAL_PERSISTENT_CACHE=1` is explicitly set.
