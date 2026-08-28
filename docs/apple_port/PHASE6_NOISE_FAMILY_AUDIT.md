# Phase 6 Noise-Family Audit

## Scope and ranking

The corpus scan found 23 explicit `NoiseFbm` graph nodes. The broader
`CoherentNoise` implementation exposes FBM, Ridged, IQ, Jordan, Parberry,
PingPong, and Swiss groups. The current Metal backend's resident FBM evaluator
supports Perlin, Perlin billow, Perlin half, Simplex2, Value, and Value
linear. `VALUE_CUBIC` and `WORLEY` are present in the HighMap enum and HSD
mapping but not in the staged Metal evaluator.

The ranking heuristic used for this phase is transparent and deliberately
simple:

```text
score = 4 * graph-frequency + 3 * direct-built-in-example +
        2 * host-boundary-impact + 1 * resident-API-readiness
```

`GaborWaveFbm` wins on graph frequency and direct example coverage. The
NoiseFbm families are the next audit target, but they are not ported in this
phase because the brief limits Phase 6 to one major family plus small
supporting operations, and Gabor is the stronger measured boundary.

## Findings

| Family/configuration | Metal status | Phase 6 decision |
|---|---|---|
| CoherentNoise FBM, supported enum | resident from Phase 5 | retain and measure |
| CoherentNoise FBM, Value cubic/Worley | fallback | retain until evaluator parity exists |
| Ridged/IQ/Jordan/Parberry/PingPong/Swiss | fallback | no speculative port |
| Standalone NoiseFbm node | fallback | retain explicit host boundary |
| GaborWaveFbm | added resident operation | selected Phase 6 family |

The missing Value cubic and Worley paths are not harmless enum aliases: their
lattice/cell-distance semantics differ from the existing evaluator and need
their own parity tests. Porting them without that evidence would broaden the
surface area while weakening the fallback contract.

## No-Metal behavior

No noise-family change removes the OpenCL/CPU implementation. The Metal API
additions have unavailable stubs, and the no-Metal HighMap/Hesiod builds remain
the validation path for this decision.
