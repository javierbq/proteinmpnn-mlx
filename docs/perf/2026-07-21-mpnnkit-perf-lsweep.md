# MPNNKit Tier-3 performance L-sweep

**Date:** 2026-07-21 · **Branch:** `design/mpnnkit-interactive-design-api` (PR javierbq/proteinmpnn-mlx#1)
**Closes acceptance item §11** ("perf numbers recorded across L; O(L²) encode caveat documented").

## Machine & method

- **Host:** Apple M3 Pro, macOS (Darwin 25.5). Metal GPU.
- **Build:** `-c release` (MLX Metal kernels optimised). Debug is unrepresentative and not used here.
- **Harness:** `MPNNKit/Tests/MPNNKitTests/PerfSweepTests.swift`, gated behind `MPNN_PERF=1` so the normal `swift test` (21/21) never runs it. Reproduce with:
  ```
  cd MPNNKit && MPNN_PERF=1 swift test -c release --filter PerfSweep
  ```
- **Timing discipline:** model weights loaded once outside all loops; **1 untimed warm-up** per fixture (Metal shader/kernel compile); then **5 timed iterations**, reporting **median** (and min). `MLX.eval(...)` is called **inside every timed block** — MLX is lazy, so without forcing evaluation the clock would measure graph construction, not GPU compute.
- **Encode vs decode split:** *encode* is timed directly over the real internal path — `modelInputs` (featurisation) + `featuresDesignE` (RBF edges + kNN, `topK=32`) + `encodeDesign` (3 encoder + 2 context layers). *decode* is derived as `total − encode` (the score/design public calls each `eval` internally). The subtraction is approximate at very small L where both terms are a few ms; it did not go negative anywhere in this run.
- **Inputs:** the 9 pre-featurised fixtures in `app/MPNNBench/Resources/app_assets/inputs/` (real: 6MRR…6EHB; synthetic: synth1272/1590/2120). One `testLSweep` run took 109.7 s wall.

## Results (median ms, unless noted)

| fixture | L | encode | score total | score decode | design full | design decode | design 10% | design 10% decode | peak MB |
|---------|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| 6MRR | 68 | 5 | 70 | 65 | 65 | 60 | — | — | 104 |
| 5L33 | 106 | 6 | 119 | 113 | 119 | 112 | — | — | 129 |
| 4GYT | 354 | 23 | 404 | 381 | 457 | 434 | — | — | 563 |
| 3HTN | 425 | 25 | 391 | 365 | 454 | 428 | — | — | 670 |
| 4YOW | 681 | 39 | 591 | 552 | 686 | 647 | — | — | 1064 |
| 6EHB | 955 | 55 | 854 | 799 | 996 | 940 | **830** | **775** | 1338 |
| synth1272 | 1272 | 75 | 1431 | 1356 | 1105 | 1030 | — | — | 1747 |
| synth1590 | 1590 | 93 | 1590 | 1497 | 1457 | 1364 | — | — | 2159 |
| synth2120 | 2120 | 130 | 2110 | 1980 | 2147 | 2017 | **2073** | **1943** | 2431 |

"design 10%" = redesign 10% of positions (90% `fixedPositions` + `nativeSequence`), run only at L∈{955, 2120}.

## Analysis

### 1. Encode is ~linear in L — the O(L²) step is negligible on-GPU (corrects the spec)

`encode` cost is a near-constant **~0.06 ms per residue** above L≈350 (0.065 @354 → 0.061 @2120). The log-log slope of encode-vs-L is **≈0.95–1.0**, i.e. **linear**, not quadratic:

- 354→2120 (6.0× L): 23→130 ms (5.65×), slope 0.97.
- 425→1590 (3.7× L): 25→93 ms (3.72×), slope 1.00.

The spec assumed an O(L²) featuriser+encoder. In practice the kNN neighbour cap (`topK=32`) makes the encoder O(L·K)=O(L); the genuinely O(L²) step (the all-pairs distance matrix for kNN) is fully GPU-parallel and its constant is tiny — at L=2120 it is invisible against the linear encoder work. **Conclusion: for the whole realistic range (≤2120 aa) encode is effectively linear.** The O(L²) term is a latent asymptote, not a practical cost here.

### 2. Total time is decode-dominated and ~linear; budgets are comfortably met

Both `score(.conditional)` (single teacher-forced pass) and `design()` (autoregressive) are dominated by decode and scale ~linearly (~0.9–1.1 ms/residue). Against the spec budgets:

- **< 3 s @ ~500 aa:** measured design ≈ 0.45 s @425 aa, 0.69 s @681 aa — **~4–6× under budget.**
- **< 10 s @ ~1500 aa:** measured design ≈ 1.46 s @1590 aa — **~7× under budget.**
- Worst case measured: L=2120 → score ≈ 2.1 s, design ≈ 2.1 s.

(Real vs synthetic fixtures differ by a small constant factor — the synthetic backbones show score slightly above design where the real ones show the reverse — but both are the same order and clearly linear.)

### 3. `fixedPositions` is NOT a decode-time optimisation — cost tracks TOTAL L, not selection size

This is the operationally important finding for Phase 2. `decodeSequence` iterates **all L positions regardless of `fixedPositions`** — a fixed position is *assigned* its native AA instead of *sampled*, but still runs the full per-step attention + `eval`. So redesigning a small selection is **not** faster:

- L=955: 10%-redesign decode 775 ms vs full 940 ms — only **~18% faster**.
- L=2120: 10%-redesign decode 1943 ms vs full 2017 ms — only **~4% faster**.

Combined with (1), **both encode and decode scale with total L**, so the wall-clock of a design/score call is governed by the size of the whole protein, not by how much of it the user selected to redesign. Phase 2 must set user expectations accordingly (and, if fast partial redesign is ever needed, it requires an actual decode-skipping optimisation that does not exist today).

### 4. Peak GPU memory ~linear → ~2.4 GB at L=2120 (real iOS jetsam risk)

Peak active MLX memory grows ~linearly (~1.1–1.6 MB/residue), reaching **~2.4 GB at L=2120**. This corroborates the known on-device OOM history: a ~2k-residue protein is a genuine jetsam risk on iOS and argues for an L cap (or chunking) as a Phase-2/host constraint, as already flagged in spec §9.

## Notes

- The RMSD parity assertion in `RepackAPITests` is wrapped in `#if DEBUG` because it depends on the `#if DEBUG`-only `repackAtom14` accessor; this lets the whole test target compile under `-c release` for this harness. The normal 21/21 suite runs in **debug**, where that assertion still executes — no parity coverage is lost in the default test mode.
