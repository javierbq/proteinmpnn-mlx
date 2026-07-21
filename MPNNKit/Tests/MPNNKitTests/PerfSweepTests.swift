// PerfSweepTests.swift — Tier-3 L-sweep performance harness.
//
// GATE: this test is skipped unless the environment variable MPNN_PERF=1 is set.
// Normal `swift test` will skip it, keeping the 21-test suite green.
//
// To run:
//   MPNN_PERF=1 swift test -c release --filter PerfSweep
//
// Machine: Apple M3 Pro, macOS (Darwin 25.5 / Sequoia).
// Build config: -c release (MLX Metal kernels compiled with optimisations).
// Methodology: 1 untimed warm-up per fixture (triggers Metal shader cache),
//              then nIter=5 timed iterations; report median + min.
//              MLX.eval() is called inside every timed block to force GPU evaluation
//              before the clock stops — without this, MLX only builds the lazy graph.

import XCTest
import MLX
@testable import MPNNKit

final class PerfSweepTests: XCTestCase {

    // ---- Fixtures in ascending L order (all 9) ----
    static let fixtures: [(id: String, L: Int)] = [
        ("6MRR",      68),
        ("5L33",     106),
        ("4GYT",     354),
        ("3HTN",     425),
        ("4YOW",     681),
        ("6EHB",     955),
        ("synth1272", 1272),
        ("synth1590", 1590),
        ("synth2120", 2120),
    ]

    // ---- Config ----
    static let nIter = 5
    // Selection-size comparison: run design_10pct at these L values
    static let largeL: Set<Int> = [955, 2120]

    // ---- Helpers ----

    /// Median of a Double array.
    func med(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        let n = s.count
        return n % 2 == 1 ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2
    }

    /// Time a throwing block in milliseconds (wall time via DispatchTime).
    @discardableResult
    func timeMs(_ block: () throws -> Void) rethrows -> Double {
        let t0 = DispatchTime.now().uptimeNanoseconds
        try block()
        return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    }

    // ---- Main sweep ----

    func testLSweep() throws {
        // GATE — skip immediately unless explicitly enabled
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MPNN_PERF"] == "1",
            "perf sweep disabled; set MPNN_PERF=1 to enable"
        )
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: mpnnPackURL().path),
            "MPNN.mpnnpack not present — run `make dist` first"
        )

        // Load model ONCE outside all timing loops.
        // Model loading is O(weights size) and not part of any timed region.
        print("\nMPNN_PERF: loading model …")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        print("MPNN_PERF: model loaded. Starting L-sweep (nIter=\(Self.nIter)).\n")

        // CSV header — printed once, prefixed so it's grep-able from test logs.
        print(
            "PERF_CSV: " +
            "id,L," +
            "encode_med_ms,encode_min_ms," +
            "score_total_med_ms,score_decode_med_ms," +
            "design_full_med_ms,design_full_min_ms,design_decode_full_med_ms," +
            "design_10pct_med_ms,design_decode_10pct_med_ms," +
            "peak_active_MB"
        )

        for fix in Self.fixtures {
            let id = fix.id
            let L = fix.L

            guard FileManager.default.fileExists(atPath: mpnnInputURL(id).path) else {
                print("PERF_SKIP: \(id) fixture not found")
                continue
            }
            print("PERF_PROGRESS: \(id) L=\(L) …")

            let residues = try loadResidues(id)
            let native = try loadNative(id) ?? [Int](repeating: 0, count: L)

            // ---- 1 untimed warm-up ----
            // Triggers Metal shader compilation + caching so timed runs reflect
            // steady-state performance, not first-run JIT overhead.
            var wopts = MPNNModel.DesignOptions(); wopts.temperature = 0.1; wopts.seed = 42
            _ = try model.design(residues, options: wopts)

            // ---- (a) Encode-only timing ----
            // encode = modelInputs (featurization) + featuresDesignE (RBF edges + kNN)
            //        + encodeDesign (3 encoder layers + 2 context layers)
            // MLX.eval(hV, hE, eIdx) forces the Metal compute graph to completion
            // before the clock stops; without this, MLX returns immediately after
            // building the lazy graph and the time would be ~0.
            var encodeTimes: [Double] = []
            for _ in 0 ..< Self.nIter {
                let ms = timeMs {
                    let (X, mask, Ridx, chainLabels) = model.modelInputs(residues)
                    let (E, eIdx) = featuresDesignE(
                        model.designWeights, X, mask, Ridx, chainLabels, topK: 32)
                    let (hV, hE) = encodeDesign(model.designWeights, E, eIdx, mask)
                    // Force GPU evaluation — mandatory for trustworthy timing
                    MLX.eval(hV, hE, eIdx)
                }
                encodeTimes.append(ms)
            }

            // ---- (b) Score total timing (.conditional, teacher-forced) ----
            // score() internally calls MLX.eval(logp) before returning, so timing
            // the full call captures encode + decode (O(L) autoregressive steps).
            // We use seed: 0 for reproducibility (.conditional is RNG-free regardless).
            var scoreTotalTimes: [Double] = []
            for _ in 0 ..< Self.nIter {
                let ms = try timeMs {
                    _ = try model.score(residues, sequence: native, mode: .conditional, seed: 0)
                }
                scoreTotalTimes.append(ms)
            }

            // ---- (c) Design full timing (all L positions free) ----
            // design() calls MLX.eval(S, logits) inside decodeSequence (once per
            // position in the autoregressive loop) and again before returning.
            var designFullTimes: [Double] = []
            for _ in 0 ..< Self.nIter {
                var opts = MPNNModel.DesignOptions(); opts.temperature = 0.1; opts.seed = nil
                let ms = try timeMs {
                    _ = try model.design(residues, options: opts)
                }
                designFullTimes.append(ms)
            }

            // ---- (d) Design 10% timing (90% fixed, 10% free) — large L only ----
            // Key claim: encode is L-dominated and IDENTICAL to (c) regardless of
            // fixedPositions size, because the encoder always sees the full protein.
            // Note: decodeSequence iterates ALL L positions even with fixedPositions;
            // fixed positions assign SnT (native) instead of sampling but still run
            // the full attention + MLX.eval per step. So decode time is similar.
            var design10PctTimes: [Double] = []
            if Self.largeL.contains(L) {
                let nFree = max(1, L / 10)
                let fixedPos = Set(nFree ..< L)  // 90% fixed
                var opts = MPNNModel.DesignOptions()
                opts.temperature = 0.1; opts.seed = nil
                opts.fixedPositions = fixedPos
                opts.nativeSequence = native
                for _ in 0 ..< Self.nIter {
                    let ms = try timeMs {
                        _ = try model.design(residues, options: opts)
                    }
                    design10PctTimes.append(ms)
                }
            }

            // ---- (e) Peak GPU active memory (best-effort) ----
            // Reset peak counter, run one design pass, then snapshot.
            // Memory.peakMemory setter calls mlx_reset_peak_memory() (ignores newValue).
            Memory.peakMemory = 0
            var peakOpts = MPNNModel.DesignOptions(); peakOpts.temperature = 0.1; peakOpts.seed = nil
            _ = try model.design(residues, options: peakOpts)
            let peakBytes = Memory.snapshot().peakMemory
            let peakMB = Double(peakBytes) / (1024 * 1024)

            // ---- Derived metrics ----
            let encMed = med(encodeTimes)
            let encMin = encodeTimes.min()!

            let scoreTotalMed = med(scoreTotalTimes)
            let scoreDecodeMed = scoreTotalMed - encMed

            let designFullMed = med(designFullTimes)
            let designFullMin = designFullTimes.min()!
            let designDecodeFullMed = designFullMed - encMed

            let design10PctMed = design10PctTimes.isEmpty ? -1.0 : med(design10PctTimes)
            let designDecode10PctMed = design10PctTimes.isEmpty
                ? -1.0
                : design10PctMed - encMed

            let tenStr  = design10PctTimes.isEmpty ? "n/a" : String(format: "%.0f", design10PctMed)
            let tenDStr = design10PctTimes.isEmpty ? "n/a" : String(format: "%.0f", designDecode10PctMed)

            print(String(
                format: "PERF_CSV: %@,%d,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%@,%@,%.0f",
                id, L,
                encMed, encMin,
                scoreTotalMed, scoreDecodeMed,
                designFullMed, designFullMin, designDecodeFullMed,
                tenStr, tenDStr,
                peakMB
            ))
        }

        print("\nMPNN_PERF: sweep complete.")
    }
}
