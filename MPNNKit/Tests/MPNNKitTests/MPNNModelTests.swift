import XCTest
import MLX
@testable import MPNNKit

/// End-to-end smoke test on real Metal: load dist/MPNN.mpnnpack + a backbone (6MRR from the
/// benchmark fixtures) and run design + repack through the public API. Skips if assets absent.
final class MPNNModelTests: XCTestCase {
    // repo root = <root>/MPNNKit/Tests/MPNNKitTests/MPNNModelTests.swift  -> up 4
    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    func testDesignRepackFrom6MRR() throws {
        let pack = repoRoot.appendingPathComponent("dist/MPNN.mpnnpack")
        let fixture = repoRoot.appendingPathComponent("app/MPNNBench/Resources/app_assets/inputs/6MRR.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: pack.path)
            && FileManager.default.fileExists(atPath: fixture.path), "pack/fixture not present")

        // build residues from the 6MRR backbone fixture (X [L,4,3], R_idx, chain_labels)
        let a = try loadArrays(url: fixture)
        let X = a["X"]!.asType(.float32).asArray(Float.self)      // L*4*3
        let ridx = a["R_idx"]!.asType(.int32).asArray(Int32.self)
        let chain = a["chain_labels"]!.asType(.int32).asArray(Int32.self)
        let L = ridx.count
        var residues: [MPNNModel.Residue] = []
        for i in 0 ..< L {
            func v(_ atom: Int) -> SIMD3<Float> {
                let b = (i * 4 + atom) * 3
                return SIMD3(X[b], X[b + 1], X[b + 2])
            }
            residues.append(.init(n: v(0), ca: v(1), c: v(2), o: v(3),
                                  chain: Int(chain[i]), resSeq: Int(ridx[i])))
        }

        let model = try MPNNModel(packDirectory: pack)
        var opts = MPNNModel.Options(); opts.temperature = 0; opts.seed = 0   // greedy, deterministic
        let r = try model.run(residues, options: opts)

        XCTAssertEqual(r.sequence.count, L, "one designed residue per input")
        XCTAssertFalse(r.sequence.contains("X"), "no unknown residues")
        XCTAssertNotNil(r.pdb)
        XCTAssertTrue(r.pdb!.contains("ATOM  "), "PDB has ATOM records")
        XCTAssertGreaterThan(r.designMs, 0)
        print("[MPNNKit] 6MRR L=\(L) design=\(r.designMs)ms repack=\(r.repackMs)ms seq[:40]=\(r.sequence.prefix(40))")
    }
}
