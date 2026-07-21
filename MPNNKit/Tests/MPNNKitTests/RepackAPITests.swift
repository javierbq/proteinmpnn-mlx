import XCTest
import MLX
@testable import MPNNKit

final class RepackAPITests: XCTestCase {
    func testRepackMatchesOracleSideChains() throws {
        let id = "6MRR"
        try skipUnlessAssets(id)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "repack").path), "repack oracle missing")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues(id)
        let L = residues.count
        // The repack oracle was generated from the MLX-designed sequence (design_top1),
        // not the native PDB sequence. Load that sequence from the design oracle.
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "design").path), "design oracle missing")
        let designOra = try loadArrays(url: mpnnOracleURL(id, "design"))
        let sdes = designOra["design_top1"]!.asType(.int32).asArray(Int32.self).map { Int($0) }
        let out = try model.repack(residues, sequence: sdes)
        XCTAssertTrue(out.pdb.contains("ATOM  "))
        XCTAssertEqual(out.atomConfidence.count, L)
        XCTAssertEqual(out.atomConfidence.first?.count, 14)

        // Side-chain (atom14 idx 4..13) RMSD vs oracle atom14, masked.
        let ora = try loadArrays(url: mpnnOracleURL(id, "repack"))
        let refA14 = ora["atom14"]!.asType(.float32).expandedDimensions(axis: 0)    // [L,14,3] -> [1,L,14,3]
        let refMask = ora["atom14_mask"]!.asType(.float32).expandedDimensions(axis: 0) // [L,14] -> [1,L,14]
        let mineA14 = model.repackAtom14(residues, indices: sdes)   // test-only helper (Step 3)
        let scSlice = 4 ..< 14
        let diff = (mineA14[0..., 0..., scSlice] - refA14[0..., 0..., scSlice])
        let m = refMask[0..., 0..., scSlice].expandedDimensions(axis: -1)
        let sq = (diff * diff * m)
        let rmsd = sqrt((sum(sq) / maximum(sum(m) * 3, MLXArray(1))).item(Float.self))
        XCTAssertLessThan(rmsd, 1e-2, "side-chain RMSD within 1e-2 Å of oracle")
    }
}
