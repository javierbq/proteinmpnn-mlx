import XCTest
import MLX
@testable import MPNNKit

final class DesignAPITests: XCTestCase {
    // Load a design oracle. Files store without batch dim: decoding_order [L] i32,
    // design_top1 [L] i32, design_logits [L,21] f32. Expand to [1,L] / [1,L,21].
    private func designOracle(_ id: String) throws -> (order: MLXArray, top1: MLXArray, logits: MLXArray) {
        let a = try loadArrays(url: mpnnOracleURL(id, "design"))
        let order  = a["decoding_order"]!.asType(.int32).expandedDimensions(axis: 0)   // [L]    -> [1,L]
        let top1   = a["design_top1"]!.asType(.int32).expandedDimensions(axis: 0)      // [L]    -> [1,L]
        let logits = a["design_logits"]!.asType(.float32).expandedDimensions(axis: 0)  // [L,21] -> [1,L,21]
        return (order, top1, logits)
    }

    func testDecodeLogitBiasNilMatchesDesignOracle() throws {
        for id in ["6MRR", "5L33"] {
            try skipUnlessAssets(id)
            try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "design").path), "design oracle missing")
            let model = try MPNNModel(packDirectory: mpnnPackURL())
            let residues = try loadResidues(id)
            let L = residues.count
            let inp = model.modelInputs(residues)
            let ora = try designOracle(id)
            let (E, eIdx) = featuresDesignE(model.designWeights, inp.X, inp.mask, inp.Ridx, inp.chainLabels, topK: 32)
            let (hV, hE) = encodeDesign(model.designWeights, E, eIdx, inp.mask)
            let Snative = MLXArray.zeros([1, L]).asType(.int32)
            let chainMask = MLXArray.ones([1, L])
            // logitBias defaults to nil → must reproduce the oracle exactly.
            let (S, logits, _) = decodeSequence(model.designWeights, hV, hE, eIdx, Snative, inp.mask, chainMask,
                                                ora.order, mode: .greedy)
            let top1 = mean((S .== ora.top1).asType(.float32)).item(Float.self)
            let dLogits = max(abs(logits - ora.logits)).item(Float.self)
            XCTAssertEqual(top1, 1.0, "\(id): design top-1 must match oracle exactly")
            XCTAssertLessThan(dLogits, 1e-3, "\(id): logits within 1e-3 of oracle")
        }
    }
}
