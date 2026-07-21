import XCTest
import MLX
@testable import MPNNKit

final class ScoreAPITests: XCTestCase {
    private func scoreOracle(_ id: String, _ key: String) throws -> MLXArray {
        try loadArrays(url: mpnnOracleURL(id, "score"))[key]!.asType(.float32)
    }
    private func assertClose(_ mine: [[Float]], _ ref: MLXArray, _ tol: Float, _ msg: String) {
        let flat = mine.flatMap { $0 }
        let mineA = MLXArray(flat, ref.shape)
        XCTAssertLessThan(max(abs(mineA - ref)).item(Float.self), tol, msg)
    }

    func testConditionalParity() throws {
        for id in ["6MRR", "5L33"] {
            try skipUnlessAssets(id)
            try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "score").path), "score oracle missing")
            let model = try MPNNModel(packDirectory: mpnnPackURL())
            let residues = try loadResidues(id)
            let native = try XCTUnwrap(loadNative(id))
            let r = try model.score(residues, sequence: native, mode: .conditional, seed: 0)
            assertClose(r.logProbs, try scoreOracle(id, "logprobs_conditional"), 1e-3, "\(id) conditional")
            // currentAALogProb must equal the native-AA column of logProbs.
            let cur = try XCTUnwrap(r.currentAALogProb)
            for i in 0 ..< residues.count { XCTAssertEqual(cur[i], r.logProbs[i][native[i]], accuracy: 1e-6) }
        }
    }

    func testUnconditionalParityAndOrderIndependence() throws {
        let id = "6MRR"
        try skipUnlessAssets(id)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "score").path), "score oracle missing")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues(id)
        let a = try model.score(residues, mode: .unconditional, seed: 1)
        let b = try model.score(residues, mode: .unconditional, seed: 999)
        XCTAssertEqual(a.logProbs.flatMap { $0 }, b.logProbs.flatMap { $0 }, "unconditional must be seed-independent")
        assertClose(a.logProbs, try scoreOracle(id, "logprobs_unconditional"), 1e-3, "\(id) unconditional")
    }

    func testLeaveOneOutParity() throws {
        let id = "6MRR"
        try skipUnlessAssets(id)
        try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnOracleURL(id, "score").path), "score oracle missing")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues(id)
        let native = try XCTUnwrap(loadNative(id))
        let r = try model.score(residues, sequence: native, mode: .leaveOneOut, seed: 0)
        assertClose(r.logProbs, try scoreOracle(id, "logprobs_leaveoneout"), 1e-3, "\(id) leaveOneOut")
    }

    func testConditionalRequiresSequence() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        XCTAssertThrowsError(try model.score(residues, sequence: nil, mode: .conditional)) { err in
            XCTAssertEqual(err as? MPNNModel.MPNNInputError, .sequenceRequired(.conditional))
        }
    }
}
