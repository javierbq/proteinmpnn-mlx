import XCTest
import MLX
@testable import MPNNBench

/// Swift MLX design path vs the bundled PyTorch-derived oracle (greedy, fixed order).
/// Runs on the iOS simulator (MLX on CPU); proves the transcription is numerically faithful.
final class DesignParityTests: XCTestCase {
    override func setUp() {
        // load assets from the hosting app bundle (works on simulator + device)
        if let u = Bundle.main.url(forResource: "app_assets", withExtension: nil) {
            AssetsBase.directory = u
        }
    }

    func testDesignGreedyParity() throws {
        let w = try Assets.designWeights()
        let manifest = try Assets.manifest()
        for id in ["6MRR", "5L33"] {
            let meta = manifest.first { $0.id == id }!
            let inp = try Assets.input(meta)
            let ora = try Assets.designOracle(id)

            let (E, eIdx) = featuresDesignE(w, inp.X, inp.mask, inp.Ridx, inp.chainLabels, topK: 32)
            let (hV, hE) = encodeDesign(w, E, eIdx, inp.mask)
            let (S, logits, _) = decodeSequence(
                w, hV, hE, eIdx, inp.Snative, inp.mask, MLXArray.ones([1, inp.L]),
                ora.decodingOrder, mode: .greedy)

            let top1 = mean((S .== ora.top1).asType(.float32)).item(Float.self)
            let dLogits = max(abs(logits - ora.logits)).item(Float.self)
            print("[\(id)] L=\(inp.L) top1=\(top1 * 100)%  |dlogits|=\(dLogits)")
            XCTAssertEqual(top1, 1.0, "\(id): design top-1 must match oracle exactly")
            XCTAssertLessThan(dLogits, 1e-3, "\(id): logits within 1e-3 of oracle")
        }
    }
}
