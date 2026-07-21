import XCTest
import MLX
@testable import MPNNKit

final class RunRewrapTests: XCTestCase {

    func testRunGreedySeed0IsBitStable() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        var opts = MPNNModel.Options(); opts.temperature = 0; opts.seed = 0
        let r = try model.run(residues, options: opts)
        let golden = "MVDEELEKYKKELEAFLKKQGVTNVTIKIENGTLTIEMKGGSEEVKKFLEELEKELKAKGYTVNITIS"
        XCTAssertEqual(r.sequence, golden, "run() greedy+seed0 must be byte-identical after the rewrap")
        XCTAssertNotNil(r.pdb)
        XCTAssertTrue(r.pdb!.contains("ATOM  "))
        XCTAssertEqual(r.pdb!.count, 42420, "PDB byte count must match golden (same sequence → same PDB)")
    }
}
