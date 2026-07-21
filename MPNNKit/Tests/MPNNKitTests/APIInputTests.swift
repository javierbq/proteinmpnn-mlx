import XCTest
import MLX
@testable import MPNNKit

final class APIInputTests: XCTestCase {
    func testAlphabet() {
        XCTAssertEqual(String(MPNNModel.alphabet), "ACDEFGHIKLMNPQRSTVWYX")
        XCTAssertEqual(MPNNModel.alphabet.count, 21)
        XCTAssertEqual(MPNNModel.alphabet[20], "X")
    }

    func testModelInputsShapes() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let residues = try loadResidues("6MRR")
        let inp = model.modelInputs(residues)
        XCTAssertEqual(inp.X.shape, [1, residues.count, 4, 3])
        XCTAssertEqual(inp.mask.shape, [1, residues.count])
        XCTAssertEqual(inp.Ridx.shape, [1, residues.count])
        XCTAssertEqual(inp.chainLabels.shape, [1, residues.count])
    }

    func testBuildLogitBiasNilWhenBothNil() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        XCTAssertNil(model.buildLogitBias(L: 10, bias: nil, omit: nil))
    }

    func testBuildLogitBiasShapeAndOmit() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let b = model.buildLogitBias(L: 3, bias: nil, omit: [[], [5], []])
        XCTAssertEqual(b?.shape, [1, 3, 21])
        let flat = b![0].asType(.float32).asArray(Float.self)
        XCTAssertLessThan(flat[1 * 21 + 5], -1e8)   // omitted AA gets a large negative
        XCTAssertEqual(flat[0], 0)                    // untouched entry stays 0
    }

    func testToRows() throws {
        try skipUnlessAssets("6MRR")
        let model = try MPNNModel(packDirectory: mpnnPackURL())
        let a = MLXArray([1, 2, 3, 4, 5, 6].map { Float($0) }, [2, 3])
        let rows = model.toRows(a, rows: 2, cols: 3)
        XCTAssertEqual(rows, [[1, 2, 3], [4, 5, 6]])
    }
}
