import XCTest
import MLX

final class SmokeTests: XCTestCase {
    /// Proves that mlx-swift links and executes: build a small MLXArray,
    /// reduce it, and pull the scalar result back to the CPU.
    func testMLXSumRuns() {
        let a = MLXArray([1.0, 2.0, 3.0])
        let s = a.sum().item(Float.self)
        XCTAssertEqual(s, 6.0, accuracy: 1e-6)
    }
}
