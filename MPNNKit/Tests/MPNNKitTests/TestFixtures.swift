import XCTest
import MLX
@testable import MPNNKit

/// Repo root = up 4 from MPNNKit/Tests/MPNNKitTests/<thisfile>.swift.
func mpnnRepoRoot(_ file: StaticString = #filePath) -> URL {
    URL(fileURLWithPath: "\(file)")
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
}

func mpnnPackURL() -> URL { mpnnRepoRoot().appendingPathComponent("dist/MPNN.mpnnpack") }

func mpnnInputURL(_ id: String) -> URL {
    mpnnRepoRoot().appendingPathComponent("app/MPNNBench/Resources/app_assets/inputs/\(id).safetensors")
}
func mpnnOracleURL(_ id: String, _ kind: String) -> URL {
    mpnnRepoRoot().appendingPathComponent("app/MPNNBench/Resources/app_assets/oracles/\(id)_\(kind).safetensors")
}

/// Build [Residue] from an input fixture (keys X [1,L,4,3] f32, R_idx i32, chain_labels i32).
func loadResidues(_ id: String) throws -> [MPNNModel.Residue] {
    let a = try loadArrays(url: mpnnInputURL(id))
    let X = a["X"]!.asType(.float32).asArray(Float.self)
    let ridx = a["R_idx"]!.asType(.int32).asArray(Int32.self)
    let chain = a["chain_labels"]!.asType(.int32).asArray(Int32.self)
    let L = ridx.count
    return (0 ..< L).map { i in
        func v(_ atom: Int) -> SIMD3<Float> { let b = (i * 4 + atom) * 3; return SIMD3(X[b], X[b + 1], X[b + 2]) }
        return .init(n: v(0), ca: v(1), c: v(2), o: v(3), chain: Int(chain[i]), resSeq: Int(ridx[i]))
    }
}

/// Native sequence (alphabet indices) from an input fixture (key S_native i32), or nil.
func loadNative(_ id: String) throws -> [Int]? {
    let a = try loadArrays(url: mpnnInputURL(id))
    guard let s = a["S_native"] else { return nil }
    return s.asType(.int32).asArray(Int32.self).map { Int($0) }
}

/// Skip a test unless the pack + a given input fixture are present.
func skipUnlessAssets(_ id: String, _ msg: String = "pack/fixture not present") throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: mpnnPackURL().path)
        && FileManager.default.fileExists(atPath: mpnnInputURL(id).path), msg)
}
