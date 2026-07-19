// Assets.swift — load bundled inputs / oracles / weights (safetensors + manifest.json).
// AssetsBase.directory points at the bundled `app_assets` folder in the app; dev-time
// tests override it with the on-disk source directory.
import Foundation
import MLX

enum AssetsBase {
    nonisolated(unsafe) static var directory: URL = {
        if let u = Bundle.main.url(forResource: "app_assets", withExtension: nil) { return u }
        return URL(fileURLWithPath: "app_assets")
    }()
}

struct ManifestEntry: Codable, Identifiable {
    let id: String
    let L: Int
    let n_chains: Int
    let synthetic: Bool
    let native_seq: String
}

struct ProteinInput {
    let meta: ManifestEntry
    let X: MLXArray            // [1,L,4,3]
    let Ridx: MLXArray         // [1,L] f32
    let chainLabels: MLXArray  // [1,L] f32
    let Snative: MLXArray      // [1,L] i32
    let mask: MLXArray         // [1,L] f32 (all ones — clean structures)
    var L: Int { meta.L }
}

struct DesignOracle {
    let decodingOrder: MLXArray  // [1,L] i32
    let top1: MLXArray           // [1,L] i32
    let logits: MLXArray         // [1,L,21] f32
    let eIdx: MLXArray?          // [1,L,K] i32 (kNN graph, for tie-break-free parity)
}

struct RepackOracle {
    let atom14: MLXArray         // [1,L,14,3] f32
    let atom14Mask: MLXArray     // [1,L,14] f32
    let bFactors: MLXArray       // [1,L,14] f32
}

enum Assets {
    static func manifest() throws -> [ManifestEntry] {
        let url = AssetsBase.directory.appendingPathComponent("manifest.json")
        return try JSONDecoder().decode([ManifestEntry].self, from: Data(contentsOf: url))
    }

    static func arrays(_ rel: String) throws -> [String: MLXArray] {
        let url = AssetsBase.directory.appendingPathComponent(rel)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(domain: "MPNNBench", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "missing asset: \(url.path)"])
        }
        let arr = try loadArrays(url: url)
        MLX.eval(Array(arr.values))     // materialize (avoid lazy mmap views)
        return arr
    }

    static func designWeights() throws -> Weights { try arrays("weights/design.safetensors") }
    static func packerWeights() throws -> Weights { try arrays("weights/packer.safetensors") }

    static func geometry() throws -> [String: MLXArray] { try arrays("geometry.safetensors") }

    static func input(_ meta: ManifestEntry) throws -> ProteinInput {
        let a = try arrays("inputs/\(meta.id).safetensors")
        let b = { (x: MLXArray) in x.expandedDimensions(axis: 0) }
        let L = meta.L
        return ProteinInput(
            meta: meta,
            X: b(a["X"]!),
            Ridx: b(a["R_idx"]!.asType(.float32)),
            chainLabels: b(a["chain_labels"]!.asType(.float32)),
            Snative: b(a["S_native"]!.asType(.int32)),
            mask: MLXArray.ones([1, L]))
    }

    static func designOracle(_ id: String) throws -> DesignOracle {
        let a = try arrays("oracles/\(id)_design.safetensors")
        let b = { (x: MLXArray) in x.expandedDimensions(axis: 0) }
        return DesignOracle(decodingOrder: b(a["decoding_order"]!.asType(.int32)),
                            top1: b(a["design_top1"]!.asType(.int32)),
                            logits: b(a["design_logits"]!),
                            eIdx: a["E_idx"].map { b($0.asType(.int32)) })
    }

    static func repackOracle(_ id: String) throws -> RepackOracle {
        let a = try arrays("oracles/\(id)_repack.safetensors")
        let b = { (x: MLXArray) in x.expandedDimensions(axis: 0) }
        return RepackOracle(atom14: b(a["atom14"]!), atom14Mask: b(a["atom14_mask"]!),
                            bFactors: b(a["b_factors"]!))
    }
}
