// MPNNPack.swift — load + validate a `.mpnnpack` model bundle (manifest, weights,
// geometry constants, atom-name table). Layout matches port/build_mpnnpack.py.
import Foundation
import CryptoKit
import MLX

public struct MPNNManifest: Decodable {
    public struct FileEntry: Decodable { public let bytes: Int; public let sha256: String; public let params: Int? }
    public struct Model: Decodable { public let file: String }
    public let format: String
    public let version: Int
    public let min_loader_version: Int
    public let description: String
    public let models: [String: Model]
    public let geometry: String
    public let atom_names: String
    public let files: [String: FileEntry]
}

public enum MPNNPackError: Error, CustomStringConvertible {
    case badFormat(String)
    case unsupportedVersion(Int)
    case missingFile(String)
    case hashMismatch(String)
    case missingModel(String)
    public var description: String {
        switch self {
        case .badFormat(let f): return "not an mpnnpack (format=\(f))"
        case .unsupportedVersion(let v): return "pack needs loader version \(v) > supported \(MPNNPack.loaderVersion)"
        case .missingFile(let f): return "missing file in pack: \(f)"
        case .hashMismatch(let f): return "sha256 mismatch: \(f)"
        case .missingModel(let m): return "manifest missing model '\(m)'"
        }
    }
}

struct MPNNPack {
    static let loaderVersion = 1
    let dir: URL
    let manifest: MPNNManifest

    init(directory: URL, verifyHashes: Bool) throws {
        dir = directory
        let man = try JSONDecoder().decode(
            MPNNManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard man.format == "mpnnpack" else { throw MPNNPackError.badFormat(man.format) }
        guard man.min_loader_version <= Self.loaderVersion else {
            throw MPNNPackError.unsupportedVersion(man.min_loader_version)
        }
        for (rel, e) in man.files {
            let f = directory.appendingPathComponent(rel)
            guard FileManager.default.fileExists(atPath: f.path) else { throw MPNNPackError.missingFile(rel) }
            if verifyHashes, try Self.sha256Hex(of: f) != e.sha256 { throw MPNNPackError.hashMismatch(rel) }
        }
        manifest = man
    }

    func modelFile(_ name: String) throws -> String {
        guard let m = manifest.models[name] else { throw MPNNPackError.missingModel(name) }
        return m.file
    }

    func arrays(_ rel: String) throws -> [String: MLXArray] {
        let a = try loadArrays(url: dir.appendingPathComponent(rel))
        MLX.eval(Array(a.values))       // materialize (avoid lazy mmap views)
        return a
    }

    func data(_ rel: String) throws -> Data { try Data(contentsOf: dir.appendingPathComponent(rel)) }

    static func sha256Hex(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hasher.update(data: chunk) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
