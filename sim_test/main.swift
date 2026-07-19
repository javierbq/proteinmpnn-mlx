// ProteinMPNN unconditional_probs — Core ML parity + latency test, run in the iOS Simulator.
// Usage: <exe> <model.mlmodelc dir> <fixtures dir>
import Foundation
import CoreML

func die(_ m: String) -> Never { FileHandle.standardError.write((m + "\n").data(using: .utf8)!); exit(1) }

let args = CommandLine.arguments
guard args.count >= 3 else { die("usage: sim_test <model.mlmodelc> <fixtures>") }
let modelURL = URL(fileURLWithPath: args[1])
let fx = args[2]

func readFloats(_ name: String) -> [Float] {
    guard let d = FileManager.default.contents(atPath: "\(fx)/\(name)") else { die("missing \(name)") }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
func readInt32(_ name: String) -> [Int32] {
    guard let d = FileManager.default.contents(atPath: "\(fx)/\(name)") else { die("missing \(name)") }
    return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
}

// shapes.json -> {B, L, A}
struct Shapes: Codable { let B: Int; let L: Int; let A: Int }
let shp = try! JSONDecoder().decode(Shapes.self,
            from: FileManager.default.contents(atPath: "\(fx)/shapes.json")!)
let (B, L, A) = (shp.B, shp.L, shp.A)

func floatArr(_ vals: [Float], _ shape: [Int]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    let p = a.dataPointer.bindMemory(to: Float.self, capacity: vals.count)
    for i in 0..<vals.count { p[i] = vals[i] }
    return a
}
func int32Arr(_ vals: [Int32], _ shape: [Int]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
    let p = a.dataPointer.bindMemory(to: Int32.self, capacity: vals.count)
    for i in 0..<vals.count { p[i] = vals[i] }
    return a
}

let X    = floatArr(readFloats("X.bin"), [B, L, 4, 3])
let mask = floatArr(readFloats("mask.bin"), [B, L])
let ridx = int32Arr(readInt32("residue_idx.bin"), [B, L])
let cenc = int32Arr(readInt32("chain_encoding_all.bin"), [B, L])
let ref  = readFloats("ref_logp.bin")   // [B*L*A]

let cfg = MLModelConfiguration()
cfg.computeUnits = .all
let model: MLModel
do { model = try MLModel(contentsOf: modelURL, configuration: cfg) }
catch { die("model load failed: \(error)") }

let feats = try! MLDictionaryFeatureProvider(dictionary: [
    "X": X, "mask": mask, "residue_idx": ridx, "chain_encoding_all": cenc])

// warm-up + correctness
let out = try! model.prediction(from: feats)
guard let logp = out.featureValue(for: "log_softmax")?.multiArrayValue else { die("no output") }
var maxDiff: Float = 0, agree = 0, total = 0
let lp = logp.dataPointer.bindMemory(to: Float.self, capacity: ref.count)
for pos in 0..<(B*L) {
    var bestI = 0, bestR = 0; var bv = -Float.infinity, br = -Float.infinity
    for a in 0..<A {
        let idx = pos*A + a
        maxDiff = max(maxDiff, abs(lp[idx] - ref[idx]))
        if lp[idx] > bv { bv = lp[idx]; bestI = a }
        if ref[idx] > br { br = ref[idx]; bestR = a }
    }
    total += 1; if bestI == bestR { agree += 1 }
}

// latency
let N = 50
let t0 = Date()
for _ in 0..<N { _ = try! model.prediction(from: feats) }
let ms = Date().timeIntervalSince(t0) / Double(N) * 1000.0

print("[sim] iOS Simulator Core ML run")
print(String(format: "[sim] L=%d  max|diff| vs fp32 oracle = %.3e", L, maxDiff))
print(String(format: "[sim] top-1 AA agreement = %.4f", Double(agree)/Double(total)))
print(String(format: "[sim] mean latency over %d runs = %.2f ms  (Simulator: CPU/GPU, no ANE)", N, ms))
print(maxDiff < 5e-2 ? "[sim] PASS" : "[sim] CHECK")
