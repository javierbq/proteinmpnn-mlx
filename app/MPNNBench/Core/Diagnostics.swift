// Diagnostics.swift — on-device parity self-test. Prints "PARITY|" lines to stderr
// (readable via `simctl launch --console` or the macOS mpnncheck tool).
import Foundation
import MLX

@inline(__always) func plog(_ s: String) {
    FileHandle.standardError.write(Data(("PARITY| " + s + "\n").utf8))
}

enum Diagnostics {
    static func setup() {
        if let u = Bundle.main.url(forResource: "app_assets", withExtension: nil) {
            AssetsBase.directory = u
        }
    }

    static func runDesignParity(_ ids: [String] = ["6MRR", "5L33"]) {
        setup()
        do {
            let w = try Assets.designWeights()
            let manifest = try Assets.manifest()
            for id in ids {
                guard let meta = manifest.first(where: { $0.id == id }) else { continue }
                let inp = try Assets.input(meta)
                let ora = try Assets.designOracle(id)
                let (E, eIdx) = featuresDesignE(w, inp.X, inp.mask, inp.Ridx, inp.chainLabels, topK: 32)
                let (hV, hE) = encodeDesign(w, E, eIdx, inp.mask)
                let (S, logits, _) = decodeSequence(
                    w, hV, hE, eIdx, inp.Snative, inp.mask, MLXArray.ones([1, inp.L]),
                    ora.decodingOrder, mode: .greedy)
                let mism = Int(sum((S .!= ora.top1).asType(.int32)).item(Int32.self))
                let top1 = Double(inp.L - mism) / Double(inp.L) * 100
                let dLog = max(abs(logits - ora.logits)).item(Float.self)
                let pass = mism == 0 && dLog < 1e-3
                plog("design \(id) L=\(inp.L) top1=\(top1)% mismatches=\(mism) dlogits=\(dLog) \(pass ? "PASS" : "FAIL")")
            }
            plog("DESIGN COMPLETE")
        } catch {
            plog("ERROR \(error)")
        }
    }

    static func runRepackParity(_ ids: [String] = ["6MRR", "5L33"]) {
        setup()
        do {
            let w = try Assets.packerWeights()
            let gc = RepackConstants(try Assets.geometry())
            let manifest = try Assets.manifest()
            for id in ids {
                guard let meta = manifest.first(where: { $0.id == id }) else { continue }
                let inp = try Assets.input(meta)
                let ora = try Assets.repackOracle(id)
                let Sdes = try Assets.designOracle(id).top1   // repack runs on the designed seq
                let r = repackFull(w, gc, Xbb: inp.X, S: Sdes, mask: inp.mask,
                                   Ridx: inp.Ridx, chainLabels: inp.chainLabels)
                let scMask = ora.atom14Mask[0..., 0..., 4...].expandedDimensions(axis: -1)
                let diff = (r.atom14[0..., 0..., 4...] - ora.atom14[0..., 0..., 4...]) * scMask
                let rmsd = Foundation.sqrt(sum(square(diff)).item(Float.self) / (sum(scMask).item(Float.self) * 3))
                let maxd = max(abs(diff)).item(Float.self)
                plog("repack \(id) L=\(inp.L) sc_rmsd=\(rmsd) maxd=\(maxd) \(rmsd < 1e-2 ? "PASS" : "FAIL")")
            }
            plog("REPACK COMPLETE")
        } catch {
            plog("ERROR \(error)")
        }
    }

    static func runBenchmark(_ ids: [String], docsDir: URL) {
        setup()
        do {
            let wD = try Assets.designWeights()
            let wP = try Assets.packerWeights()
            let gc = RepackConstants(try Assets.geometry())
            let names = try PDBWriter.loadNames()
            let manifest = try Assets.manifest()
            Bench.warmup(wD)
            var results: [BenchResult] = []
            for id in ids {
                guard let meta = manifest.first(where: { $0.id == id }) else { continue }
                let r = Bench.run(meta, wD: wD, wP: wP, gc: gc, names: names, docsDir: docsDir)
                results.append(r)
                plog("bench \(r.id) L=\(r.L) feat=\(rnd(r.featurizeMs)) enc=\(rnd(r.encodeMs)) design=\(rnd(r.designMs)) repack=\(rnd(r.repackMs)) pdb=\(rnd(r.pdbMs)) total=\(rnd(r.totalMs))ms peak=\(Int(r.peakMB))MB scAtoms=\(r.scAtoms) top1=\(r.designTop1.map{rnd($0)} ?? "-") rmsd=\(r.repackRmsd ?? -1) file=\(r.pdbFile ?? "none")")
            }
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? enc.encode(results) {
                try? data.write(to: docsDir.appendingPathComponent("results.json"))
            }
            plog("BENCH COMPLETE — wrote results.json")
        } catch { plog("ERROR \(error)") }
    }

    private static func rnd(_ x: Double) -> String { String(format: "%.1f", x) }
}
