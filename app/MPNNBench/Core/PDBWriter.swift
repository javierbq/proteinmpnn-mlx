// PDBWriter.swift — assemble a PDB from the designed sequence + packed side chains.
// Backbone atoms (N,CA,C,O) come from the input coordinates; side chains (atoms 4..13)
// from the repacked atom14. Atom names are looked up per AF2 residue type.
import Foundation
import MLX

struct AtomNames: Decodable {
    let restypes3: [String]       // AF2 order, 3-letter
    let atom14_names: [[String]]  // 14 names per restype (AF2 order)
    let mpnn_to_af2: [Int]        // MPNN alphabet index -> AF2 restype index
}

enum PDBWriter {
    static func loadNames() throws -> AtomNames {
        let url = AssetsBase.directory.appendingPathComponent("atom14_names.json")
        return try JSONDecoder().decode(AtomNames.self, from: Data(contentsOf: url))
    }

    /// backbone [1,L,4,3], atom14 [1,L,14,3], atom14Mask [1,L,14], bFactors [1,L,14]
    static func write(backbone: MLXArray, atom14: MLXArray, atom14Mask: MLXArray, bFactors: MLXArray,
                      seqMPNN: [Int], chainLabels: [Int], residueIdx: [Int], names: AtomNames) -> String {
        let L = seqMPNN.count
        let bb = backbone.asType(.float32).asArray(Float.self)     // L*4*3
        let a14 = atom14.asType(.float32).asArray(Float.self)      // L*14*3
        let am = atom14Mask.asType(.float32).asArray(Float.self)   // L*14
        let bf = bFactors.asType(.float32).asArray(Float.self)     // L*14

        let uniq = Array(Set(chainLabels)).sorted()
        var chainLetter: [Int: String] = [:]
        for (k, lbl) in uniq.enumerated() {
            chainLetter[lbl] = String(UnicodeScalar(UInt8(65 + (k % 26))))
        }

        var lines: [String] = ["REMARK   MPNNBench designed sequence + repacked side chains"]
        var serial = 1
        var prevChain = chainLabels.first ?? 0
        for i in 0 ..< L {
            if chainLabels[i] != prevChain { lines.append("TER"); prevChain = chainLabels[i] }
            let af2 = names.mpnn_to_af2[seqMPNN[i]]
            let res3 = names.restypes3[af2]
            let anames = names.atom14_names[af2]
            let ch = chainLetter[chainLabels[i]] ?? "A"
            for j in 0 ..< 14 {
                if am[i * 14 + j] < 0.5 { continue }
                let name = anames[j]
                if name.isEmpty { continue }
                let x: Float, y: Float, z: Float
                if j < 4 { x = bb[(i * 4 + j) * 3]; y = bb[(i * 4 + j) * 3 + 1]; z = bb[(i * 4 + j) * 3 + 2] }
                else { x = a14[(i * 14 + j) * 3]; y = a14[(i * 14 + j) * 3 + 1]; z = a14[(i * 14 + j) * 3 + 2] }
                lines.append(atomRecord(serial: serial, name: name, res: res3, chain: ch,
                                        resSeq: residueIdx[i], x: x, y: y, z: z, bfac: bf[i * 14 + j]))
                serial += 1
            }
        }
        lines.append("TER")
        lines.append("END")
        return lines.joined(separator: "\n") + "\n"
    }

    private static func atomRecord(serial: Int, name: String, res: String, chain: String,
                                   resSeq: Int, x: Float, y: Float, z: Float, bfac: Float) -> String {
        // atom-name field: leading space then left-justified into 4 cols (element in col 13-14)
        let nameField = String((" " + name + "   ").prefix(4))
        let elem = String(name.prefix(1))
        return String(format: "ATOM  %5d %@ %@ %@%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2@",
                      serial, nameField, res, chain, resSeq, x, y, z, 1.00, bfac, elem)
    }
}
