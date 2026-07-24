import XCTest
import MLX
@testable import MPNNKit

/// Regression for the repack crash on structures containing a non-canonical residue.
///
/// The MPNN alphabet is "ACDEFGHIKLMNPQRSTVWYX" (index 20 = 'X', unknown/masked).
/// `mpnn_to_af2[20] == 20`, but the AF2 name tables (`restypes3`, `atom14_names`)
/// only cover the 20 canonical residues (indices 0..<20). A sequence value of 20
/// therefore indexed `restypes3[20]` out of bounds and trapped — reproduced by any
/// structure with a modified residue that still has an N/CA/C/O backbone (e.g. MSE).
///
/// These tests are asset-free (no model weights): they build a minimal `AtomNames`
/// with the exact real table SHAPE that triggered the bug, plus tiny coordinate
/// arrays, and exercise `PDBWriter.write` directly.
final class PDBWriterUnknownTests: XCTestCase {

    // A minimal name table mirroring the real MPNN.mpnnpack shape:
    //   restypes3 / atom14_names cover only the 20 canonical residues,
    //   mpnn_to_af2 has 21 entries with [20] -> 20 (past the canonical tables).
    private func makeNames() -> AtomNames {
        let restypes3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                         "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
        // 14 atom-name slots per canonical restype; only the backbone + a CB are populated
        // (enough to assert canonical output; the rest are empty like the real table's tails).
        let backbone = ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""]
        let atom14_names = Array(repeating: backbone, count: 20)
        let mpnn_to_af2 = [0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20]
        return AtomNames(restypes3: restypes3, atom14_names: atom14_names, mpnn_to_af2: mpnn_to_af2)
    }

    /// backbone [1,L,4,3], atom14 [1,L,14,3], atom14Mask [1,L,14], bFactors [1,L,14].
    /// Backbone-atom mask bits are set; side-chain bits mirror `sidechainOn`.
    private func makeArrays(L: Int, sidechainOn: Bool)
        -> (bb: MLXArray, a14: MLXArray, mask: MLXArray, bf: MLXArray) {
        let bb = MLXArray((0 ..< L * 4 * 3).map { Float($0) }, [1, L, 4, 3])
        let a14 = MLXArray((0 ..< L * 14 * 3).map { Float($0) }, [1, L, 14, 3])
        var maskVals = [Float](repeating: 0, count: L * 14)
        for i in 0 ..< L {
            for j in 0 ..< 4 { maskVals[i * 14 + j] = 1 }        // backbone always present
            if sidechainOn { maskVals[i * 14 + 4] = 1 }          // one CB slot
        }
        let mask = MLXArray(maskVals, [1, L, 14])
        let bf = MLXArray([Float](repeating: 0.5, count: L * 14), [1, L, 14])
        return (bb, a14, mask, bf)
    }

    /// The crash case: a single unknown residue (X = 20). Must not trap; must emit
    /// a backbone-only UNK record instead of indexing the canonical tables OOB.
    func testWriteHandlesUnknownRestypeX() {
        let names = makeNames()
        let a = makeArrays(L: 1, sidechainOn: false)
        let pdb = PDBWriter.write(backbone: a.bb, atom14: a.a14, atom14Mask: a.mask, bFactors: a.bf,
                                  seqMPNN: [20], chainLabels: [0], residueIdx: [1], names: names)
        XCTAssertTrue(pdb.contains("UNK"), "unknown residue should be written as UNK")
        XCTAssertTrue(pdb.contains(" CA "), "backbone CA should be present for the unknown residue")
        XCTAssertTrue(pdb.contains("ATOM  "), "should contain ATOM records")
    }

    /// A canonical residue still writes normally (fix must not regress the happy path).
    func testWriteCanonicalResidueUnchanged() {
        let names = makeNames()
        let a = makeArrays(L: 1, sidechainOn: true)
        let pdb = PDBWriter.write(backbone: a.bb, atom14: a.a14, atom14Mask: a.mask, bFactors: a.bf,
                                  seqMPNN: [0], chainLabels: [0], residueIdx: [1], names: names)  // 0 = ALA
        XCTAssertTrue(pdb.contains("ALA"), "canonical residue keeps its 3-letter name")
        XCTAssertTrue(pdb.contains(" CB "), "canonical side-chain atom (CB) present")
    }

    /// Regression: an unknown residue in the MIDDLE must not abort the loop —
    /// canonical residues after it must still be written.
    func testWriteContinuesPastUnknownResidue() {
        let names = makeNames()
        let a = makeArrays(L: 2, sidechainOn: true)
        // position 0 = X (unknown), position 1 = ALA (canonical)
        let pdb = PDBWriter.write(backbone: a.bb, atom14: a.a14, atom14Mask: a.mask, bFactors: a.bf,
                                  seqMPNN: [20, 0], chainLabels: [0, 0], residueIdx: [1, 2], names: names)
        XCTAssertTrue(pdb.contains("UNK"), "unknown residue written")
        XCTAssertTrue(pdb.contains("ALA"), "canonical residue after the unknown still written")
    }
}
