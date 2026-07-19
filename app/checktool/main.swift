// macOS command-line validator for the Swift MLX transcription (real Metal, no iOS
// device needed). Shares the exact Core/*.swift sources used by the app.
// Usage: mpnncheck <path-to-app_assets>
import Foundation
import MLX

MLX.GPU.set(cacheLimit: 96 * 1024 * 1024)

let assetsPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/Users/jcastellanos/repos/proteinmpnn-ios/app/MPNNBench/Resources/app_assets"
AssetsBase.directory = URL(fileURLWithPath: assetsPath)
print("mpnncheck: assets = \(assetsPath)")

Diagnostics.runDesignParity(["6MRR", "5L33", "4GYT", "3HTN"])
Diagnostics.runRepackParity(["6MRR", "5L33", "4GYT", "3HTN"])

// full pipeline (design -> repack -> write PDB) end-to-end, writing PDBs to a scratch dir
let outDir = URL(fileURLWithPath: "/private/tmp/claude-501/-Users-jcastellanos-repos-proteinmpnn-ios/883f49b5-ce95-4da2-abb5-d1eb715890f1/scratchpad/pdbout")
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
Diagnostics.runBenchmark(["6MRR", "6EHB"], docsDir: outDir)
