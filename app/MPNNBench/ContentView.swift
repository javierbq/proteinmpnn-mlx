import SwiftUI

struct ContentView: View {
    var body: some View {
        BenchView()
            .task {
                // Headless capture path: `devicectl ... launch --environment-variables
                // '{"MPNN_AUTORUN":"1"}'` runs the full sweep and prints PARITY|bench lines
                // to stderr so on-device numbers can be captured over the console.
                if ProcessInfo.processInfo.environment["MPNN_AUTORUN"] == "1" {
                    await Task.detached(priority: .userInitiated) {
                        Diagnostics.setup()
                        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                        let ids = ((try? Assets.manifest()) ?? []).map { $0.id }
                        Diagnostics.runBenchmark(ids, docsDir: docs)
                    }.value
                }
            }
    }
}
