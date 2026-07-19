import SwiftUI
import UIKit

@MainActor
final class BenchModel: ObservableObject {
    @Published var results: [BenchResult] = []
    @Published var running = false
    @Published var status = "Tap Run to benchmark on-device"
    @Published var manifest: [ManifestEntry] = []
    @Published var deviceLine = ""

    func load() {
        Diagnostics.setup()
        manifest = (try? Assets.manifest()) ?? []
        deviceLine = "\(UIDevice.current.model) · iOS \(UIDevice.current.systemVersion)"
    }

    func runSweep() {
        guard !running else { return }
        running = true; results = []; status = "Loading models…"
        Task.detached(priority: .userInitiated) {
            do {
                let wD = try Assets.designWeights()
                let wP = try Assets.packerWeights()
                let gc = RepackConstants(try Assets.geometry())
                let names = try PDBWriter.loadNames()
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                await MainActor.run { self.status = "Warming up…" }
                Bench.warmup(wD)
                let manifest = try Assets.manifest()
                for meta in manifest {
                    await MainActor.run { self.status = "Running \(meta.id) (L=\(meta.L))…" }
                    let res = Bench.run(meta, wD: wD, wP: wP, gc: gc, names: names, docsDir: docs)
                    await MainActor.run { self.results.append(res) }
                }
                await MainActor.run { self.status = "Done — \(self.results.count) proteins"; self.running = false }
            } catch {
                await MainActor.run { self.status = "Error: \(error)"; self.running = false }
            }
        }
    }

    /// results.json + every generated PDB, for the share sheet.
    func exportURLs() -> [URL] {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        var urls: [URL] = []
        if let data = try? JSONEncoder.pretty.encode(results) {
            let u = docs.appendingPathComponent("mpnnbench_results.json")
            try? data.write(to: u); urls.append(u)
        }
        for r in results where r.pdbFile != nil {
            urls.append(docs.appendingPathComponent(r.pdbFile!))
        }
        return urls
    }
}

extension JSONEncoder {
    static var pretty: JSONEncoder { let e = JSONEncoder(); e.outputFormatting = [.prettyPrinted, .sortedKeys]; return e }
}

struct BenchView: View {
    @StateObject private var model = BenchModel()
    @State private var showShare = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                header
                Divider()
                if model.results.isEmpty {
                    proteinList
                } else {
                    resultsList
                }
            }
            .navigationTitle("MPNNBench")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    if !model.results.isEmpty && !model.running {
                        Button { showShare = true } label: { Image(systemName: "square.and.arrow.up") }
                    }
                }
            }
            .sheet(isPresented: $showShare) { ShareSheet(items: model.exportURLs()) }
        }
        .onAppear { model.load() }
    }

    private var header: some View {
        VStack(spacing: 8) {
            Text(model.deviceLine).font(.caption).foregroundStyle(.secondary)
            Text(model.status).font(.subheadline.weight(.medium))
                .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal)
            Button(action: model.runSweep) {
                HStack {
                    if model.running { ProgressView().tint(.white) }
                    Text(model.running ? "Running…" : "Run benchmark")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity).padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.running)
            .padding(.horizontal)
        }
        .padding(.vertical, 10)
    }

    private var proteinList: some View {
        List(model.manifest) { m in
            HStack {
                Text(m.id).font(.body.monospaced().weight(.semibold))
                if m.synthetic {
                    Text("synthetic").font(.caption2).padding(.horizontal, 6).padding(.vertical, 2)
                        .background(.orange.opacity(0.2)).clipShape(Capsule())
                }
                Spacer()
                Text("L=\(m.L) · \(m.n_chains) chain\(m.n_chains == 1 ? "" : "s")")
                    .font(.callout.monospacedDigit()).foregroundStyle(.secondary)
            }
        }
        .listStyle(.plain)
    }

    private var resultsList: some View {
        List(model.results) { r in ResultRow(r: r) }.listStyle(.plain)
    }
}

private struct ResultRow: View {
    let r: BenchResult
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(r.id).font(.body.monospaced().weight(.semibold))
                if r.synthetic {
                    Text("synth").font(.caption2).padding(.horizontal, 5).padding(.vertical, 1)
                        .background(.orange.opacity(0.2)).clipShape(Capsule())
                }
                Spacer()
                Text("L \(r.L)").font(.callout.monospacedDigit()).foregroundStyle(.secondary)
                parityBadge
            }
            if let e = r.error {
                Text(e).font(.caption).foregroundStyle(.red)
            } else {
                HStack(spacing: 14) {
                    stat("design", r.designMs)
                    stat("repack", r.repackMs)
                    stat("total", r.totalMs)
                    Spacer()
                    VStack(alignment: .trailing, spacing: 1) {
                        Text("\(Int(r.peakMB)) MB").font(.callout.monospacedDigit().weight(.medium))
                        Text("peak").font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func stat(_ label: String, _ ms: Double) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(ms >= 1000 ? String(format: "%.2fs", ms / 1000) : String(format: "%.0fms", ms))
                .font(.callout.monospacedDigit().weight(.medium))
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    @ViewBuilder private var parityBadge: some View {
        if let pass = r.parityPass {
            Image(systemName: pass ? "checkmark.seal.fill" : "xmark.seal.fill")
                .foregroundStyle(pass ? .green : .red)
        } else if !r.synthetic {
            Image(systemName: "questionmark.circle").foregroundStyle(.secondary)
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
