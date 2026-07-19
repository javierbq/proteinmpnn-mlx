import SwiftUI
import MLX

@main
struct MPNNBenchApp: App {
    init() {
        // Bound MLX's buffer cache so the memory pool doesn't balloon toward the iOS
        // jetsam limit on large proteins (measured >5 GB unbounded at L~1000).
        MLX.GPU.set(cacheLimit: 96 * 1024 * 1024)

        #if targetEnvironment(simulator)
        // The iOS Simulator's Metal cannot allocate MLX's private-storage heaps
        // (MTLStorageModePrivate assertion), and its architecture()->name() is null
        // (std::string(nullptr) abort under iOS 26 libc++ hardening). Force the CPU
        // backend and supply an arch string so validation can run in the Simulator.
        // Real devices use the GPU — this block is simulator-only.
        setenv("MLX_METAL_GPU_ARCH", "applegpu_g15g", 1)
        MLX.Device.setDefault(device: Device(.cpu))
        #endif
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
