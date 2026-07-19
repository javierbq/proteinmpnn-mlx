// MemorySampler.swift — peak resident memory (phys_footprint) via a background poller.
import Foundation

func currentFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<Int32>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return Double(info.phys_footprint) / (1024 * 1024)
}

final class MemorySampler {
    private var timer: DispatchSourceTimer?
    private let lock = NSLock()
    private var _peak: Double = 0
    let baselineMB: Double

    init() {
        let f = currentFootprintMB()
        baselineMB = f
        _peak = f
    }

    func start() {
        let t = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInitiated))
        t.schedule(deadline: .now(), repeating: .milliseconds(15))
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let f = currentFootprintMB()
            self.lock.lock(); if f > self._peak { self._peak = f }; self.lock.unlock()
        }
        t.resume()
        timer = t
    }

    func stop() { timer?.cancel(); timer = nil }

    var peakMB: Double { lock.lock(); defer { lock.unlock() }; return _peak }
    var deltaMB: Double { peakMB - baselineMB }
}
