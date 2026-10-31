import SwiftUI
import GLiNERMLX

enum DemoRuntimeSettings {
    static let startupTF32 = UserDefaults.standard.bool(forKey: "demo.allowTF32")
    #if os(iOS)
    static let defaultCache = "20"
    static let defaultClear = true
    #else
    static let defaultCache = "auto"
    static let defaultClear = false
    #endif
    static func configuration(tf32: Bool, cache: String, clear: Bool) -> RuntimeConfiguration {
        let bytes = cache == "auto" ? nil : Int(cache).map { $0 * 1024 * 1024 }
        return RuntimeConfiguration(allowTF32: tf32, cacheLimitBytes: bytes, clearCacheAfterRequest: clear)
    }
    static func startupConfiguration() -> RuntimeConfiguration {
        let defaults = UserDefaults.standard
        let cache = defaults.string(forKey: "demo.cache") ?? defaultCache
        let clear = defaults.object(forKey: "demo.clear") == nil ? defaultClear : defaults.bool(forKey: "demo.clear")
        return configuration(tf32: startupTF32, cache: cache, clear: clear)
    }
}
@main
struct GLiNERDemoApp: App {
    init() {
        // Apply TF32 before the first model operation; subsequent errors are surfaced on extraction.
        let arguments = ProcessInfo.processInfo.arguments
        let candleTest = arguments.firstIndex(of: "--self-test-backend").map { $0 + 1 < arguments.count && arguments[$0 + 1] == "candle" } ?? false
        if !candleTest { try? GLiNERRuntime.configure(DemoRuntimeSettings.startupConfiguration()) }
    }
    var body: some Scene { WindowGroup { ContentView() } }
}
