import Foundation
import Darwin
import MLX

public struct RuntimeMemorySnapshot: Codable, Sendable {
    public let activeBytes: Int
    public let cachedBytes: Int
    public let peakActiveBytes: Int
}

public struct RuntimeConfiguration: Equatable, Codable, Sendable {
    public var allowTF32: Bool
    /// Nil restores MLX's initial cache policy; zero disables unused-buffer caching.
    public var cacheLimitBytes: Int?
    public var clearCacheAfterRequest: Bool

    public init(allowTF32: Bool = false, cacheLimitBytes: Int? = nil, clearCacheAfterRequest: Bool = false) {
        self.allowTF32 = allowTF32; self.cacheLimitBytes = cacheLimitBytes
        self.clearCacheAfterRequest = clearCacheAfterRequest
    }
    public static var platformDefault: RuntimeConfiguration {
        #if os(iOS)
        return RuntimeConfiguration(cacheLimitBytes: 20 * 1024 * 1024, clearCacheAfterRequest: true)
        #else
        return RuntimeConfiguration()
        #endif
    }
}

public enum InferenceProfile: String, Codable, CaseIterable, Sendable {
    case accurate, fast
    case lowMemory = "low-memory"

    public var precision: ModelPrecision { self == .accurate ? .float32 : .float16 }
    public var runtimeConfiguration: RuntimeConfiguration {
        var configuration = RuntimeConfiguration.platformDefault
        switch self {
        case .accurate: break
        case .fast: configuration.clearCacheAfterRequest = false
        case .lowMemory:
            configuration.cacheLimitBytes = 20 * 1024 * 1024
            configuration.clearCacheAfterRequest = true
        }
        return configuration
    }
}

/// Configure once at application startup, before other MLX matrix multiplications.
public enum GLiNERRuntime {
    private static let configurationLock = NSLock()
    private static var originalCacheLimit: Int?
    private static var appliedConfiguration: RuntimeConfiguration?
    private static var computationTF32: Bool?

    /// Disable the lower-precision TF32 route used by MLX on newer tensor accelerators.
    ///
    /// A tensor's `float32` dtype alone does not guarantee full F32 multiplication.
    /// MLX 0.33 caches this environment setting at the first relevant multiplication.
    /// Applications that also use other MLX models must call this before those models run.
    /// `GLiNERModel` calls it automatically before loading or evaluating any weights.
    public static func configureForReferenceParity() throws {
        try configure(.platformDefault)
    }

    /// TF32 is a process-start option in MLX 0.33. Cache settings can change later.
    public static func configure(_ configuration: RuntimeConfiguration) throws {
        configurationLock.lock(); defer { configurationLock.unlock() }
        if let bytes = configuration.cacheLimitBytes, bytes < 0 {
            throw GLiNERError.invalidConfiguration("Cache size must be nonnegative, or automatic")
        }
        if let computationTF32, computationTF32 != configuration.allowTF32 {
            throw GLiNERError.invalidConfiguration("TF32 cannot change after MLX computation has started. Close and reopen the application, or start a new CLI process.")
        }
        guard setenv("MLX_ENABLE_TF32", configuration.allowTF32 ? "1" : "0", 1) == 0 else {
            throw GLiNERError.invalidConfiguration("Cannot configure MLX precision: \(String(cString: strerror(errno)))")
        }
        if originalCacheLimit == nil { originalCacheLimit = Memory.cacheLimit }
        Memory.cacheLimit = configuration.cacheLimitBytes ?? originalCacheLimit!
        appliedConfiguration = configuration
    }

    static func markComputationStarted() {
        configurationLock.lock(); defer { configurationLock.unlock() }
        if computationTF32 == nil { computationTF32 = appliedConfiguration?.allowTF32 ?? false }
    }

    public static func memorySnapshot() -> RuntimeMemorySnapshot {
        let snapshot = Memory.snapshot()
        return RuntimeMemorySnapshot(activeBytes: snapshot.activeMemory, cachedBytes: snapshot.cacheMemory,
                                     peakActiveBytes: snapshot.peakMemory)
    }

    /// Release unused MLX buffers. Live model weights and arrays remain valid.
    public static func releaseUnusedMemory() { Memory.clearCache() }
}
