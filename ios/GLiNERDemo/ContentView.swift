import SwiftUI
import Foundation
import Darwin
import GLiNERMLX

enum DemoVariant: String, CaseIterable, Identifiable {
    case small, multi
    var id: String { rawValue }
    var title: String { self == .small ? "Small · inglese, circa 304 MB" : "Multi · italiano incluso, circa 1,17 GB" }
    var repo: String { "fastino/gliner2.5-\(rawValue)-v1" }
    var revision: String { self == .small ? "df5910e44bc4ffdb0d95399a83b0ca4516349aa5" : "cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3" }
    var sizes: [String: Int64] {
        self == .small ? ["config.json": 4104, "encoder_config/config.json": 856, "tokenizer.json": 8341713, "model.safetensors": 295567700]
                       : ["config.json": 4104, "encoder_config/config.json": 857, "tokenizer.json": 16035853, "model.safetensors": 1149461028]
    }
}

private struct Checkpoint: Codable {
    struct File: Codable { let size: Int64 }
    let repo: String, revision: String
    let files: [String: File]
}
private let modelFiles = ["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"]

enum DemoBackend: String, CaseIterable {
    case mlx, candle
    var title: String { self == .mlx ? "Swift · MLX Metal" : "Rust · Candle Metal (F32)" }
}
#if CANDLE_ENABLED
@_silgen_name("gliner_candle_open") private func candleOpen(_ path: UnsafePointer<CChar>, _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> OpaquePointer?
@_silgen_name("gliner_candle_extract") private func candleExtract(_ handle: OpaquePointer, _ request: UnsafePointer<CChar>, _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> UnsafeMutablePointer<CChar>?
@_silgen_name("gliner_candle_close") private func candleClose(_ handle: OpaquePointer)
@_silgen_name("gliner_candle_string_free") private func candleFree(_ value: UnsafeMutablePointer<CChar>)
private struct CandleRequest: Encodable {
    struct Options: Encodable {
        let threshold: Float, includeConfidence: Bool, includeSpans: Bool
        let overlapPolicy: String?, maxWords: Int?
        init(_ value: ExtractOptions) { threshold = value.threshold; includeConfidence = value.includeConfidence; includeSpans = value.includeSpans; overlapPolicy = value.overlapPolicy?.rawValue; maxWords = value.maxWords }
    }
    let text: String, schema: Schema, options: Options, wordSplitter: String
}
private final class CandleModel {
    private let handle: OpaquePointer
    init(directory: URL) throws {
        var error: UnsafeMutablePointer<CChar>?
        let result = directory.path.withCString { candleOpen($0, &error) }
        guard let result else { throw Self.failure(error) }; self.handle = result
    }
    deinit { candleClose(handle) }
    private static func failure(_ pointer: UnsafeMutablePointer<CChar>?) -> Error {
        guard let pointer else { return GLiNERError.invalidInput("Candle Metal: errore sconosciuto") }
        defer { candleFree(pointer) }; return GLiNERError.invalidInput("Candle Metal: " + String(cString: pointer))
    }
    func extract(text: String, schema: Schema, options: ExtractOptions, splitter: WordSplitter) throws -> JSONValue {
        let request = CandleRequest(text: text, schema: schema, options: .init(options), wordSplitter: splitter.rawValue)
        let data = try JSONEncoder().encode(request)
        guard let string = String(data: data, encoding: .utf8) else { throw GLiNERError.invalidInput("Invalid request encoding") }
        var error: UnsafeMutablePointer<CChar>?
        let result = string.withCString { candleExtract(handle, $0, &error) }
        guard let result else { throw Self.failure(error) }; defer { candleFree(result) }
        return try JSONDecoder().decode(JSONValue.self, from: Data(String(cString: result).utf8))
    }
}
#endif
actor ModelWorker {
    private struct LoadKey: Equatable {
        let variant: DemoVariant, directory: URL, precision: ModelPrecision, tf32: Bool, cache: String, clear: Bool, backend: DemoBackend
    }
    private var model: GLiNERModel?
    private var loaded: LoadKey?
    #if CANDLE_ENABLED
    private var candle: CandleModel?
    #endif
    func extract(variant: DemoVariant, directory: URL, text: String, labels: [String], precision: ModelPrecision,
                 tf32: Bool, cache: String, clear: Bool, backend: DemoBackend) throws -> (String, Double) {
        let start = Date()
        let key = LoadKey(variant: variant, directory: directory, precision: precision, tf32: tf32, cache: cache, clear: clear, backend: backend)
        #if CANDLE_ENABLED
        if backend == .candle {
            if loaded != key || candle == nil {
                model = nil; candle = nil; loaded = nil; GLiNERRuntime.releaseUnusedMemory()
                candle = try CandleModel(directory: directory); loaded = key
            }
            let value = try candle!.extract(text: text, schema: Schema(entities: labels), options: .init(includeSpans: true, includeConfidence: true), splitter: .whitespace)
            return (try value.prettyPrinted(), Date().timeIntervalSince(start))
        }
        #else
        if backend == .candle { throw GLiNERError.unsupported("Candle Metal non incluso in questa build") }
        #endif
        if loaded != key || model == nil {
            let config = DemoRuntimeSettings.configuration(tf32: tf32, cache: cache, clear: clear)
            if let previous = loaded, let existing = model,
               previous.variant == key.variant, previous.directory == key.directory,
               previous.precision == key.precision, previous.tf32 == key.tf32 {
                try existing.updateRuntimeConfiguration(config)
                loaded = key
            } else {
                model = nil; loaded = nil
                #if CANDLE_ENABLED
                candle = nil
                #endif
                GLiNERRuntime.releaseUnusedMemory()
                try GLiNERRuntime.configure(config)
                model = try GLiNERModel(modelDirectory: directory, precision: precision, runtimeConfiguration: config)
                loaded = key
            }
        }
        let extractionStart = Date()
        var extractionSeconds = 0.0, formattingSeconds = 0.0
        let json: String = try autoreleasepool {
            let value = try model!.extract(text, schema: Schema(entities: labels), options: .init(includeSpans: true, includeConfidence: true))
            extractionSeconds = Date().timeIntervalSince(extractionStart)
            let formattingStart = Date()
            let json = try value.prettyPrinted()
            formattingSeconds = Date().timeIntervalSince(formattingStart)
            return json
        }
        let total = Date().timeIntervalSince(start)
        if ProcessInfo.processInfo.environment["GLINER_DEMO_PROFILE"] == "1" {
            let line = String(format: "GLINER_DEMO_PROFILE total=%.6f extract=%.6f json=%.6f cache=%@ clear=%@ qos=%ld\n",
                              total, extractionSeconds, formattingSeconds, cache, clear ? "true" : "false", Thread.current.qualityOfService.rawValue)
            FileHandle.standardError.write(Data(line.utf8))
        }
        return (json, total)
    }
}

private struct SelfTestSummary: Encodable {
    var runID: String?
    var tf32Environment: String?
    var status: String = "failed"
    var caseCount: Int = 0
    var precision: String = "float32"
    var backend: String = "mlx"
    var loadSeconds: Double = 0
    var totalInferenceSeconds: Double = 0
    var perCaseSeconds: [Double] = []
    var peakResidentBytes: Int64?
    var mlxMemory: RuntimeMemorySnapshot?
    var error: String?
}
private actor SelfTestWorker {
    func run(arguments: [String]) -> String {
        let fm = FileManager.default
        var summary = SelfTestSummary()
        if let index = arguments.firstIndex(of: "--self-test-id"), index + 1 < arguments.count {
            summary.runID = arguments[index + 1]
        }
        if let setting = getenv("MLX_ENABLE_TF32") { summary.tf32Environment = String(cString: setting) }
        var temporary: URL?
        do {
            let docs = try fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let summaryURL = docs.appendingPathComponent("apple-self-test-summary.json")
            let resultsURL = docs.appendingPathComponent("apple-self-test-results.jsonl")
            let partURL = docs.appendingPathComponent("apple-self-test-results.jsonl.part")
            let traceURL = docs.appendingPathComponent("apple-self-test-encoder.jsonl")
            let tracePartURL = docs.appendingPathComponent("apple-self-test-encoder.jsonl.part")
            let wantsTrace = arguments.contains("--self-test-trace")
            let benchmark = arguments.contains("--benchmark")
            // Remove old success markers before any work starts.
            for url in [summaryURL, resultsURL, partURL, traceURL, tracePartURL] where fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
            temporary = partURL
            do {
                if let index = arguments.firstIndex(of: "--self-test-precision") {
                    guard index + 1 < arguments.count else { throw SelfTestError.invalidPrecision("missing value") }
                    summary.precision = arguments[index + 1]
                }
                guard let precision = ModelPrecision(rawValue: summary.precision) else { throw SelfTestError.invalidPrecision(summary.precision) }
                if let index = arguments.firstIndex(of: "--self-test-backend"), index + 1 < arguments.count { summary.backend = arguments[index + 1] }
                guard ["mlx", "candle"].contains(summary.backend) else { throw GLiNERError.invalidInput("Invalid backend: \(summary.backend)") }
                if summary.backend == "candle" && (precision != .float32 || wantsTrace) { throw GLiNERError.unsupported("Candle Metal supports F32 and no encoder trace in this app") }
                let directory = docs.appendingPathComponent("self-test-model", isDirectory: true)
                for file in ["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"] {
                    guard fm.fileExists(atPath: directory.appendingPathComponent(file).path) else { throw SelfTestError.missingFile(file) }
                }
                let caseFile = benchmark ? "apple-benchmark-cases.json" : "apple_cases.json"
                let cases = try InferenceRequest.loadCases(url: docs.appendingPathComponent(caseFile))
                summary.caseCount = cases.count
                let loadStart = DispatchTime.now().uptimeNanoseconds
                let model: GLiNERModel?
                #if CANDLE_ENABLED
                let candle: CandleModel?
                if summary.backend == "candle" { model = nil; candle = try CandleModel(directory: directory) }
                else { candle = nil; model = try GLiNERModel(modelDirectory: directory, precision: precision, runtimeConfiguration: DemoRuntimeSettings.startupConfiguration()) }
                #else
                guard summary.backend == "mlx" else { throw GLiNERError.unsupported("Candle Metal is not linked in this build") }
                model = try GLiNERModel(modelDirectory: directory, precision: precision, runtimeConfiguration: DemoRuntimeSettings.startupConfiguration())
                #endif
                if let setting = getenv("MLX_ENABLE_TF32") { summary.tf32Environment = String(cString: setting) }
                summary.loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1_000_000_000
                guard fm.createFile(atPath: partURL.path, contents: nil) else { throw SelfTestError.cannotCreateResults }
                let stream = try FileHandle(forWritingTo: partURL)
                var traceStream: FileHandle?
                do {
                    if wantsTrace {
                        guard fm.createFile(atPath: tracePartURL.path, contents: nil) else { throw SelfTestError.cannotCreateResults }
                        traceStream = try FileHandle(forWritingTo: tracePartURL)
                    }
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
                    for (caseIndex, request) in cases.enumerated() {
                        let iterations = benchmark ? 35 : 1
                        for iteration in 0..<iterations {
                            let elapsed: Double = try autoreleasepool {
                                let start = DispatchTime.now().uptimeNanoseconds
                                let result: JSONValue
                                #if CANDLE_ENABLED
                                if let candle { result = try candle.extract(text: request.text, schema: request.schema, options: request.options, splitter: request.wordSplitter) }
                                else {
                                    model!.wordSplitter = request.wordSplitter
                                    result = try model!.extract(request.text, schema: request.schema, options: request.options)
                                }
                                #else
                                model!.wordSplitter = request.wordSplitter
                                result = try model!.extract(request.text, schema: request.schema, options: request.options)
                                #endif
                                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000
                                if benchmark {
                                    let entry: JSONValue = .object(["runID": .string(summary.runID ?? ""), "backend": .string(summary.backend), "precision": .string(summary.precision), "caseIndex": .number(Double(caseIndex)), "iteration": .number(Double(iteration)), "warmup": .bool(iteration < 5), "seconds": .number(elapsed), "result": result])
                                    try stream.write(contentsOf: encoder.encode(entry))
                                } else { try stream.write(contentsOf: encoder.encode(result)) }
                                try stream.write(contentsOf: Data([10]))
                                if let traceStream, let model {
                                    let input = try model.prepare(request.text, schema: request.schema, maxWords: request.options.maxWords)
                                    let states = try model.encode(input)
                                    let trace: [String: Any] = ["input_ids": input.inputIds, "word_positions": input.wordPositions,
                                        "marker_positions": input.groups.flatMap(\.markerPositions), "encoder_shape": [1] + states.shape,
                                        "encoder_values": states.asArray(Float.self)]
                                    try traceStream.write(contentsOf: JSONSerialization.data(withJSONObject: trace, options: [.sortedKeys]))
                                    try traceStream.write(contentsOf: Data([10]))
                                }
                                return elapsed
                            }
                            summary.perCaseSeconds.append(elapsed); summary.totalInferenceSeconds += elapsed
                            if summary.backend == "mlx" {
                                if !benchmark { GLiNERRuntime.releaseUnusedMemory() }
                                summary.mlxMemory = GLiNERRuntime.memorySnapshot()
                            }
                        }
                    }
                    try stream.synchronize(); try stream.close()
                    try traceStream?.synchronize(); try traceStream?.close()
                } catch { try? stream.close(); try? traceStream?.close(); throw error }
                if wantsTrace { try fm.moveItem(at: tracePartURL, to: traceURL) }
                try fm.moveItem(at: partURL, to: resultsURL)
                temporary = nil
                summary.status = "passed"
            } catch { summary.error = error.localizedDescription; try? fm.removeItem(at: tracePartURL) }
            var usage = rusage()
            if getrusage(RUSAGE_SELF, &usage) == 0 { summary.peakResidentBytes = Int64(usage.ru_maxrss) }
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(summary).write(to: summaryURL, options: .atomic)
            if let temporary { try? fm.removeItem(at: temporary) }
            return summary.status == "passed" ? "Verifica automatica completata." : "Verifica automatica non riuscita: \(summary.error ?? "errore sconosciuto")"
        } catch {
            if let temporary { try? fm.removeItem(at: temporary) }
            return "Verifica automatica non riuscita: \(error.localizedDescription)"
        }
    }
}
private enum SelfTestError: LocalizedError {
    case invalidPrecision(String), missingFile(String), cannotCreateResults
    var errorDescription: String? {
        switch self {
        case .invalidPrecision(let value): return "Precisione non valida: \(value)"
        case .missingFile(let file): return "File del modello mancante: \(file)"
        case .cannotCreateResults: return "Impossibile creare il file dei risultati"
        }
    }
}

@MainActor
final class DemoState: ObservableObject {
    @Published var variant: DemoVariant = .multi
    @Published var text = "Michele vive a Milano e lavora per Acme."
    @Published var labels = "persona, città, organizzazione"
    @Published var status = "Seleziona e scarica un modello."
    @Published var output = ""
    @Published var busy = false
    @Published var downloading = false
    @Published var completedFiles = 0
    private let worker = ModelWorker()
    private let selfTestWorker = SelfTestWorker()
    private var selfTestStarted = false
    func runSelfTestIfRequested() async {
        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.contains("--self-test"), !selfTestStarted else { return }
        selfTestStarted = true; busy = true; status = "Verifica automatica in corso…"
        status = await selfTestWorker.run(arguments: arguments)
        busy = false
    }
    private let fm = FileManager.default
    func directory(_ variant: DemoVariant) throws -> URL {
        try fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("GLiNERMLX", isDirectory: true).appendingPathComponent(variant.rawValue, isDirectory: true)
    }
    private func verifiedDirectory(_ dir: URL, variant: DemoVariant) -> Bool {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("checkpoint.json")),
              let checkpoint = try? JSONDecoder().decode(Checkpoint.self, from: data), checkpoint.repo == variant.repo,
              checkpoint.revision == variant.revision else { return false }
        return modelFiles.allSatisfy { file in
            guard let expected = variant.sizes[file], checkpoint.files[file]?.size == expected,
                  let attrs = try? fm.attributesOfItem(atPath: dir.appendingPathComponent(file).path),
                  let size = attrs[.size] as? NSNumber else { return false }
            return size.int64Value == expected
        }
    }
    func readyDirectory(_ variant: DemoVariant) -> URL? {
        if let dir = try? directory(variant), verifiedDirectory(dir, variant: variant) { return dir }
        if let docs = try? fm.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true) {
            let dir = docs.appendingPathComponent("self-test-model", isDirectory: true)
            if verifiedDirectory(dir, variant: variant) { return dir }
        }
        return nil
    }
    func isReady(_ variant: DemoVariant) -> Bool { readyDirectory(variant) != nil }
    func refresh() { status = isReady(variant) ? "Modello disponibile offline." : "Modello non ancora scaricato." }
    func download() {
        guard !busy else { return }; busy = true; downloading = true; completedFiles = 0
        let chosen = variant
        Task {
            do {
                let dir = try directory(chosen)
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
                if isReady(chosen) { completedFiles = 4; status = "Modello disponibile offline."; busy = false; downloading = false; return }
                for (i, name) in modelFiles.enumerated() {
                    let displayName = ["Configurazione", "Configurazione encoder", "Vocabolario", "Pesi del modello"][i]
                    status = "Download \(i + 1) di 4: \(displayName)"
                    let destination = dir.appendingPathComponent(name)
                    try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
                    let source = URL(string: "https://huggingface.co/\(chosen.repo)/resolve/\(chosen.revision)/\(name)?download=true")!
                    let (temporary, response) = try await URLSession.shared.download(from: source)
                    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else { throw DemoError.download(name) }
                    let attrs = try fm.attributesOfItem(atPath: temporary.path)
                    guard let actual = attrs[.size] as? NSNumber, actual.int64Value == chosen.sizes[name] else { throw DemoError.size(name) }
                    let staging = destination.deletingLastPathComponent().appendingPathComponent(".\(destination.lastPathComponent).part")
                    try? fm.removeItem(at: staging)
                    try fm.moveItem(at: temporary, to: staging)
                    if fm.fileExists(atPath: destination.path) { _ = try fm.replaceItemAt(destination, withItemAt: staging) }
                    else { try fm.moveItem(at: staging, to: destination) }
                    completedFiles = i + 1
                }
                let checkpoint = Checkpoint(repo: chosen.repo, revision: chosen.revision, files: chosen.sizes.mapValues { Checkpoint.File(size: $0) })
                try JSONEncoder().encode(checkpoint).write(to: dir.appendingPathComponent("checkpoint.json"), options: .atomic)
                status = "Download completato. Modello disponibile offline."
            } catch { status = "Download non riuscito: \(error.localizedDescription)" }
            busy = false; downloading = false
        }
    }
    func extract(profile: InferenceProfile, tf32: Bool, cache: String, clear: Bool, backend: DemoBackend) {
        guard !busy else { return }
        let chosen = variant, input = text, parsed = labels.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        guard let modelDirectory = readyDirectory(chosen) else { status = "Scarica prima il modello selezionato."; return }
        guard !parsed.isEmpty else { status = "Inserisci almeno un’etichetta."; return }
        busy = true; status = "Estrazione in corso…"; output = ""
        Task {
            do {
                let (json, seconds) = try await worker.extract(variant: chosen, directory: modelDirectory, text: input, labels: parsed,
                                                               precision: profile.precision, tf32: tf32, cache: cache, clear: clear, backend: backend)
                output = json; status = String(format: "Completato in %.2f secondi. Il primo risultato include il caricamento del modello.", seconds)
            } catch { status = "Estrazione non riuscita: \(error.localizedDescription)" }
            busy = false
        }
    }
}
private enum DemoError: LocalizedError {
    case download(String), size(String)
    var errorDescription: String? {
        switch self { case .download(let file): return "Risposta HTTP non valida per \(file)."; case .size(let file): return "Dimensione inattesa per \(file)." }
    }
}
private struct EntityDisplayGroup: Identifiable {
    struct Item: Identifiable { let id: Int; let text: String; let confidence: Double? }
    let label: String
    let items: [Item]
    var id: String { label }
}
private func displayEntities(_ json: String) -> [EntityDisplayGroup] {
    guard let data = json.data(using: .utf8),
          let result = try? JSONDecoder().decode(JSONValue.self, from: data),
          case .object(let entities)? = result["entities"] else { return [] }
    return entities.keys.sorted().compactMap { label in
        guard let entry = entities[label] else { return nil }
        let values: [JSONValue]
        switch entry { case .array(let items): values = items; case .null: values = []; default: values = [entry] }
        let items: [EntityDisplayGroup.Item] = values.enumerated().compactMap { index, item in
            switch item {
            case .string(let text): return .init(id: index, text: text, confidence: nil)
            case .object(let fields):
                guard case .string(let text)? = fields["text"] else { return nil }
                let confidence: Double?
                if case .number(let score)? = fields["confidence"] { confidence = score } else { confidence = nil }
                return .init(id: index, text: text, confidence: confidence)
            default: return nil
            }
        }
        return items.isEmpty ? nil : EntityDisplayGroup(label: label, items: items)
    }
}
struct ContentView: View {
    @StateObject private var state = DemoState()
    @State private var backend: DemoBackend = .mlx
    @AppStorage("demo.profile") private var profile = InferenceProfile.accurate.rawValue
    @AppStorage("demo.cache") private var cache = DemoRuntimeSettings.defaultCache
    @AppStorage("demo.clear") private var clear = DemoRuntimeSettings.defaultClear
    @AppStorage("demo.allowTF32") private var tf32 = false
    private var selectedProfile: InferenceProfile { InferenceProfile(rawValue: profile) ?? .accurate }
    private var needsRestart: Bool { tf32 != DemoRuntimeSettings.startupTF32 }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("GLiNER").font(.largeTitle.bold())
                Picker("Modello", selection: $state.variant) { ForEach(DemoVariant.allCases) { Text($0.title).tag($0) } }
                    .disabled(state.busy).onChange(of: state.variant) { _, _ in state.refresh() }
                Button("Scarica modello") { state.download() }.disabled(state.busy)
                #if CANDLE_ENABLED
                Picker("Motore", selection: $backend) {
                    Text(DemoBackend.mlx.title).tag(DemoBackend.mlx)
                    Text(DemoBackend.candle.title).tag(DemoBackend.candle)
                }.disabled(state.busy)
                #endif
                Picker("Profilo", selection: $profile) {
                    Text("Precisione").tag(InferenceProfile.accurate.rawValue)
                    Text("Velocità").tag(InferenceProfile.fast.rawValue)
                    Text("Memoria ridotta").tag(InferenceProfile.lowMemory.rawValue)
                }.disabled(state.busy).onChange(of: profile) { _, _ in
                    if selectedProfile == .lowMemory { cache = "20"; clear = true }
                    else if selectedProfile == .fast { cache = DemoRuntimeSettings.defaultCache; clear = false }
                    else { cache = DemoRuntimeSettings.defaultCache; clear = DemoRuntimeSettings.defaultClear }
                }
                if selectedProfile != .accurate { Text("La precisione F16 è approssimata e può cambiare le risposte.").font(.caption).foregroundStyle(.secondary) }
                DisclosureGroup("Opzioni avanzate") {
                    Picker("Cache", selection: $cache) {
                        Text("Auto").tag("auto"); Text("20 MB").tag("20"); Text("128 MB").tag("128")
                        Text("512 MB").tag("512"); Text("0 MB").tag("0")
                    }
                    Toggle("Libera la cache tra le richieste", isOn: $clear)
                    Toggle("TF32 (chip recenti, può cambiare i risultati)", isOn: $tf32)
                }.disabled(state.busy || backend == .candle)
                if needsRestart && backend == .mlx { Text("Chiudi e riapri l’app per applicare TF32.").foregroundStyle(.orange) }
                if backend == .candle && selectedProfile != .accurate { Text("Candle Metal richiede il profilo Precisione F32.").foregroundStyle(.orange) }
                if state.downloading { ProgressView(value: Double(state.completedFiles), total: 4) { Text("File completati: \(state.completedFiles) di 4") } }
                Text(state.status).foregroundStyle(.secondary)
                Text("Testo").font(.headline)
                TextEditor(text: $state.text).frame(minHeight: 120).overlay(RoundedRectangle(cornerRadius: 6).stroke(.gray.opacity(0.3)))
                TextField("Etichette separate da virgole", text: $state.labels).textFieldStyle(.roundedBorder)
                Button("Estrai") { state.extract(profile: selectedProfile, tf32: tf32, cache: cache, clear: clear, backend: backend) }
                    .buttonStyle(.borderedProminent).disabled(state.busy || (backend == .mlx && needsRestart) || (backend == .candle && selectedProfile != .accurate))
                if !state.output.isEmpty {
                    let groups = displayEntities(state.output)
                    Text("Entità trovate").font(.headline)
                    if groups.isEmpty { Text("Nessuna entità trovata.").foregroundStyle(.secondary) }
                    ForEach(groups) { group in
                        VStack(alignment: .leading, spacing: 6) {
                            Text(group.label).font(.subheadline.bold())
                            ForEach(group.items) { item in
                                HStack(alignment: .firstTextBaseline) {
                                    Text(item.text).textSelection(.enabled)
                                    Spacer()
                                    if let confidence = item.confidence { Text(confidence, format: .percent.precision(.fractionLength(0))).foregroundStyle(.secondary) }
                                }
                            }
                        }
                    }
                    DisclosureGroup("Dettagli JSON") {
                        Text(state.output).font(.system(.body, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }.padding().frame(maxWidth: 800, alignment: .leading)
        }.onAppear { state.refresh() }.task { await state.runSelfTestIfRequested() }
    }
}
