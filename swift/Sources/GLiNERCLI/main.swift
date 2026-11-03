import Foundation
import GLiNERMLX
import MLX

let arguments = Array(CommandLine.arguments.dropFirst())
func value(_ flag: String) -> String? {
    guard let i = arguments.firstIndex(of: flag), i + 1 < arguments.count else { return nil }; return arguments[i + 1]
}
func values(_ flag: String) -> [String] {
    arguments.indices.compactMap { i in arguments[i] == flag && i + 1 < arguments.count ? arguments[i + 1] : nil }
}
let valueFlags: Set<String> = ["--model", "--cases", "--request", "--text", "--entities", "--relations", "--classify", "--structure",
                               "--threshold", "--max-tokens", "--chunk-size", "--chunk-overlap", "--profile", "--cache-mb"]
let booleanFlags: Set<String> = ["--fp16", "--fp32", "--tf32", "--no-tf32", "--clear-cache", "--retain-cache", "--cpu",
                                  "--no-spans", "--no-confidence", "--chunked", "--char-split"]
func validateArguments() throws {
    var i = 0, seen: [String: Int] = [:]
    let repeatable: Set<String> = ["--classify", "--structure"]
    while i < arguments.count {
        let flag = arguments[i]
        guard valueFlags.contains(flag) || booleanFlags.contains(flag) || flag == "--help" || flag == "-h" else {
            throw GLiNERError.invalidInput("Unknown option or unexpected argument: \(flag)")
        }
        seen[flag, default: 0] += 1
        if valueFlags.contains(flag) {
            guard i + 1 < arguments.count, !arguments[i + 1].hasPrefix("--") else {
                throw GLiNERError.invalidInput("Missing value for \(flag)")
            }
            if seen[flag, default: 0] > 1 && !repeatable.contains(flag) {
                throw GLiNERError.invalidInput("Option \(flag) may only be specified once")
            }
            i += 2
        } else {
            if seen[flag, default: 0] > 1 { throw GLiNERError.invalidInput("Option \(flag) may only be specified once") }
            i += 1
        }
    }
    func has(_ flag: String) -> Bool { seen[flag, default: 0] > 0 }
    if has("--tf32") && has("--no-tf32") { throw GLiNERError.invalidInput("Choose either --tf32 or --no-tf32") }
    if has("--clear-cache") && has("--retain-cache") { throw GLiNERError.invalidInput("Choose either --clear-cache or --retain-cache") }
    if has("--fp16") && has("--fp32") { throw GLiNERError.invalidInput("Choose either --fp16 or --fp32") }
    let sources = ["--cases", "--request", "--text"].filter(has)
    if sources.count > 1 { throw GLiNERError.invalidInput("Choose one input source: --cases, --request, or --text") }
    if has("--cases") || has("--request") {
        let direct = ["--entities", "--relations", "--classify", "--structure", "--threshold", "--char-split", "--no-spans", "--no-confidence"].filter(has)
        if !direct.isEmpty { throw GLiNERError.invalidInput("These per-text options cannot be combined with --cases/--request: \(direct.joined(separator: ", "))") }
    }
    if (has("--chunk-size") || has("--chunk-overlap")) && !has("--chunked") {
        throw GLiNERError.invalidInput("--chunk-size and --chunk-overlap require --chunked")
    }
    if let threshold = value("--threshold"), let number = Float(threshold), !number.isFinite {
        throw GLiNERError.invalidInput("--threshold must be finite")
    }
    for flag in ["--max-tokens", "--chunk-size"] {
        if let raw = value(flag) {
            guard let number = Int(raw), number > 0 else { throw GLiNERError.invalidInput("\(flag) must be a positive integer") }
        }
    }
    if let overlap = value("--chunk-overlap") {
        guard let number = Int(overlap), number >= 0 else { throw GLiNERError.invalidInput("--chunk-overlap must be a nonnegative integer") }
        let size = Int(value("--chunk-size") ?? "384") ?? 384
        guard number < size else { throw GLiNERError.invalidInput("--chunk-overlap must be smaller than --chunk-size") }
    }
    if let cache = value("--cache-mb"), cache != "auto" {
        guard let megabytes = Int(cache), megabytes >= 0,
              !megabytes.multipliedReportingOverflow(by: 1_048_576).overflow else {
            throw GLiNERError.invalidInput("--cache-mb must be auto or a nonnegative integer number of MiB")
        }
    }
    if let profile = value("--profile"), !["accurate", "fast", "low-memory", "lowMemory"].contains(profile) {
        throw GLiNERError.invalidInput("--profile must be accurate, fast, or low-memory")
    }
}
func positiveInt(_ flag: String, default fallback: Int) throws -> Int {
    guard let raw = value(flag) else { return fallback }
    guard let result = Int(raw), result > 0 else { throw GLiNERError.invalidInput("\(flag) must be a positive integer") }
    return result
}
func note(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }
func emit(_ result: JSONValue) throws {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    FileHandle.standardOutput.write(try encoder.encode(result) + Data([10]))
}
let help = """
GLiNER on Apple Silicon — local MLX / Metal inference

gliner-mlx --model DIRECTORY --text TEXT --entities person,company,location
gliner-mlx --model DIRECTORY --text TEXT --classify sentiment=positive,negative,neutral
gliner-mlx --model DIRECTORY --text TEXT --relations works_for
gliner-mlx --model DIRECTORY --text TEXT --structure 'purchase=buyer::str,item::str'
gliner-mlx --model DIRECTORY --cases CASES.json

Options:
  --profile accurate|fast|low-memory  Runtime preset (default: accurate)
  --fp16 / --fp32           Override profile precision (F16 is experimental)
  --tf32 / --no-tf32        Enable or disable TF32 (mutually exclusive)
  --cache-mb auto|0|20|128|512|N  MLX cache limit in MiB (default: profile)
  --clear-cache / --retain-cache  Clear MLX cache after each request or retain it
  --cpu                     Execute MLX on CPU (default: Apple GPU via Metal)
  --threshold 0.5           Extraction score cutoff
  --char-split              Character word splitting for CJK
  --no-spans                Omit character offsets (included by default)
  --no-confidence           Omit confidence (included by default)
  --chunked                 Split long documents into overlapping word windows
  --chunk-size 384          Words per window
  --chunk-overlap 64        Overlapping words
  --max-tokens 4096         Maximum encoder tokens, schema included; never silently truncated
  --request REQUEST.json    A single JSON extraction request

Download a model checkpoint from https://huggingface.co/fastino and pass its directory with --model.
"""

func run(precision: ModelPrecision = .float32, runtimeConfiguration: RuntimeConfiguration = .platformDefault) throws {
    if arguments.isEmpty || arguments.contains("--help") || arguments.contains("-h") { print(help); return }
    guard let modelPath = value("--model") else { throw GLiNERError.invalidInput("Provide --model DIRECTORY. Use --help for examples.") }
    let directory = URL(fileURLWithPath: modelPath)
    var requests: [InferenceRequest]
    if let cases = value("--cases") {
        requests = try InferenceRequest.loadCases(url: URL(fileURLWithPath: cases))
    } else if let path = value("--request") {
        requests = [try InferenceRequest(json: Data(contentsOf: URL(fileURLWithPath: path)))]
    } else {
        let text = value("--text") ?? String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .newlines)
        var object: [String: Any] = ["text": text]
        if let labels = value("--entities") { object["entities"] = labels.split(separator: ",").map(String.init) }
        if let labels = value("--relations") { object["relations"] = labels.split(separator: ",").map(String.init) }
        let tasks: [[String: Any]] = try values("--classify").map { task in
            let parts = task.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { throw GLiNERError.invalidInput("--classify expects task=label1,label2") }
            return ["task": parts[0], "labels": parts[1].split(separator: ",").map(String.init)]
        }
        if !tasks.isEmpty { object["classifications"] = tasks }
        var structures: [String: [String]] = [:]
        for definition in values("--structure") {
            let parts = definition.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { throw GLiNERError.invalidInput("--structure expects name=field::str,field::list") }
            structures[parts[0]] = parts[1].split(separator: ",").map(String.init)
        }
        if !structures.isEmpty { object["json"] = structures }
        if let raw = value("--threshold") {
            guard let threshold = Float(raw) else { throw GLiNERError.invalidInput("--threshold must be numeric") }; object["threshold"] = threshold
        }
        object["include_spans"] = !arguments.contains("--no-spans")
        object["include_confidence"] = !arguments.contains("--no-confidence")
        object["char_split"] = arguments.contains("--char-split")
        requests = [try InferenceRequest(json: JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]))]
    }
    let limit = try positiveInt("--max-tokens", default: 4096)
    note("Loading \(directory.lastPathComponent) (\(precision.rawValue), \(arguments.contains("--cpu") ? "CPU" : "Metal GPU"))…")
    let model = try GLiNERModel(modelDirectory: directory, precision: precision, maxInputTokens: limit,
                                runtimeConfiguration: runtimeConfiguration)
    for request in requests {
        model.wordSplitter = request.wordSplitter
        if arguments.contains("--chunked") {
            let chunks = ChunkOptions(chunkSize: try positiveInt("--chunk-size", default: 384),
                                      chunkOverlap: Int(value("--chunk-overlap") ?? "64") ?? 64)
            try emit(model.extractLong(request.text, schema: request.schema, options: request.options, chunks: chunks))
        } else {
            try emit(model.extract(request.text, schema: request.schema, options: request.options))
        }
    }
}

func runtimeSettings() throws -> (precision: ModelPrecision, configuration: RuntimeConfiguration) {
    let requested = value("--profile") ?? "accurate"
    let profile: InferenceProfile
    switch requested {
    case "accurate": profile = .accurate
    case "fast": profile = .fast
    case "low-memory", "lowMemory": profile = .lowMemory
    default: throw GLiNERError.invalidInput("--profile must be accurate, fast, or low-memory")
    }
    var precision = profile.precision
    if arguments.contains("--fp16") { precision = .float16 }
    if arguments.contains("--fp32") { precision = .float32 }

    let defaults = profile.runtimeConfiguration
    let allowTF32 = arguments.contains("--tf32") ? true : arguments.contains("--no-tf32") ? false : defaults.allowTF32
    let cacheLimitBytes: Int?
    if let cache = value("--cache-mb") {
        if cache == "auto" { cacheLimitBytes = nil }
        else {
            let megabytes = Int(cache)!
            cacheLimitBytes = megabytes * 1_048_576
        }
    } else { cacheLimitBytes = defaults.cacheLimitBytes }
    let clearCache = arguments.contains("--clear-cache") ? true : arguments.contains("--retain-cache") ? false : defaults.clearCacheAfterRequest
    let runtime = RuntimeConfiguration(allowTF32: allowTF32, cacheLimitBytes: cacheLimitBytes,
                                       clearCacheAfterRequest: clearCache)
    return (precision, runtime)
}

do {
    if arguments.isEmpty || arguments.contains("--help") || arguments.contains("-h") {
        try run()
    } else {
        try validateArguments()
        let settings = try runtimeSettings()
        try GLiNERRuntime.configure(settings.configuration)
        if settings.precision == .float16 {
            note("Warning: F16 is experimental and may change extracted spans or confidence scores.")
        }
        if arguments.contains("--cpu") {
            try Device.withDefaultDevice(.cpu) { try run(precision: settings.precision, runtimeConfiguration: settings.configuration) }
        } else {
            try run(precision: settings.precision, runtimeConfiguration: settings.configuration)
        }
    }
} catch { note("Error: \(error.localizedDescription)"); exit(1) }
