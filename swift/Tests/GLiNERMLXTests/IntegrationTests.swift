import Foundation
import XCTest
@testable import GLiNERMLX

/// Runs only when a complete local checkpoint is available; never downloads weights.
final class IntegrationTests: XCTestCase {
    private static let lock = NSLock()
    private static var f32: GLiNERModel?
    private static var f16: GLiNERModel?
    private static func checkpoint() throws -> URL {
        let env = ProcessInfo.processInfo.environment["GLINER_MODEL_DIR"]
        let file = URL(fileURLWithPath: #filePath)
        let repo = file.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let dir = env.map { URL(fileURLWithPath: $0, isDirectory: true) } ?? repo.appendingPathComponent(".models/gliner2.5-small-v1", isDirectory: true)
        let files = ["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"]
        guard files.allSatisfy({ FileManager.default.fileExists(atPath: dir.appendingPathComponent($0).path) }) else { throw XCTSkip("Set GLINER_MODEL_DIR to a complete GLiNER small checkpoint") }
        return dir
    }
    private func model(_ precision: ModelPrecision = .float32) throws -> GLiNERModel {
        let directory = try Self.checkpoint()
        Self.lock.lock(); defer { Self.lock.unlock() }
        if precision == .float32 {
            if let model = Self.f32 { return model }
            let model = try GLiNERModel(modelDirectory: directory, precision: .float32); Self.f32 = model; return model
        }
        if let model = Self.f16 { return model }
        let model = try GLiNERModel(modelDirectory: directory, precision: .float16); Self.f16 = model; return model
    }
    private func assertJSON(_ a: JSONValue, _ b: JSONValue, tolerance: Double, path: String = "$", file: StaticString = #filePath, line: UInt = #line) {
        switch (a,b) {
        case let (.object(x), .object(y)):
            XCTAssertEqual(Set(x.keys), Set(y.keys), "keys at \(path)", file: file, line: line)
            for key in x.keys where y[key] != nil { assertJSON(x[key]!, y[key]!, tolerance: tolerance, path: "\(path).\(key)", file: file, line: line) }
        case let (.array(x), .array(y)):
            XCTAssertEqual(x.count, y.count, "count at \(path)", file: file, line: line)
            for (i, pair) in zip(x,y).enumerated() { assertJSON(pair.0, pair.1, tolerance: tolerance, path: "\(path)[\(i)]", file: file, line: line) }
        case let (.number(x), .number(y)):
            if path.hasSuffix(".confidence") { XCTAssertEqual(x, y, accuracy: tolerance, "at \(path)", file: file, line: line) }
            else { XCTAssertEqual(x, y, "at \(path)", file: file, line: line) }
        default: XCTAssertEqual(a,b,"at \(path)",file: file,line: line)
        }
    }
    func testBatchPaddingMatchesSingleInference() throws {
        let model = try model(), texts = ["Alice works for Acme in London.", "Tim Cook visited Berlin yesterday for a technology conference.", "Microsoft hired Bob."]
        var schema = Schema(entities: ["person", "company", "location"])
        schema.classifications = [ClassificationSpec(task: "topic", labels: ["business", "sports", "weather"])]
        let options = ExtractOptions(threshold: 0.5, includeSpans: true, includeConfidence: true)
        let batch = try model.extractBatch(texts, schema: schema, options: options)
        XCTAssertEqual(batch.count, texts.count)
        for (text,result) in zip(texts,batch) { assertJSON(result, try model.extract(text, schema: schema, options: options), tolerance: 1e-4) }
    }
    func testBudgetAndInvalidSchemas() throws {
        let model = try model(), schema = Schema(entities: ["person"])
        let oversized = Array(repeating: "Alice", count: model.maxInputTokens + 32).joined(separator: " ")
        XCTAssertThrowsError(try model.prepare(oversized, schema: schema)) { error in
            guard case GLiNERError.invalidInput = error else { return XCTFail("Unexpected error: \(error)") }
        }
        XCTAssertThrowsError(try model.extract("Hello", schema: Schema()))
        var emptyLabels = Schema(); emptyLabels.classifications = [ClassificationSpec(task: "topic", labels: [])]
        XCTAssertThrowsError(try model.extract("Hello", schema: emptyLabels))
    }
    func testLongDocumentOffsetsAndStripping() throws {
        let model = try model(), schema = Schema(entities: ["person", "company", "location"])
        let text = "Zoë met Alice at Microsoft in London. Tim Cook visited Apple in Berlin. Zoë met Alice at Microsoft in London."
        let chunks = ChunkOptions(chunkSize: 9, chunkOverlap: 3)
        let rich = try model.extractLong(text, schema: schema, options: .init(threshold: 0.3, includeSpans: true, includeConfidence: true), chunks: chunks)
        let scalars = Array(text.unicodeScalars)
        func check(_ value: JSONValue) {
            switch value {
            case .array(let items): items.forEach(check)
            case .object(let map):
                if case .string(let surface)? = map["text"], case .number(let a)? = map["start"], case .number(let b)? = map["end"] {
                    let start = Int(a), end = Int(b)
                    XCTAssertTrue(start >= 0 && start <= end && end <= scalars.count)
                    if start >= 0 && start <= end && end <= scalars.count { XCTAssertEqual(surface, String(String.UnicodeScalarView(scalars[start..<end]))) }
                }
                map.values.forEach(check)
            default: break
            }
        }
        check(rich)
        let plain = try model.extractLong(text, schema: schema, options: .init(threshold: 0.3), chunks: chunks)
        XCTAssertEqual(plain, stripSpanMetadata(rich, includeConfidence: false, includeSpans: false))
    }
    func testScalarEntityAndValidator() throws {
        let model = try model()
        var schema = Schema()
        schema.entities = [EntitySpec(name: "person", dtype: .str, validators: [RegexValidator(pattern: "Alice", mode: .full)])]
        let result = try model.extract("Alice met Bob in London.", schema: schema,
                                       options: .init(threshold: 0.1, includeSpans: true, includeConfidence: true))
        guard let person = result["entities"]?["person"] else { return XCTFail("Missing scalar entity output") }
        switch person {
        case .null: break
        case .object(let object): XCTAssertEqual(object["text"], .string("Alice"))
        default: XCTFail("A scalar entity must be null or a span object")
        }
    }
    func testHalfPrecisionAgainstFloat() throws {
        let full = try model(.float32), half = try model(.float16)
        var schema = Schema(entities: ["person", "company"])
        schema.classifications = [ClassificationSpec(task: "topic", labels: ["technology", "sports", "cooking"])]
        let options = ExtractOptions(threshold: 0.8, includeSpans: true, includeConfidence: true)
        for text in ["Apple CEO Tim Cook announced a new iPhone.", "The basketball team won the championship game."] {
            let expected = try full.extract(text, schema: schema, options: options)
            let actual = try half.extract(text, schema: schema, options: options)
            assertJSON(actual, expected, tolerance: 0.02)
        }
    }

    func testCacheControlsPreserveResultsAndTF32CannotSilentlyChange() throws {
        let model = try model(), original = model.runtimeConfiguration
        defer { try? model.updateRuntimeConfiguration(original) }
        let text = "Apple CEO Tim Cook visited London.", schema = Schema(entities: ["company", "person", "location"])
        let options = ExtractOptions(includeSpans: true, includeConfidence: true)
        let expected = try model.extract(text, schema: schema, options: options)
        let minimal = RuntimeConfiguration(allowTF32: false, cacheLimitBytes: 0, clearCacheAfterRequest: true)
        try model.updateRuntimeConfiguration(minimal)
        assertJSON(try model.extract(text, schema: schema, options: options), expected, tolerance: 1e-4)
        XCTAssertEqual(model.runtimeConfiguration, minimal)
        GLiNERRuntime.releaseUnusedMemory()
        XCTAssertEqual(GLiNERRuntime.memorySnapshot().cachedBytes, 0)
        var changed = minimal; changed.allowTF32 = true
        XCTAssertThrowsError(try model.updateRuntimeConfiguration(changed)) { error in
            guard case GLiNERError.invalidConfiguration = error else { return XCTFail("Unexpected error: \(error)") }
        }
        XCTAssertEqual(model.runtimeConfiguration, minimal, "A rejected precision change must preserve the effective settings")
        assertJSON(try model.extract(text, schema: schema, options: options), expected, tolerance: 1e-4)
    }
}
