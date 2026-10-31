import Foundation

/// JSON output with offsets measured in Unicode scalar values, as in Python and Rust.
public enum JSONValue: Codable, Equatable, Sendable {
    case object([String: JSONValue]), array([JSONValue]), string(String), number(Double), bool(Bool), null

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let value = try? c.decode(Bool.self) { self = .bool(value) }
        else if let value = try? c.decode(Double.self) { self = .number(value) }
        else if let value = try? c.decode(String.self) { self = .string(value) }
        else if let value = try? c.decode([JSONValue].self) { self = .array(value) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let x): try c.encode(x)
        case .number(let x): try c.encode(x)
        case .string(let x): try c.encode(x)
        case .array(let x): try c.encode(x)
        case .object(let x): try c.encode(x)
        }
    }
    public func prettyPrinted() throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(self), as: UTF8.self)
    }
    public subscript(_ key: String) -> JSONValue? {
        guard case .object(let values) = self else { return nil }; return values[key]
    }
    var isEmptyValue: Bool { if self == .null { return true }; if case .array(let a) = self { return a.isEmpty }; return false }
}

public struct ExtractOptions {
    public var threshold: Float
    public var includeSpans: Bool
    public var includeConfidence: Bool
    public var overlapPolicy: OverlapPolicy?
    public var maxWords: Int?
    public init(threshold: Float = 0.5, includeSpans: Bool = false, includeConfidence: Bool = false,
                overlapPolicy: OverlapPolicy? = nil, maxWords: Int? = nil) {
        self.threshold = threshold; self.includeSpans = includeSpans; self.includeConfidence = includeConfidence
        self.overlapPolicy = overlapPolicy; self.maxWords = maxWords
    }
}

public enum ModelPrecision: String, Codable, Sendable {
    case float32
    /// Approximate, opt-in encoder precision. Can change candidates and final answers.
    case float16
}

func scalarSigmoid(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }
func scalarSoftmax(_ xs: [Float]) -> [Float] {
    guard let maximum = xs.max() else { return [] }
    let exps = xs.map { Foundation.exp($0 - maximum) }, total = exps.reduce(Float(0), +)
    return exps.map { $0 / total }
}
func firstArgmax(_ xs: [Float]) -> Int {
    var best = 0
    for i in xs.indices where xs[i] > xs[best] { best = i }
    return best
}

struct Mention {
    var text: String; var score: Float; var start: Int; var end: Int
    func formatted(_ opts: ExtractOptions, attributes: [String: JSONValue] = [:]) -> JSONValue {
        if !opts.includeSpans && !opts.includeConfidence && attributes.isEmpty { return .string(text) }
        var object: [String: JSONValue] = ["text": .string(text)]
        if opts.includeSpans { object["start"] = .number(Double(start)); object["end"] = .number(Double(end)) }
        if opts.includeConfidence { object["confidence"] = .number(Double(score)) }
        object.merge(attributes) { _, new in new }
        return .object(object)
    }
}
func uniqueJSON(_ values: [JSONValue]) -> [JSONValue] {
    var seen = Set<String>(), out: [JSONValue] = []
    for value in values {
        let text: String, coordinates: String
        switch value {
        case .string(let s): text = s; coordinates = ""
        case .object(let object):
            if case .string(let s) = object["text"] { text = s } else { text = "" }
            if case .number(let start) = object["start"], case .number(let end) = object["end"] {
                coordinates = "\(start),\(end)"
            } else { coordinates = "" }
        default: out.append(value); continue
        }
        // Length-prefix the text to avoid ambiguous composite keys.
        let lower = rustLowercased(text), key = "\(lower.utf8.count):\(lower):\(coordinates)"
        if !text.isEmpty && seen.insert(key).inserted { out.append(value) }
    }
    return out
}
func finalizeMentions(_ values: [Mention], scalar: Bool, policy: OverlapPolicy) -> [Mention] {
    let retained = resolveOverlaps(values.map { ScoredSpan(start: $0.start, end: $0.end, score: $0.score) }, policy: policy)
    let selected = retained.compactMap { s in values.first { $0.start == s.start && $0.end == s.end && $0.score == s.score } }
    return scalar ? Array(selected.prefix(1)) : selected
}
func formatStructureField(_ mentions: [Mention], scalar: Bool, options: ExtractOptions) -> JSONValue {
    if scalar { return mentions.first?.formatted(options) ?? .null }
    return .array(uniqueJSON(mentions.map { $0.formatted(options) }))
}
