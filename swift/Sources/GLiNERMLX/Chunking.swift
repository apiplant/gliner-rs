import Foundation

public struct ChunkOptions {
    public var chunkSize: Int, chunkOverlap: Int
    public init(chunkSize: Int = 384, chunkOverlap: Int = 64) { self.chunkSize = chunkSize; self.chunkOverlap = chunkOverlap }
}
public struct TextChunk: Equatable {
    public var text: String, startChar: Int
    public init(text: String, startChar: Int) { self.text = text; self.startChar = startChar }
}
public func splitTextIntoChunks(text: String, options: ChunkOptions = .init(), splitter: WordSplitter = .whitespace) throws -> [TextChunk] {
    guard options.chunkSize > 0 else { throw GLiNERError.invalidInput("chunkSize must be greater than 0") }
    guard options.chunkOverlap >= 0 && options.chunkOverlap < options.chunkSize else { throw GLiNERError.invalidInput("chunkOverlap must be nonnegative and smaller than chunkSize") }
    let words = splitter.split(text, lower: false), bytes = Array(text.utf8)
    guard !words.isEmpty else { return [TextChunk(text: text, startChar: 0)] }
    var out: [TextChunk] = [], start = 0
    while start < words.count {
        let end = min(words.count, start + min(options.chunkSize, words.count - start))
        out.append(TextChunk(text: String(decoding: bytes[words[start].byteStart..<words[end-1].byteEnd], as: UTF8.self), startChar: words[start].charStart))
        if end == words.count { break }; start += options.chunkSize - options.chunkOverlap
    }
    return out
}
private func object(_ v: JSONValue) -> [String: JSONValue]? { if case .object(let x) = v { return x }; return nil }
private func array(_ v: JSONValue) -> [JSONValue]? { if case .array(let x) = v { return x }; return nil }
private func string(_ v: JSONValue?) -> String? { if case .string(let x)? = v { return x }; return nil }
private func number(_ v: JSONValue?) -> Double? { if case .number(let x)? = v { return x }; return nil }
private func span(_ v: JSONValue) -> (Int,Int)? {
    guard let m = object(v), m["text"] != nil, let s = number(m["start"]), let e = number(m["end"]), s >= 0, e >= 0,
          s.isFinite, e.isFinite, s.rounded(.towardZero) == s, e.rounded(.towardZero) == e, s < Double(Int.max), e < Double(Int.max) else { return nil }
    return (Int(s),Int(e))
}
public func remapChunkSpans(_ value: JSONValue, startChar: Int, originalText: String) -> JSONValue {
    let scalars = Array(originalText.unicodeScalars)
    func visit(_ value: JSONValue) -> JSONValue {
        if case .array(let items) = value { return .array(items.map(visit)) }
        guard case .object(let map) = value else { return value }
        var out = map.mapValues(visit)
        if let (s,e) = span(.object(out)), s <= Int.max - startChar, e <= Int.max - startChar {
            let start = s + startChar, end = e + startChar
            out["start"] = .number(Double(start)); out["end"] = .number(Double(end))
            if start >= 0 && start <= end && end <= scalars.count { out["text"] = .string(String(String.UnicodeScalarView(scalars[start..<end]))) }
        }
        return .object(out)
    }
    return visit(value)
}
private func confidence(_ value: JSONValue) -> Double { number(value["confidence"]) ?? 0 }
private func representative(_ value: JSONValue) -> Double {
    switch value {
    case .object(let m): if let x = number(m["confidence"]) { return x }; return m.values.map(representative).max() ?? 0
    case .array(let a): return a.map(representative).max() ?? 0
    default: return 0
    }
}
private func canonical(_ value: JSONValue) -> String {
    switch value {
    case .object(let m): return "{" + m.keys.filter { $0 != "confidence" }.sorted().map { key in "\(key.debugDescription):\(canonical(m[key]!))" }.joined(separator: ",") + "}"
    case .array(let a): return "[" + a.map(canonical).joined(separator: ",") + "]"
    case .string(let s): return "s" + s.debugDescription
    case .number(let n): return "n\(n)"
    case .bool(let b): return "b\(b)"
    case .null: return "null"
    }
}
private func dedupOthers(_ items: [JSONValue]) -> [JSONValue] {
    var out: [JSONValue] = [], seen: [String:Int] = [:]
    for item in items { let key = canonical(item); if let i = seen[key] { if representative(item) > representative(out[i]) { out[i] = item } } else { seen[key] = out.count; out.append(item) } }
    return out
}
private func dedupSpans(_ items: [JSONValue], _ policy: OverlapPolicy) -> [JSONValue] {
    let valid = items.compactMap { item -> (JSONValue, ScoredSpan)? in guard let (s,e) = span(item) else { return nil }; return (item, ScoredSpan(start: s, end: e, score: Float(confidence(item)))) }
    let selected = resolveOverlaps(valid.map(\.1), policy: policy)
    var used = Set<Int>(), out: [JSONValue] = []
    for s in selected { if let i = valid.indices.first(where: { !used.contains($0) && valid[$0].1 == s }) { used.insert(i); out.append(valid[i].0) } }
    return out.sorted { a,b in let x = span(a)!, y = span(b)!; if x.0 != y.0 { return x.0 < y.0 }; if x.1 != y.1 { return x.1 < y.1 }; return (string(a["text"]) ?? "") < (string(b["text"]) ?? "") }
}
private func dedupItems(_ items: [JSONValue], _ policy: OverlapPolicy) -> [JSONValue] {
    dedupSpans(items.filter { span($0) != nil }, policy) + dedupOthers(items.filter { span($0) == nil })
}
private func isEmpty(_ v: JSONValue) -> Bool { if case .object(let m) = v { return m.isEmpty }; return v.isEmptyValue }
private func isClassification(_ v: JSONValue) -> Bool { object(v)?["label"] != nil && object(v)?["confidence"] != nil }
private func keys(_ values: [JSONValue]) -> [String] { Array(Set(values.flatMap { object($0)?.keys.map { $0 } ?? [] })).sorted() }
private func mergeValues(_ values: [JSONValue], _ policy: OverlapPolicy) -> JSONValue {
    let nonempty = values.filter { !isEmpty($0) }
    guard let first = nonempty.first else { return values.first ?? .null }
    if nonempty.allSatisfy(isClassification) {
        return nonempty.dropFirst().reduce(first) { confidence($1) >= confidence($0) ? $1 : $0 }
    }
    if nonempty.allSatisfy({ string($0) != nil }) {
        let strings = nonempty.compactMap { string($0) }, counts = Dictionary(strings.map { ($0,1) }, uniquingKeysWith: +)
        var best = strings[0]; for s in strings.dropFirst() where counts[s]! > counts[best]! { best = s }; return .string(best)
    }
    if nonempty.allSatisfy({ array($0) != nil }) { return .array(dedupItems(nonempty.flatMap { array($0)! }, policy)) }
    if nonempty.allSatisfy({ object($0) != nil }) {
        var result: [String: JSONValue] = [:]
        for k in keys(nonempty) { result[k] = mergeValues(nonempty.compactMap { $0[k] }, policy) }; return .object(result)
    }
    return first
}
private func asList(_ v: JSONValue) -> [JSONValue] { if v == .null { return [] }; return array(v) ?? [v] }
public func mergeChunkResults(_ results: [JSONValue], scalarEntityLabels: Set<String> = [], policy: OverlapPolicy = .flat) -> JSONValue {
    var merged: [String: JSONValue] = [:]
    for key in keys(results) {
        let values = results.compactMap { $0[key] }
        if key == "entities" || key == "relation_extraction" {
            var map: [String: JSONValue] = [:]
            for label in keys(values) {
                let items = values.compactMap { $0[label] }.flatMap(asList)
                if key == "entities" {
                    let selected = dedupSpans(items, policy)
                    map[label] = scalarEntityLabels.contains(label) ? (selected.first ?? .null) : .array(selected)
                } else { map[label] = .array(dedupItems(items, .allow)) }
            }
            merged[key] = .object(map)
        } else { merged[key] = mergeValues(values, policy) }
    }
    return .object(merged)
}
public func stripSpanMetadata(_ value: JSONValue, includeConfidence: Bool, includeSpans: Bool) -> JSONValue {
    if case .array(let a) = value { return .array(a.map { stripSpanMetadata($0, includeConfidence: includeConfidence, includeSpans: includeSpans) }) }
    guard case .object(let m) = value else { return value }
    if span(value) != nil {
        let extras = m.filter { !["text","confidence","start","end"].contains($0.key) }
        if !includeConfidence && !includeSpans && extras.isEmpty { return m["text"] ?? .string("") }
        var out = extras; out["text"] = m["text"] ?? .string("")
        if includeConfidence { out["confidence"] = m["confidence"] }; if includeSpans { out["start"] = m["start"]; out["end"] = m["end"] }
        return .object(out)
    }
    if isClassification(value) { return includeConfidence ? value : (m["label"] ?? .null) }
    if m["text"] != nil && m["confidence"] != nil && m["start"] == nil && m["end"] == nil { return includeConfidence ? value : (m["text"] ?? .null) }
    return .object(m.mapValues { stripSpanMetadata($0, includeConfidence: includeConfidence, includeSpans: includeSpans) })
}
extension GLiNERModel {
    public func extractLong(_ text: String, schema: Schema, options: ExtractOptions = .init(), chunks: ChunkOptions = .init()) throws -> JSONValue {
        let parts = try splitTextIntoChunks(text: text, options: chunks, splitter: wordSplitter)
        var inner = options; inner.includeConfidence = true; inner.includeSpans = true
        // A per-chunk maxWords would silently discard document text.
        if let maxWords = options.maxWords, parts.contains(where: { wordSplitter.split($0.text).count > maxWords }) { throw GLiNERError.invalidInput("maxWords would truncate a chunk; increase it or reduce chunkSize") }
        let results = try parts.map { chunk in remapChunkSpans(try extract(chunk.text, schema: schema, options: inner), startChar: chunk.startChar, originalText: text) }
        let scalar = Set(schema.entities.filter { $0.dtype == .str }.map(\.name))
        let policy = options.overlapPolicy ?? OverlapPolicy(rawValue: config.boundaryHead.overlapPolicy) ?? .flat
        return stripSpanMetadata(mergeChunkResults(results, scalarEntityLabels: scalar, policy: policy), includeConfidence: options.includeConfidence, includeSpans: options.includeSpans)
    }
}
