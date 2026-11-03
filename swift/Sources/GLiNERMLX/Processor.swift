import Foundation

/// Unicode default lowercase, including contextual final Greek sigma, independent of user locale.
public func rustLowercased(_ text: String) -> String {
    text.lowercased(with: Locale(identifier: "en_US_POSIX"))
}

public struct Word: Equatable {
    public let text: String
    public let byteStart: Int, byteEnd: Int, charStart: Int, charEnd: Int
    public init(text: String, byteStart: Int, byteEnd: Int, charStart: Int, charEnd: Int) {
        self.text = text; self.byteStart = byteStart; self.byteEnd = byteEnd
        self.charStart = charStart; self.charEnd = charEnd
    }
}
extension WordSplitter {
    public func split(_ text: String, lower: Bool = false) -> [Word] {
        let pattern: String
        switch self {
        case .whitespace: pattern = #"(?i)(?:https?://[^\s]+|www\.[^\s]+)|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}|@[a-z0-9_]+|[\p{L}\p{N}_]+(?:[\-_][\p{L}\p{N}_]+)*|\S"#
        case .char: pattern = #"[A-Za-z0-9@._\-+]+|\S"#
        }
        let regex = try! NSRegularExpression(pattern: pattern)
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { match in
            guard let range = Range(match.range, in: text) else { return nil }
            let before = text[..<range.lowerBound], through = text[..<range.upperBound]
            let value = String(text[range])
            return Word(text: lower ? rustLowercased(value) : value, byteStart: before.utf8.count,
                        byteEnd: through.utf8.count, charStart: before.unicodeScalars.count,
                        charEnd: through.unicodeScalars.count)
        }
    }
}
public enum TaskKind { case structures, entities, relations, classifications }
public struct PromptGroup {
    public let kind: TaskKind
    public let specIndex: Int
    public let name: String
    public let fields: [String]
    public let tokens: [String]
    public var markerPositions: [Int]
    public var promptPosition: Int
}
public struct PreparedInput {
    public let text: String
    public let inputIds: [Int32]
    public let prefixTokens: [String]
    public let words: [Word]
    public let wordPositions: [Int]
    public let groups: [PromptGroup]
}
public struct Processor {
    public let tokenizer: UnigramTokenizer
    public var wordSplitter: WordSplitter = .whitespace
    public init(tokenizer: UnigramTokenizer) { self.tokenizer = tokenizer }
    private func transform(_ name: String, _ fields: [String], _ marker: String, prompt: String? = nil,
                           descriptions: [(String,String)] = [], examples: [(String,String)] = []) -> [String] {
        var title = prompt.map { "\(name): \($0)" } ?? name
        for (label, desc) in descriptions where fields.contains(label) { title += " [DESCRIPTION] \(label): \(desc)" }
        for (input, output) in examples where fields.contains(output) { title += " [EXAMPLE] \(input) [OUTPUT] \(output)" }
        return ["(", "[P]", title, "("] + fields.flatMap { [marker, $0] } + [")", ")"]
    }
    public func prepare(text original: String, schema: Schema, maxWords: Int? = nil) throws -> PreparedInput {
        let text = original.isEmpty ? "." : (original.hasSuffix(".") || original.hasSuffix("!") || original.hasSuffix("?") ? original : original + ".")
        var words = wordSplitter.split(text, lower: true)
        if let maxWords = maxWords { guard maxWords >= 0 else { throw GLiNERError.invalidInput("maxWords must be nonnegative") }; words = Array(words.prefix(maxWords)) }
        var groups: [PromptGroup] = []
        func append(_ kind: TaskKind, _ index: Int, _ name: String, _ fields: [String], _ tokens: [String]) {
            groups.append(PromptGroup(kind: kind, specIndex: index, name: name, fields: fields, tokens: tokens, markerPositions: [], promptPosition: 0))
        }
        var seen = Set<String>()
        for (i, s) in schema.structures.enumerated() where seen.insert(s.name).inserted {
            let fields = s.fields.map(\.name), desc = s.fields.compactMap { f in f.description.map { (f.name, $0) } }
            append(.structures, i, s.name, fields, transform(s.name, fields, "[C]", descriptions: desc))
        }
        if !schema.entities.isEmpty {
            let fields = schema.entities.map(\.name), desc = schema.entities.compactMap { e in e.description.map { (e.name, $0) } }
            append(.entities, 0, "entities", fields, transform("entities", fields, "[E]", descriptions: desc))
        }
        seen.removeAll()
        for (i, r) in schema.relations.enumerated() where seen.insert(r.name).inserted {
            let fields = ["head", "tail"]
            append(.relations, i, r.name, fields, transform(r.name, fields, "[R]", prompt: r.description))
        }
        for (i, c) in schema.classifications.enumerated() {
            append(.classifications, i, c.task, c.labels, transform(c.task, c.labels, "[L]", prompt: c.prompt,
                descriptions: c.labelDescriptions.map { ($0.label, $0.description) }, examples: c.examples.map { ($0.input, $0.output) }))
        }
        var prefix: [String] = []
        for s in schema.structures {
            var inner: [String] = []
            for f in s.fields {
                guard let choices = f.choices else { continue }
                inner += [f.name, "("]
                for (i, c) in choices.enumerated() { if i > 0 { inner.append("|") }; inner.append(c) }
                inner += [")", ","]
            }
            if !inner.isEmpty { inner.removeLast(); prefix += ["(", "\(s.name):"] + inner + [")"] }
        }
        var ids: [Int32] = [], positions: [Int] = []
        for gi in groups.indices {
            for (ti, token) in groups[gi].tokens.enumerated() {
                if ti == 1 { groups[gi].promptPosition = ids.count }
                else if ti >= 4 && ti < groups[gi].tokens.count - 2 && ti % 2 == 0 { groups[gi].markerPositions.append(ids.count) }
                ids += try tokenizer.encode(token)
            }
            if gi + 1 < groups.count { ids += try tokenizer.encode("[SEP_STRUCT]") }
        }
        ids += try tokenizer.encode("[SEP_TEXT]")
        for token in prefix + words.map(\.text) { positions.append(ids.count); ids += try tokenizer.encode(token) }
        return PreparedInput(text: text, inputIds: ids, prefixTokens: prefix, words: words, wordPositions: positions, groups: groups)
    }
}
