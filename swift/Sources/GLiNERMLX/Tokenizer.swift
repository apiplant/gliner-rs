import Foundation

/// Native Hugging Face Unigram tokenizer. Unsupported pipeline components fail at load time.
public final class UnigramTokenizer {
    private struct Piece { let text: String; let score: Double }
    private final class Node { var children: [UInt8: Node] = [:]; var id: Int? }
    private struct Added { let content: String; let id: Int32; let singleWord: Bool; let left: Bool; let right: Bool; let normalized: Bool }
    private let pieces: [Piece], root = Node(), unknown: Int?, byteFallback: Bool
    private let ids: [String: Int32], added: [Added]
    private let normalizer: [String: Any]?, pretokenizer: [String: Any]?
    private let minScore: Double

    public convenience init(url: URL) throws { try self.init(data: Data(contentsOf: url)) }
    public convenience init(path: String) throws { try self.init(url: URL(fileURLWithPath: path)) }
    public init(data: Data) throws {
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = obj["model"] as? [String: Any], model["type"] as? String == "Unigram",
              let vocab = model["vocab"] as? [[Any]], !vocab.isEmpty else {
            throw GLiNERError.unsupported("Expected a Hugging Face Unigram tokenizer")
        }
        var p: [Piece] = [], lookup: [String: Int32] = [:]
        for (i, row) in vocab.enumerated() {
            guard row.count == 2, let token = row[0] as? String, let score = row[1] as? NSNumber else { throw GLiNERError.tokenization("Invalid Unigram vocabulary") }
            p.append(Piece(text: token, score: score.doubleValue)); lookup[token] = Int32(i)
        }
        pieces = p; ids = lookup; minScore = p.map(\.score).min()!
        unknown = (model["unk_id"] as? NSNumber)?.intValue
        if let unknown = unknown, !(0..<p.count).contains(unknown) { throw GLiNERError.tokenization("Invalid unknown token id") }
        byteFallback = model["byte_fallback"] as? Bool ?? false
        normalizer = obj["normalizer"] as? [String: Any]
        let rawPre = obj["pre_tokenizer"] as? [String: Any]
        if rawPre?["type"] as? String == "Sequence" {
            guard let components = rawPre?["pretokenizers"] as? [[String: Any]], components.count == 1 else { throw GLiNERError.unsupported("Unsupported pre-tokenizer sequence") }
            pretokenizer = components[0]
        } else { pretokenizer = rawPre }
        added = (obj["added_tokens"] as? [[String: Any]] ?? []).compactMap { a in
            guard let content = a["content"] as? String, !content.isEmpty, let id = (a["id"] as? NSNumber)?.int32Value else { return nil }
            return Added(content: content, id: id, singleWord: a["single_word"] as? Bool ?? false,
                         left: a["lstrip"] as? Bool ?? false, right: a["rstrip"] as? Bool ?? false,
                         normalized: a["normalized"] as? Bool ?? true)
        }
        try Self.validateNormalizer(normalizer)
        if let pt = pretokenizer {
            guard pt["type"] as? String == "Metaspace", (pt["replacement"] as? String ?? "▁").unicodeScalars.count == 1,
                  ["always", "first", "never"].contains(pt["prepend_scheme"] as? String ?? "always") else {
                throw GLiNERError.unsupported("Unsupported pre-tokenizer")
            }
        }
        for (id, piece) in p.enumerated() where !piece.text.isEmpty {
            var node = root
            for b in piece.text.utf8 { if node.children[b] == nil { node.children[b] = Node() }; node = node.children[b]! }
            node.id = id
        }
    }
    private static func validateNormalizer(_ n: [String: Any]?) throws {
        guard let n = n else { return }
        switch n["type"] as? String {
        case "Sequence":
            guard let children = n["normalizers"] as? [[String: Any]] else { throw GLiNERError.unsupported("Invalid normalizer sequence") }
            for c in children { try validateNormalizer(c) }
        case "Replace":
            guard let pattern = n["pattern"] as? [String: String], pattern["String"] != nil || pattern["Regex"] != nil,
                  n["content"] is String else { throw GLiNERError.unsupported("Unsupported Replace normalizer") }
            if let regex = pattern["Regex"] { _ = try NSRegularExpression(pattern: regex) }
        case "NFC", "Strip": break
        default: throw GLiNERError.unsupported("Unsupported normalizer: \(n["type"] ?? "unknown")")
        }
    }
    private func normalize(_ input: String, _ n: [String: Any]?) throws -> String {
        guard let n = n else { return input }
        switch n["type"] as? String {
        case "Sequence": return try (n["normalizers"] as! [[String: Any]]).reduce(input) { try normalize($0, $1) }
        case "NFC": return input.precomposedStringWithCanonicalMapping
        case "Strip":
            var s = input
            if n["strip_left"] as? Bool ?? true { while let c = s.unicodeScalars.first, CharacterSet.whitespacesAndNewlines.contains(c) { s.removeSubrange(s.startIndex..<cEnd(s)) } }
            if n["strip_right"] as? Bool ?? true { while let c = s.unicodeScalars.last, CharacterSet.whitespacesAndNewlines.contains(c) { s.removeSubrange(s.unicodeScalars.index(before: s.endIndex)..<s.endIndex) } }
            return s
        case "Replace":
            let p = n["pattern"] as! [String: String], replacement = n["content"] as! String
            if let literal = p["String"] { return input.replacingOccurrences(of: literal, with: replacement) }
            let regex = try NSRegularExpression(pattern: p["Regex"]!)
            // tokenizers' Replace uses literal replacement text, not capture expansion.
            let matches = regex.matches(in: input, range: NSRange(input.startIndex..., in: input))
            var output = input
            for m in matches.reversed() { if let r = Range(m.range, in: output) { output.replaceSubrange(r, with: replacement) } }
            return output
        default: return input
        }
    }
    private func cEnd(_ s: String) -> String.Index { s.unicodeScalars.index(after: s.startIndex) }
    public func encode(_ text: String) throws -> [Int32] {
        // Extract unnormalized added tokens, normalize the rest, then extract normalized tokens.
        return try extract(text, candidates: added.filter { !$0.normalized }) { segment in
            let normalized = try self.normalize(segment, self.normalizer)
            return try self.extract(normalized, candidates: self.added.filter { $0.normalized }) { try self.encodeOrdinary($0) }
        }
    }
    private func extract(_ input: String, candidates: [Added], ordinary: (String) throws -> [Int32]) throws -> [Int32] {
        var out: [Int32] = [], cursor = input.startIndex
        while cursor < input.endIndex {
            var best: (Range<String.Index>, Added)?
            for token in candidates {
                var search = cursor..<input.endIndex
                while let r = input.range(of: token.content, options: .literal, range: search) {
                    func isWord(_ c: Character) -> Bool { c == "_" || c.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) } }
                    let validLeft = r.lowerBound == input.startIndex || !isWord(input[input.index(before: r.lowerBound)])
                    let validRight = r.upperBound == input.endIndex || !isWord(input[r.upperBound])
                    if !token.singleWord || (validLeft && validRight) {
                        if best == nil || r.lowerBound < best!.0.lowerBound || (r.lowerBound == best!.0.lowerBound && token.content.utf8.count > best!.1.content.utf8.count) { best = (r, token) }
                        break
                    }
                    let next = input.index(after: r.lowerBound); if next >= input.endIndex { break }; search = next..<input.endIndex
                }
            }
            guard let (range, token) = best else { out += try ordinary(String(input[cursor...])); break }
            var start = range.lowerBound, end = range.upperBound
            if token.left { while start > cursor { let prev = input.index(before: start); if !input[prev].isWhitespace { break }; start = prev } }
            if token.right { while end < input.endIndex && input[end].isWhitespace { end = input.index(after: end) } }
            if cursor < start { out += try ordinary(String(input[cursor..<start])) }
            out.append(token.id); cursor = end
        }
        return out
    }
    private func encodeOrdinary(_ text: String) throws -> [Int32] {
        guard !text.isEmpty else { return [] }
        guard let pt = pretokenizer else { return try viterbi(text) }
        let replacement = pt["replacement"] as? String ?? "▁", scheme = pt["prepend_scheme"] as? String ?? "always"
        var s = text.replacingOccurrences(of: " ", with: replacement)
        if scheme != "never" && !s.hasPrefix(replacement) { s = replacement + s }
        guard pt["split"] as? Bool ?? true else { return try viterbi(s) }
        var chunks: [String] = [], current = ""
        for c in s { if String(c) == replacement && !current.isEmpty { chunks.append(current); current = "" }; current.append(c) }
        if !current.isEmpty { chunks.append(current) }
        return try chunks.flatMap { try viterbi($0) }
    }
    private func viterbi(_ text: String) throws -> [Int32] {
        let bytes = Array(text.utf8), n = bytes.count
        struct Best { var score = -Double.infinity; var prev = 0; var id = -1 }
        var best = Array(repeating: Best(), count: n + 1); best[0].score = 0
        var starts: [Int] = [], offset = 0
        for scalar in text.unicodeScalars { starts.append(offset); offset += scalar.utf8.count }
        for (si, start) in starts.enumerated() {
            guard best[start].score.isFinite else { continue }
            let next = si + 1 < starts.count ? starts[si + 1] : n
            var node = root, end = start, hasSingle = false
            while end < n, let child = node.children[bytes[end]] {
                node = child; end += 1
                if let id = node.id {
                    if end == next { hasSingle = true }
                    let score = best[start].score + pieces[id].score
                    if score > best[end].score { best[end] = Best(score: score, prev: start, id: id) }
                }
            }
            if !hasSingle {
                guard let id = unknown else { throw GLiNERError.tokenization("Unknown character without unk_id") }
                let score = best[start].score + minScore - 10
                if score > best[next].score { best[next] = Best(score: score, prev: start, id: id) }
            }
        }
        var path: [(Int, Int, Int)] = [], pos = n
        while pos > 0 { let b = best[pos]; guard b.id >= 0 else { throw GLiNERError.tokenization("Unigram path not found") }; path.append((b.prev, pos, b.id)); pos = b.prev }
        path.reverse(); var result: [Int32] = [], i = 0
        while i < path.count {
            let (start, _, id) = path[i]
            if id == unknown {
                var end = path[i].1; i += 1
                while i < path.count && path[i].2 == id { end = path[i].1; i += 1 }
                let fallback = bytes[start..<end].compactMap { ids[String(format: "<0x%02X>", $0)] }
                if byteFallback && fallback.count == end - start { result += fallback } else { result.append(Int32(id)) }
            } else { result.append(Int32(id)); i += 1 }
        }
        return result
    }
}
