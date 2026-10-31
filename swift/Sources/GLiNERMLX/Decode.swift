import Foundation

public struct ScoredSpan: Equatable {
    public var start: Int, end: Int, score: Float
    public init(start: Int, end: Int, score: Float) { self.start = start; self.end = end; self.score = score }
}
public struct RelationPair {
    public var relation: Int, headStart: Int, headEnd: Int, tailStart: Int, tailEnd: Int, score: Float
    public var head: (Int, Int) { (headStart, headEnd) }
    public var tail: (Int, Int) { (tailStart, tailEnd) }
    public init(relation: Int = 0, headStart: Int, headEnd: Int, tailStart: Int, tailEnd: Int, score: Float = 0) {
        self.relation = relation; self.headStart = headStart; self.headEnd = headEnd; self.tailStart = tailStart; self.tailEnd = tailEnd; self.score = score
    }
    public init(relation: Int, head: (Int, Int), tail: (Int, Int)) {
        self.init(relation: relation, headStart: head.0, headEnd: head.1, tailStart: tail.0, tailEnd: tail.1)
    }
}
public struct RelationEdge {
    public var headStart: Int, headEnd: Int, tailStart: Int, tailEnd: Int, score: Float
    public var headText: String, tailText: String
    public init(headStart: Int, headEnd: Int, tailStart: Int, tailEnd: Int, score: Float, headText: String = "", tailText: String = "") {
        self.headStart = headStart; self.headEnd = headEnd; self.tailStart = tailStart; self.tailEnd = tailEnd; self.score = score; self.headText = headText; self.tailText = tailText
    }
}
private struct Ranked { let index: Int; let span: ScoredSpan }
private func rankLess(_ a: Ranked, _ b: Ranked) -> Bool {
    if a.span.score != b.span.score { return a.span.score > b.span.score }
    if a.span.start != b.span.start { return a.span.start < b.span.start }
    if a.span.end != b.span.end { return a.span.end < b.span.end }
    return a.index < b.index
}
public func resolveOverlaps(_ items: [ScoredSpan], policy: OverlapPolicy) -> [ScoredSpan] {
    var seen = Set<[Int]>()
    let distinct = items.enumerated().map { Ranked(index: $0.offset, span: $0.element) }.sorted(by: rankLess).filter { seen.insert([$0.span.start, $0.span.end]).inserted }
    switch policy {
    case .allow: return distinct.map(\.span)
    case .nested:
        var kept: [ScoredSpan] = []
        for item in distinct {
            let c = item.span
            if !kept.contains(where: { e in c.start < e.end && e.start < c.end && !((c.start <= e.start && e.end <= c.end) || (e.start <= c.start && c.end <= e.end)) }) { kept.append(c) }
        }
        return kept
    case .longest:
        return distinct.filter { c in !distinct.contains { o in o.span.start <= c.span.start && c.span.end <= o.span.end && (o.span.start < c.span.start || c.span.end < o.span.end) } }.map(\.span)
    case .flat:
        let byEnd = distinct.sorted { a, b in
            if a.span.end != b.span.end { return a.span.end < b.span.end }
            if a.span.start != b.span.start { return a.span.start < b.span.start }
            return rankLess(a, b)
        }
        var best: [(Double, [Int])] = [(0, [])]
        for i in byEnd.indices {
            let c = byEnd[i].span
            var predecessor = 0
            while predecessor < i && byEnd[predecessor].span.end <= c.start { predecessor += 1 }
            let with = (best[predecessor].0 + Double(c.score), best[predecessor].1 + [i]), without = best[i]
            let a = with.1.map { byEnd[$0] }.sorted(by: rankLess), b = without.1.map { byEnd[$0] }.sorted(by: rankLess)
            var lex = false
            for (x, y) in zip(a, b) { if rankLess(x,y) { lex = true; break }; if rankLess(y,x) { break } }
            if with.0 > without.0 || (with.0 == without.0 && (with.1.count > without.1.count || (with.1.count == without.1.count && lex))) { best.append(with) } else { best.append(without) }
        }
        return best.last!.1.map { byEnd[$0] }.sorted(by: rankLess).map(\.span)
    }
}
public func proposeRelationPairs(heads: [ScoredSpan], tails: [ScoredSpan], headsPerType: Int, tailsPerType: Int, pairCap: Int, proposalThreshold: Float) -> [RelationPair] {
    func select(_ spans: [ScoredSpan], _ count: Int) -> [ScoredSpan] {
        Array(spans.enumerated().filter { $0.element.score >= proposalThreshold }.sorted {
            if $0.element.score != $1.element.score { return $0.element.score > $1.element.score }
            if $0.element.start != $1.element.start { return $0.element.start < $1.element.start }
            if $0.element.end != $1.element.end { return $0.element.end < $1.element.end }
            return $0.offset < $1.offset
        }.prefix(max(0, count)).map(\.element))
    }
    var pairs: [(Int, RelationPair)] = []
    for h in select(heads, headsPerType) { for t in select(tails, tailsPerType) where h.start != t.start || h.end != t.end {
        pairs.append((pairs.count, RelationPair(headStart: h.start, headEnd: h.end, tailStart: t.start, tailEnd: t.end, score: h.score * t.score)))
    } }
    return pairs.sorted { $0.1.score == $1.1.score ? $0.0 < $1.0 : $0.1.score > $1.1.score }.prefix(max(0, pairCap)).map(\.1)
}

/// Collapse contained mentions and repeated surface-text cross products.
public func deduplicateRelationEdges(_ edges: [RelationEdge]) -> [RelationEdge] {
    guard edges.count >= 2 else { return edges }
    struct Mention { var text: String; var start: Int; var end: Int }
    func canonical(_ head: Bool) -> [[Int]: Mention] {
        var order: [[Int]] = [], mentions: [[Int]: Mention] = [:]
        for edge in edges {
            let m = head ? Mention(text: edge.headText, start: edge.headStart, end: edge.headEnd) : Mention(text: edge.tailText, start: edge.tailStart, end: edge.tailEnd)
            let key = [m.start, m.end]; if mentions[key] == nil { order.append(key) }; mentions[key] = m
        }
        var result: [[Int]: Mention] = [:]
        for key in order {
            var best: Mention?
            for k in order {
                let m = mentions[k]!
                if m.start <= key[0] && m.end >= key[1] && (best == nil || m.end - m.start > best!.end - best!.start || (m.end - m.start == best!.end - best!.start && m.start < best!.start)) { best = m }
            }
            result[key] = best!
        }
        return result
    }
    let heads = canonical(true), tails = canonical(false)
    var order: [[Int]] = [], exact: [[Int]: RelationEdge] = [:]
    for e in edges {
        let h = heads[[e.headStart,e.headEnd]]!, t = tails[[e.tailStart,e.tailEnd]]!
        let key = [h.start,h.end,t.start,t.end]
        let value = RelationEdge(headStart: h.start, headEnd: h.end, tailStart: t.start, tailEnd: t.end, score: e.score, headText: h.text, tailText: t.text)
        if exact[key] == nil { order.append(key); exact[key] = value } else if e.score > exact[key]!.score { exact[key] = value }
    }
    func norm(_ s: String) -> String { rustLowercased(s).split(whereSeparator: { $0.isWhitespace }).joined(separator: " ") }
    func rankLess(_ a: RelationEdge, _ b: RelationEdge) -> Bool {
        let ad = max(0, a.headStart - a.tailEnd, a.tailStart - a.headEnd), bd = max(0, b.headStart - b.tailEnd, b.tailStart - b.headEnd)
        if ad != bd { return ad < bd }; if a.score != b.score { return a.score > b.score }
        if a.headStart != b.headStart { return a.headStart < b.headStart }; return a.tailStart < b.tailStart
    }
    var semanticOrder: [[String]] = [], semantic: [[String]: RelationEdge] = [:]
    for k in order {
        let e = exact[k]!, key = [norm(e.headText), norm(e.tailText)]
        if semantic[key] == nil { semanticOrder.append(key); semantic[key] = e }
        else if rankLess(e, semantic[key]!) { semantic[key] = e }
    }
    let values = semanticOrder.map { semantic[$0]! }
    func tokens(_ s: String) -> Set<String> { Set(norm(s).components(separatedBy: " ")) }
    return values.enumerated().filter { i, e in
        let h = tokens(e.headText), t = tokens(e.tailText)
        return !values.enumerated().contains { j, o in
            if i == j { return false }; let oh = tokens(o.headText), ot = tokens(o.tailText)
            return (h.count < oh.count && h.isSubset(of: oh) && t == ot) || (t.count < ot.count && t.isSubset(of: ot) && h == oh)
        }
    }.map(\.element).sorted { a,b in
        if a.headStart != b.headStart { return a.headStart < b.headStart }; if a.tailStart != b.tailStart { return a.tailStart < b.tailStart }; return a.score > b.score
    }
}
