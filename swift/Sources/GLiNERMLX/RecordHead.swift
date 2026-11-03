import Foundation
import MLX

public struct RecordField {
    public var query: Int, cardinality: Cardinality, isAnchor: Bool, exclusive: Bool
    public init(query: Int, cardinality: Cardinality, isAnchor: Bool, exclusive: Bool) { self.query = query; self.cardinality = cardinality; self.isAnchor = isAnchor; self.exclusive = exclusive }
}
public enum RecordMode { case natural, latent, anchorless }
public struct DecodedRecord {
    public var fields: [Int: [(Int, Float)]], anchor: Int?, score: Float
    public init(fields: [Int: [(Int, Float)]], anchor: Int?, score: Float) { self.fields = fields; self.anchor = anchor; self.score = score }
}
public struct RecordHead {
    private let inst, field, cand, object, seed, q, k, v: TensorLinear
    private let nullEmbed, instanceEmbed: MLXArray
    private let recordDim: Int
    public init(weights: WeightStore, hidden: Int, recordDim: Int, instanceQueries: Int, prefix: String = "record_decoder") throws {
        let w = weights.prefixed(prefix)
        inst = try TensorLinear(w, "inst_proj", hidden, recordDim); field = try TensorLinear(w, "field_proj", hidden, recordDim)
        cand = try TensorLinear(w, "cand_proj", hidden, recordDim); object = try TensorLinear(w, "object_head", hidden, 1)
        seed = try TensorLinear(w, "latent_seed_head", hidden, 1); q = try TensorLinear(w, "q_proj", hidden, recordDim)
        k = try TensorLinear(w, "k_proj", hidden, recordDim); v = try TensorLinear(w, "v_proj", hidden, hidden)
        nullEmbed = try w.get("null_embed", shape: [recordDim]); instanceEmbed = try w.get("instance_embed", shape: [instanceQueries, hidden]); self.recordDim = recordDim
    }
    public func latentSeedScores(_ candidateStates: MLXArray) throws -> [Float] { seed(candidateStates).asType(.float32).asArray(Float.self) }
    public func objectScores(_ states: MLXArray) throws -> [Float] { object(states).asType(.float32).asArray(Float.self) }
    public func anchorlessInstances(_ candidateStates: MLXArray) throws -> MLXArray {
        if candidateStates.dim(0) == 0 { return instanceEmbed }
        let attention = matmul(q(instanceEmbed), k(candidateStates).transposed()) / Float(recordDim).squareRoot()
        return instanceEmbed + matmul(softmax(attention, axis: 1), v(candidateStates))
    }
    public func assignLogits(instanceStates: MLXArray, candidateStates: MLXArray, fieldQueries: MLXArray) throws -> [[[Float]]] {
        let query = inst(instanceStates).expandedDimensions(axis: 0) + field(fieldQueries).expandedDimensions(axis: 1)
        let absent = matmul(query, nullEmbed.expandedDimensions(axis: 1))
        let scores = matmul(query, cand(candidateStates).transposed())
        let result = concatenated([absent, scores], axis: 2).asType(.float32)
        let raw = result.asArray(Float.self), instances = result.dim(1), columns = result.dim(2)
        return (0..<result.dim(0)).map { f in (0..<instances).map { i in Array(raw[((f * instances + i) * columns)..<((f * instances + i + 1) * columns)]) } }
    }
}
private func recordSigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
private func recordSoftmax(_ x: [Float]) -> [Float] {
    guard let m = x.max() else { return [] }; let e = x.map { exp($0 - m) }, sum = e.reduce(0,+); return e.map { $0 / sum }
}
private func scalar(_ c: Cardinality) -> Bool { c == .optionalOne || c == .requiredOne }
private func absent(_ c: Cardinality) -> Bool { c == .optionalOne || c == .zeroOrMore }
public func linearSumAssignment(_ cost: [[Double]]) -> [Int] {
    let n = cost.count; guard n > 0 else { return [] }; let m = cost[0].count
    precondition(n <= m && cost.allSatisfy { $0.count == m })
    let epsilon = Double.ulpOfOne * max(1, cost.flatMap { $0 }.map { abs($0) }.max() ?? 0)
    var u = Array(repeating: 0.0, count: n+1), v = Array(repeating: 0.0, count: m+1), p = Array(repeating: 0, count: m+1), way = p
    for i in 1...n {
        p[0] = i; var j0 = 0, minv = Array(repeating: Double.infinity, count: m+1), used = Array(repeating: false, count: m+1)
        repeat {
            used[j0] = true; let i0 = p[j0]; var delta = Double.infinity, j1 = 0
            for j in 1...m where !used[j] {
                let cur = cost[i0-1][j-1] + epsilon * Double((i0-1)*m+j-1) - u[i0] - v[j]
                if cur < minv[j] { minv[j] = cur; way[j] = j0 }; if minv[j] < delta { delta = minv[j]; j1 = j }
            }
            for j in 0...m { if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta } }
            j0 = j1
        } while p[j0] != 0
        repeat { let j1 = way[j0]; p[j0] = p[j1]; j0 = j1 } while j0 != 0
    }
    var result = Array(repeating: 0, count: n); for j in 1...m where p[j] != 0 { result[p[j]-1] = j-1 }; return result
}
public func decodeRecordGroup(mode: RecordMode, fields: [RecordField], objectLogits: [Float], spans: [CandidateSpan]?, assign: [[[Float]]], threshold: Float, temperature: Float) -> [DecodedRecord] {
    let probabilities = objectLogits.map { recordSigmoid($0 / temperature) }
    let selected = probabilities.indices.filter { probabilities[$0] >= threshold }.sorted { probabilities[$0] == probabilities[$1] ? $0 < $1 : probabilities[$0] > probabilities[$1] }
    let anchorField = mode == .natural ? fields.firstIndex(where: { $0.isAnchor }) : nil
    var scalarChoices: [[Int]: (Int, Float)] = [:], owners: [[Int]: (Int, Float)] = [:]
    for (f, field) in fields.enumerated() where field.exclusive && !selected.isEmpty && f != anchorField {
        let rows = selected.map { assign[f][$0].map { $0 / temperature } }, count = max(0, rows[0].count - 1)
        if scalar(field.cardinality) {
            if count == 0 { continue }
            let probs = rows.map(recordSoftmax)
            let costs = probs.map { $0.dropFirst().map { -log(max($0, Float.ulpOfOne)) } }
            let maxCost = costs.flatMap { $0 }.max()!
            let diagonal = probs.map { absent(field.cardinality) ? -log(max($0[0], Float.ulpOfOne)) : maxCost + 50 }
            let invalid = max(maxCost, diagonal.max()!) + 1000
            let matrix = costs.indices.map { r in costs[r].map(Double.init) + diagonal.indices.map { Double($0 == r ? diagonal[r] : invalid) } }
            let matching = linearSumAssignment(matrix)
            for (r, inst) in selected.enumerated() {
                let col = matching[r]; if col < count && (probs[r][col+1] >= threshold || !absent(field.cardinality)) { scalarChoices[[inst,f]] = (col,probs[r][col+1]) }
            }
        } else {
            for c in 0..<count {
                var best = recordSigmoid(rows[0][c+1]), row = 0
                for i in rows.indices.dropFirst() { let p = recordSigmoid(rows[i][c+1]); if p > best { best = p; row = i } }
                if best >= threshold { owners[[f,c]] = (selected[row],best) }
            }
        }
    }
    var records: [DecodedRecord] = []
    for instance in selected {
        var rec = DecodedRecord(fields: [:], anchor: mode == .anchorless ? nil : instance, score: probabilities[instance])
        for (f, field) in fields.enumerated() {
            if f == anchorField { rec.fields[f] = [(instance, rec.score)]; continue }
            let row = assign[f][instance].map { $0 / temperature }
            if scalar(field.cardinality) {
                if field.exclusive { if let choice = scalarChoices[[instance,f]] { rec.fields[f] = [choice] }; continue }
                let probs = recordSoftmax(row)
                let order = probs.indices.sorted { probs[$0] == probs[$1] ? $0 < $1 : probs[$0] > probs[$1] }
                guard let col = order.first(where: { $0 != 0 || absent(field.cardinality) }), col != 0 else { continue }
                if probs[col] >= threshold || !absent(field.cardinality) { rec.fields[f] = [(col-1,probs[col])] }
            } else {
                var picked: [(Int,Float)] = []
                for c in 0..<max(0,row.count-1) {
                    if field.exclusive { if let (owner,p) = owners[[f,c]], owner == instance { picked.append((c,p)) } }
                    else { let p = recordSigmoid(row[c+1]); if p >= threshold { picked.append((c,p)) } }
                }
                if !picked.isEmpty { rec.fields[f] = picked }
            }
        }
        if !rec.fields.isEmpty { records.append(rec) }
    }
    if mode == .natural {
        guard let spans = spans else { return records }
        return records.sorted { let a = spans[$0.anchor!], b = spans[$1.anchor!]; return a.start == b.start ? a.end < b.end : a.start < b.start }
    }
    return dedupByFields(records)
}
public func dedupByFields(_ records: [DecodedRecord]) -> [DecodedRecord] {
    func key(_ r: DecodedRecord) -> [[Int]] { r.fields.keys.sorted().map { [$0] + r.fields[$0]!.map(\.0).sorted() } }
    var result: [DecodedRecord] = [], keys: [[[Int]]] = []
    for record in records { let k = key(record); if let i = keys.firstIndex(of: k) { if result[i].score < record.score { result[i] = record } } else { keys.append(k); result.append(record) } }
    return result
}

public func findChoiceMentions(text: String, choice: String) -> [(Int, Int)] {
    guard !choice.isEmpty, let regex = try? NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: choice), options: [.caseInsensitive]) else { return [] }
    let ns = text as NSString, length = ns.length
    var start = 0, output: [(Int,Int)] = []
    func word(_ scalar: Unicode.Scalar) -> Bool { scalar == "_" || CharacterSet.alphanumerics.contains(scalar) }
    while start <= length, let m = regex.firstMatch(in: text, range: NSRange(location: start, length: length - start)), let range = Range(m.range, in: text) {
        let left = text[..<range.lowerBound].unicodeScalars.last, right = text[range.upperBound...].unicodeScalars.first
        let valid = (left == nil || !word(left!)) && (right == nil || !word(right!)) && !range.isEmpty
        if valid {
            output.append((text[..<range.lowerBound].unicodeScalars.count, text[..<range.upperBound].unicodeScalars.count)); start = m.range.location + m.range.length
        } else {
            guard range.lowerBound < text.endIndex else { break }
            let next = text.unicodeScalars.index(after: range.lowerBound)
            start = NSRange(text.startIndex..<next, in: text).length
        }
    }
    return output
}
public func recordLocalChoiceMentions(text: String, choices: [String], anchors: [(Int,Int)?]) -> (Bool, [Int: [(String,Int,Int)]]) {
    var mentions: [(String,Int,Int)] = []
    for choice in choices { mentions += findChoiceMentions(text: text, choice: choice).map { (choice,$0.0,$0.1) } }
    if mentions.isEmpty { return (false, [:]) }
    let valid = anchors.enumerated().compactMap { i, a -> (Int, (Int,Int))? in a.map { (i,$0) } }.sorted { $0.1.0 == $1.1.0 ? $0.0 < $1.0 : $0.1.0 < $1.1.0 }
    if valid.isEmpty { return (true, [:]) }
    mentions = mentions.enumerated().sorted { $0.element.1 == $1.element.1 ? $0.offset < $1.offset : $0.element.1 < $1.element.1 }.map(\.element)
    var result: [Int: [(String,Int,Int)]] = [:]
    for m in mentions {
        let owner = valid.last(where: { $0.1.0 <= m.1 })?.0 ?? valid[0].0
        if !(result[owner] ?? []).contains(where: { $0.0 == m.0 }) { result[owner, default: []].append(m) }
    }
    return (true, result)
}
