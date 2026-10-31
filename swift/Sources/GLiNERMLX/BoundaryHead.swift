import Foundation
import MLX

public let boundaryMaskLogit: Float = -1e4
public struct CandidateSpan: Equatable, Hashable, Codable, Sendable {
    public var start: Int
    public var end: Int
    public init(start: Int, end: Int) { self.start = start; self.end = end }
}
struct BoundaryMarginals {
    let start: [[Float]], end: [[Float]], insidePrefix: [[Float]], insideMean: [Float]
}
public struct BoundaryOutput {
    public let logits: [[Float]]
    public let spans: [CandidateSpan]
    public let noMatchLogits: [Float]?
    let boundary: MLXArray
    let marginals: BoundaryMarginals
}
private struct PoolCandidate { let span: CandidateSpan; let compatibility: Float }

/// Descending score, stable ascending input index for equal scores.
func stableScoreOrder(_ scores: [Float]) -> [Int] {
    scores.indices.sorted {
        // Rust f32::total_cmp also orders signed zeros and NaN payloads.
        let a = Int32(bitPattern: scores[$0].bitPattern), b = Int32(bitPattern: scores[$1].bitPattern)
        let ka = a ^ Int32(UInt32(bitPattern: a >> 31) >> 1)
        let kb = b ^ Int32(UInt32(bitPattern: b >> 31) >> 1)
        return ka == kb ? $0 < $1 : ka > kb
    }
}
func deduplicateBoundaryPool(keys: [Int], scores: [Float], valid: [Bool], capacity: Int, n: Int) -> [Int] {
    let maskedKeys = keys.indices.map { valid[$0] ? keys[$0] : n * n }
    let maskedScores = scores.indices.map { valid[$0] ? scores[$0] : boundaryMaskLogit }
    let sorted = stableScoreOrder(maskedScores)
    let order = sorted.enumerated().sorted {
        maskedKeys[$0.element] == maskedKeys[$1.element] ? $0.offset < $1.offset : maskedKeys[$0.element] < maskedKeys[$1.element]
    }.map(\.element)
    let keep = order.indices.map { valid[order[$0]] && ($0 == 0 || maskedKeys[order[$0]] != maskedKeys[order[$0 - 1]]) }
    let keptScores = order.indices.map { keep[$0] ? maskedScores[order[$0]] : boundaryMaskLogit }
    return stableScoreOrder(keptScores).prefix(max(0, capacity)).filter { keep[$0] }.map { maskedKeys[order[$0]] }
}
private func dotF32(_ a: [Float], _ b: [Float]) -> Float {
    zip(a, b).reduce(Float(0)) { $0 + $1.0 * $1.1 }
}

/// Shared document candidate pool, boundary encoder and explicit pair reranker.
public final class BoundaryHead {
    private let config: BoundaryHeadConfig
    private let hidden: Int
    private var linears: [String: TensorLinear] = [:]
    private var norms: [String: TensorLayerNorm] = [:]
    private let bos: MLXArray, eos: MLXArray

    public init(weights: WeightStore, hidden: Int, config: BoundaryHeadConfig, prefix: String = "boundary_head") throws {
        try config.validate()
        let d = config.boundaryDim, p = config.pairDim, c = config.contentDim
        guard hidden > 0, d > 0, p > 0, c > 0, config.boundaryAttentionHeads > 0,
              d % config.boundaryAttentionHeads == 0, config.boundaryFfnMultiplier.isFinite,
              config.boundaryFfnMultiplier > 0, config.boundaryAttentionLayers >= 0,
              config.boundaryRefinementLayers >= 0, config.poolBoundaryTopK >= 0,
              config.poolSize >= 0, config.minPoolPerQuery >= 0 else {
            throw GLiNERError.invalidConfiguration("Invalid boundary head dimensions")
        }
        self.config = config; self.hidden = hidden
        let w = weights.prefixed(prefix, dtype: .float32)
        bos = try w.get("boundary_encoder.bos_state", shape: [hidden])
        eos = try w.get("boundary_encoder.eos_state", shape: [hidden])
        func add(_ name: String, _ input: Int, _ output: Int) throws { linears[name] = try TensorLinear(w, name, input, output) }
        func norm(_ name: String, _ size: Int) throws { norms[name] = try TensorLayerNorm(w, name, size) }
        try add("boundary_encoder.left_projection", hidden, d)
        try add("boundary_encoder.right_projection", hidden, d)
        try add("boundary_encoder.output_projection", 2 * d, d)
        try norm("boundary_encoder.layer_norm", d)
        for i in 0..<config.boundaryAttentionLayers {
            let b = "boundary_encoder.attention_blocks.\(i)"
            try norm(b + ".norm", d); try add(b + ".qkv_projection", d, 3 * d); try add(b + ".output_projection", d, d)
        }
        let f = max(1, Int((Double(d) * config.boundaryFfnMultiplier).rounded(.down)))
        for i in 0..<config.boundaryRefinementLayers {
            let b = "boundary_encoder.refinement_blocks.\(i)"
            try norm(b + ".norm", d); try add(b + ".input_projection", d, 2 * f); try add(b + ".output_projection", f, d)
        }
        for endpoint in ["start", "end"] {
            try add("boundary_query_head.\(endpoint)_boundary_projection", d, d)
            try add("boundary_query_head.\(endpoint)_query_projection", hidden, d)
            try add("shared_pool_builder.\(endpoint)_projection", d, d)
            try add("shared_pool_scorer.\(endpoint)_projection", d, p)
        }
        try add("boundary_query_head.inside_text_projection", hidden, d)
        try add("boundary_query_head.inside_query_projection", hidden, d)
        let s = "shared_pool_scorer."
        try add(s + "length_projection", 3, p); try add(s + "prior_projection", 1, p)
        if config.enableSpanContent {
            try add(s + "content_pooler.value_projection", hidden, c); try norm(s + "content_pooler.layer_norm", c)
            try add(s + "content_projection", c, p)
        }
        try norm(s + "candidate_norm", p); try add(s + "query_projection", hidden, p)
        try add(s + "film", p, 2 * p); try add(s + "film_output.0", p, 64); try add(s + "film_output.3", 64, 1)
        if config.enableAbstention { try add("null_projection", hidden, 1) }
        if config.enableRecords { try add("candidate_encoder", 2 * d, hidden) }
        try add("boundary_proposer.start_pair_projection", d, d)
        try add("boundary_proposer.end_key_projection", d, d)
        try add("boundary_proposer.start_query_projection", hidden, d / 2)
        let r = "pair_scorer."
        try add(r + "start_endpoint_projection", d, p); try add(r + "end_endpoint_projection", d, p)
        try add(r + "query_gate", hidden, p / 2); try add(r + "length_query_projection", hidden, 3)
        try add(r + "inside_weight", hidden, 1); try add(r + "endpoint_difference_projection", 2 * p, 1)
        try add(r + "content_pooler.value_projection", hidden, c); try norm(r + "content_pooler.layer_norm", c)
        try add(r + "content_query_projection", hidden, c); try add(r + "content_bias", c, 1)
        try add(r + "compat_mix", config.multiheadPairCompatHeads, 1)
    }
    private func linear(_ name: String, _ x: MLXArray) -> MLXArray { linears[name]!(x) }
    private func norm(_ name: String, _ x: MLXArray) -> MLXArray { norms[name]!(x) }
    private func validate(text: MLXArray, query: MLXArray) throws {
        guard text.ndim == 2, query.ndim == 2, text.dim(1) == hidden, query.dim(1) == hidden,
              text.dim(0) > 0, query.dim(0) > 0 else {
            throw GLiNERError.invalidInput("Boundary text/query must be nonempty [L,H]/[Q,H] matrices")
        }
    }
    private func encode(_ text: MLXArray) -> MLXArray {
        let left = concatenated([bos.expandedDimensions(axis: 0), text], axis: 0)
        let right = concatenated([text, eos.expandedDimensions(axis: 0)], axis: 0)
        let joined = concatenated([linear("boundary_encoder.left_projection", left), linear("boundary_encoder.right_projection", right)], axis: 1)
        var states = norm("boundary_encoder.layer_norm", linear("boundary_encoder.output_projection", joined))
        let n = states.dim(0), heads = config.boundaryAttentionHeads, hd = config.boundaryDim / heads
        if config.boundaryAttentionLayers > 0 {
            let biasValues: [Float] = (0..<n * n).map { config.boundaryAttentionWindow > 0 && abs($0 / n - $0 % n) > config.boundaryAttentionWindow ? -.infinity : 0 }
            let bias = MLXArray(biasValues, [n, n])
            for i in 0..<config.boundaryAttentionLayers {
                let b = "boundary_encoder.attention_blocks.\(i)"
                let qkv = linear(b + ".qkv_projection", norm(b + ".norm", states)).reshaped(n, 3, heads, hd)
                func part(_ index: Int) -> MLXArray { tensorSlice(qkv, axis: 1, start: index, count: 1).reshaped(n, heads, hd).swappedAxes(0, 1) }
                let q = part(0), k = part(1), v = part(2)
                let scores = matmul(q, k.swappedAxes(-1, -2)) * Float(1 / sqrt(Double(hd))) + bias
                let attended = matmul(softmax(scores, axis: -1), v).swappedAxes(0, 1).reshaped(n, config.boundaryDim)
                states = states + linear(b + ".output_projection", attended)
            }
        }
        let f = max(1, Int((Double(config.boundaryDim) * config.boundaryFfnMultiplier).rounded(.down)))
        for i in 0..<config.boundaryRefinementLayers {
            let b = "boundary_encoder.refinement_blocks.\(i)"
            let projected = linear(b + ".input_projection", norm(b + ".norm", states))
            let value = tensorSlice(projected, axis: -1, start: 0, count: f)
            let gate = tensorSlice(projected, axis: -1, start: f, count: f)
            states = states + linear(b + ".output_projection", value * silu(gate))
        }
        return states
    }
    private func marginals(boundary: MLXArray, text: MLXArray, query: MLXArray) -> BoundaryMarginals {
        let scale = Float(1 / sqrt(Double(config.boundaryDim)))
        func scores(_ q: String, _ k: String, _ keys: MLXArray) -> [[Float]] {
            tensorRows(matmul(linear("boundary_query_head." + q, query), linear("boundary_query_head." + k, keys).transposed()) * scale)
        }
        let start = scores("start_query_projection", "start_boundary_projection", boundary)
        let end = scores("end_query_projection", "end_boundary_projection", boundary)
        let inside = scores("inside_query_projection", "inside_text_projection", text)
        var prefixes = [[Float]](), means = [Float]()
        for row in inside {
            let mean = row.reduce(Float(0), +) / Float(max(1, row.count))
            var prefix: [Float] = [0], acc: Float = 0
            for x in row { acc += x - mean; prefix.append(acc) }
            prefixes.append(prefix); means.append(mean)
        }
        return BoundaryMarginals(start: start, end: end, insidePrefix: prefixes, insideMean: means)
    }
    private func buildPool(boundary: MLXArray, marginals m: BoundaryMarginals) -> [PoolCandidate] {
        let n = boundary.dim(0), q = m.start.count
        let startsAll = tensorRows(linear("shared_pool_builder.start_projection", boundary))
        let endsAll = tensorRows(linear("shared_pool_builder.end_projection", boundary))
        let scale = Float(1) / Float(config.boundaryDim).squareRoot()
        let unionStart = (0..<n).map { i in m.start.reduce(boundaryMaskLogit) { max($0, $1[i]) } }
        let unionEnd = (0..<n).map { i in m.end.reduce(boundaryMaskLogit) { max($0, $1[i]) } }
        let k = min(config.poolBoundaryTopK, n)
        let starts = Array(stableScoreOrder(unionStart).prefix(k)), ends = Array(stableScoreOrder(unionEnd).prefix(k))
        var ps = [Int](), pe = [Int](), pv = [Bool](), compat = [Float]()
        for s in starts { for e in ends { ps.append(s); pe.append(e); pv.append(e > s); compat.append(dotF32(startsAll[s], endsAll[e]) * scale) } }
        var keys = [Int](), scores = [Float](), valid = [Bool]()
        let quota = min(config.minPoolPerQuery, ps.count)
        if quota > 0 { for qi in 0..<q {
            let perQuery = ps.indices.map { pv[$0] ? m.start[qi][ps[$0]] + m.end[qi][pe[$0]] + compat[$0] : boundaryMaskLogit }
            for (rank, i) in stableScoreOrder(perQuery).prefix(quota).enumerated() {
                keys.append(ps[i] * n + pe[i]); valid.append(pv[i]); scores.append(-boundaryMaskLogit * 0.5 + Float(quota - rank))
            }
        }}
        for i in ps.indices {
            keys.append(ps[i] * n + pe[i]); valid.append(pv[i]); scores.append(compat[i] + unionStart[ps[i]] + unionEnd[pe[i]])
        }
        return deduplicateBoundaryPool(keys: keys, scores: scores, valid: valid, capacity: config.poolSize, n: n).map {
            let s = $0 / n, e = $0 % n
            return PoolCandidate(span: CandidateSpan(start: s, end: e), compatibility: dotF32(startsAll[s], endsAll[e]) * scale)
        }
    }
    private func learnedScores(boundary: MLXArray, text: MLXArray, query: MLXArray, candidates: [PoolCandidate]) -> [[Float]] {
        let c = candidates.count, p = config.pairDim
        let starts = indexArray(candidates.map { $0.span.start }), ends = indexArray(candidates.map { $0.span.end })
        let lengths = candidates.map { Float(max(1, $0.span.end - $0.span.start)) }
        let tl = Float(max(1, text.dim(0)))
        let features = lengths.flatMap { [log1p($0), $0 / tl, 1 / $0.squareRoot()] }
        let s = "shared_pool_scorer."
        let sr = linear(s + "start_projection", boundary).take(starts, axis: 0)
        let er = linear(s + "end_projection", boundary).take(ends, axis: 0)
        var candidate = sr + er + linear(s + "length_projection", MLXArray(features, [c, 3]))
        candidate = candidate + linear(s + "prior_projection", MLXArray(candidates.map(\.compatibility), [c, 1]))
        if config.enableSpanContent {
            let values = linear(s + "content_pooler.value_projection", text)
            let prefix = concatenated([MLXArray.zeros([1, config.contentDim]), values.cumsum(axis: 0)], axis: 0)
            let pooled = (prefix.take(ends, axis: 0) - prefix.take(starts, axis: 0)) / MLXArray(lengths, [c, 1])
            candidate = candidate + linear(s + "content_projection", norm(s + "content_pooler.layer_norm", pooled))
        }
        candidate = norm(s + "candidate_norm", candidate)
        let query = linear(s + "query_projection", query)
        let base = matmul(candidate, query.transposed()) * Float(1 / sqrt(Double(p)))
        let film = linear(s + "film", query)
        let gamma = tensorSlice(film, axis: 1, start: 0, count: p), beta = tensorSlice(film, axis: 1, start: p, count: p)
        let conditioned = candidate.expandedDimensions(axis: 1) * (gamma + 1).expandedDimensions(axis: 0) + beta.expandedDimensions(axis: 0)
        let filmScore = linear(s + "film_output.3", geluErf(linear(s + "film_output.0", conditioned))).reshaped(c, query.dim(0))
        return tensorRows(base + filmScore)
    }
    private func inside(_ m: BoundaryMarginals, query: Int, start: Int, end: Int) -> Float {
        guard config.useInsideEvidence else { return 0 }
        let interval = m.insidePrefix[query][end] - m.insidePrefix[query][start] + m.insideMean[query] * Float(end - start)
        return interval / Float(max(1, end - start)).squareRoot()
    }
    public func forward(text: MLXArray, query: MLXArray) throws -> BoundaryOutput {
        try validate(text: text, query: query)
        let text = text.asType(.float32), query = query.asType(.float32)
        let boundary = encode(text), m = marginals(boundary: boundary, text: text, query: query)
        let pool = buildPool(boundary: boundary, marginals: m)
        let learned = pool.isEmpty ? [] : learnedScores(boundary: boundary, text: text, query: query, candidates: pool)
        let logits = (0..<query.dim(0)).map { qi in pool.indices.map { ci in
            let span = pool[ci].span
            return learned[ci][qi] + m.start[qi][span.start] + m.end[qi][span.end] + inside(m, query: qi, start: span.start, end: span.end)
        }}
        let null = config.enableAbstention ? linear("null_projection", query).asArray(Float.self) : nil
        return BoundaryOutput(logits: logits, spans: pool.map(\.span), noMatchLogits: null, boundary: boundary, marginals: m)
    }
    public func candidateStates(_ output: BoundaryOutput) throws -> MLXArray? {
        guard config.enableRecords, !output.spans.isEmpty else { return nil }
        let starts = output.boundary.take(indexArray(output.spans.map(\.start)), axis: 0)
        let ends = output.boundary.take(indexArray(output.spans.map(\.end)), axis: 0)
        return linear("candidate_encoder", concatenated([starts, ends], axis: 1))
    }
    /// Explicit half-open spans for one query, matching the rotary pair reranker.
    public func scoreExplicitSpans(output: BoundaryOutput, text: MLXArray, query: MLXArray,
                                   queryIndex: Int, spans: [CandidateSpan]) throws -> [Float] {
        if spans.isEmpty { return [] }
        try validate(text: text, query: query)
        guard queryIndex >= 0, queryIndex < query.dim(0), queryIndex < output.marginals.start.count,
              spans.allSatisfy({ $0.start >= 0 && $0.end >= 0 }) else {
            throw GLiNERError.invalidInput("Invalid explicit span or query index")
        }
        let text = text.asType(.float32), query = query.asType(.float32)
        let n = output.boundary.dim(0), c = spans.count, p = config.pairDim
        let starts = spans.map { min($0.start, n - 1) }, ends = spans.map { min($0.end, n - 1) }
        let si = indexArray(starts), ei = indexArray(ends)
        let bs = output.boundary.take(si, axis: 0), be = output.boundary.take(ei, axis: 0)
        let q = query.take(indexArray([queryIndex]), axis: 0)
        func rotated(_ name: String, _ states: MLXArray, _ positions: [Int]) -> MLXArray {
            let states = linear(name, states), dim = states.dim(-1)
            var cosine = [Float](), sine = [Float]()
            for position in positions { for i in 0..<dim / 2 {
                let inverse = 1 / pow(config.rotaryBase, Float(2 * i) / Float(dim))
                let angle = Float(position) * inverse
                cosine.append(cos(angle)); sine.append(sin(angle))
            }}
            let pairs = states.reshaped(c, dim / 2, 2)
            let even = tensorSlice(pairs, axis: 2, start: 0, count: 1).reshaped(c, dim / 2)
            let odd = tensorSlice(pairs, axis: 2, start: 1, count: 1).reshaped(c, dim / 2)
            let co = MLXArray(cosine, [c, dim / 2]), sn = MLXArray(sine, [c, dim / 2])
            let re = (even * co - odd * sn).expandedDimensions(axis: 2)
            let ro = (even * sn + odd * co).expandedDimensions(axis: 2)
            return concatenated([re, ro], axis: 2).reshaped(c, dim)
        }
        func gate(_ name: String, dimension: Int) -> MLXArray {
            let half = sigmoid(linear(name, q)).reshaped(1, dimension / 2, 1)
            return broadcast(half, to: [1, dimension / 2, 2]).reshaped(1, dimension)
        }
        let propStart = rotated("boundary_proposer.start_pair_projection", bs, starts)
        let propEnd = rotated("boundary_proposer.end_key_projection", be, ends)
        let propGate = gate("boundary_proposer.start_query_projection", dimension: config.boundaryDim)
        let prior = ((propStart * propGate) * propEnd).sum(axis: 1) * (Float(1) / Float(config.boundaryDim).squareRoot())
        let ss = rotated("pair_scorer.start_endpoint_projection", bs, starts)
        let se = rotated("pair_scorer.end_endpoint_projection", be, ends)
        let g = gate("pair_scorer.query_gate", dimension: p)
        let heads = config.multiheadPairCompatHeads
        let perHead = ((ss * g) * se).reshaped(c, heads, p / heads).sum(axis: 2)
        let compat = linear("pair_scorer.compat_mix", perHead).reshaped(c) * (Float(1) / Float(p).squareRoot())
        let difference = ss - se
        let differenceScore = linear("pair_scorer.endpoint_difference_projection", concatenated([difference, abs(difference)], axis: 1)).reshaped(c)
        let values = linear("pair_scorer.content_pooler.value_projection", text)
        let prefix = concatenated([MLXArray.zeros([1, config.contentDim]), values.cumsum(axis: 0)], axis: 0)
        let lengths = spans.map { Float(max(1, $0.end - $0.start)) }
        let content = norm("pair_scorer.content_pooler.layer_norm", (prefix.take(ei, axis: 0) - prefix.take(si, axis: 0)) / MLXArray(lengths, [c, 1]))
        let contentBias = linear("pair_scorer.content_bias", content).reshaped(c)
        let coefficient = linear("pair_scorer.content_query_projection", q)
        let contentScore = (content * coefficient).sum(axis: 1) * (Float(1) / Float(config.contentDim).squareRoot())
        // Preserve Rust's float32 accumulation order for the final score.
        let compatibility = compat.asArray(Float.self), differences = differenceScore.asArray(Float.self)
        let priors = prior.asArray(Float.self), contents = contentScore.asArray(Float.self)
        let biases = contentBias.asArray(Float.self)
        let insideWeight = linear("pair_scorer.inside_weight", q).asArray(Float.self)[0]
        let lengthCoefficient = linear("pair_scorer.length_query_projection", q).asArray(Float.self)
        let m = output.marginals, tl = Float(max(1, text.dim(0)))
        return (0..<c).map { i in
            let s = starts[i], e = ends[i], len = lengths[i]
            var score = compatibility[i] + differences[i]
            score += m.start[queryIndex][s] + m.end[queryIndex][e] + priors[i]
            score += contents[i]; score += biases[i]
            if config.useInsideEvidence {
                let interval = m.insidePrefix[queryIndex][e] - m.insidePrefix[queryIndex][s] + m.insideMean[queryIndex] * Float(e - s)
                score += insideWeight * (interval / len.squareRoot())
            }
            let features = [log1p(len), len / tl, 1 / len.squareRoot()]
            score += dotF32(features, lengthCoefficient)
            return score
        }
    }
    /// Convenience explicit scoring for a single query state [1,H].
    public func scorePairs(text: MLXArray, query: MLXArray, starts: [Int], ends: [Int]) throws -> [Float] {
        guard starts.count == ends.count else { throw GLiNERError.invalidInput("Explicit starts/ends must have equal lengths") }
        let q = query.ndim == 1 ? query.expandedDimensions(axis: 0) : query
        guard q.ndim == 2, q.dim(0) == 1 else { throw GLiNERError.invalidInput("scorePairs expects exactly one query") }
        if starts.isEmpty { return [] }
        try validate(text: text, query: q)
        let t = text.asType(.float32), qf = q.asType(.float32)
        let b = encode(t), m = marginals(boundary: b, text: t, query: qf)
        let output = BoundaryOutput(logits: [[]], spans: [], noMatchLogits: nil, boundary: b, marginals: m)
        return try scoreExplicitSpans(output: output, text: t, query: qf, queryIndex: 0,
                                      spans: zip(starts, ends).map { CandidateSpan(start: $0, end: $1) })
    }
}
