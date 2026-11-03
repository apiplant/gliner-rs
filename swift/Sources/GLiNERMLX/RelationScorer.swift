import Foundation
import MLX

/// Endpoint, positional, and optional gated biaffine relation scorer.
public struct RelationScorer {
    private let mlpHidden: TensorLinear, mlpOutput: TensorLinear
    private let headProjection: TensorLinear?, tailProjection: TensorLinear?
    private let gateProjection: TensorLinear?, contentLinear: TensorLinear?
    private let hidden: Int, relationDim: Int
    public init(weights: WeightStore, hidden: Int, config: BoundaryHeadConfig, prefix: String = "relation_scorer") throws {
        self.hidden = hidden
        relationDim = config.directionalRelationStates ? 2 * hidden : hidden
        let w = weights.prefixed(prefix, dtype: .float32)
        mlpHidden = try TensorLinear(w, "mlp.0", 4 * hidden + relationDim + 2, hidden)
        mlpOutput = try TensorLinear(w, "mlp.3", hidden, 1)
        if config.relationBiaffineContent {
            headProjection = try TensorLinear(w, "head_content_projection", hidden, hidden)
            tailProjection = try TensorLinear(w, "tail_content_projection", hidden, hidden)
            gateProjection = try TensorLinear(w, "relation_content_gate", relationDim, hidden)
            contentLinear = try TensorLinear(w, "content_linear", 2 * hidden + relationDim, 1)
        } else { headProjection = nil; tailProjection = nil; gateProjection = nil; contentLinear = nil }
    }
    public func forward(text: MLXArray, relationStates: MLXArray, pairs: [RelationPair]) throws -> [Float] {
        if pairs.isEmpty { return [] }
        guard text.ndim == 2, text.dim(0) > 0, text.dim(1) == hidden,
              relationStates.ndim == 2, relationStates.dim(1) == relationDim,
              pairs.allSatisfy({ $0.relation >= 0 && $0.relation < relationStates.dim(0) && $0.head.0 >= 0 && $0.head.1 >= 0 && $0.tail.0 >= 0 && $0.tail.1 >= 0 }) else {
            throw GLiNERError.invalidInput("Invalid relation scorer states or pair indices")
        }
        let text = text.asType(.float32), relationStates = relationStates.asType(.float32)
        let n = text.dim(0), p = pairs.count, last = n - 1
        func gather(_ positions: [Int]) -> MLXArray { text.take(indexArray(positions.map { min($0, last) }), axis: 0) }
        let hs = gather(pairs.map { $0.head.0 }), he = gather(pairs.map { max(0, $0.head.1 - 1) })
        let ts = gather(pairs.map { $0.tail.0 }), te = gather(pairs.map { max(0, $0.tail.1 - 1) })
        let rel = relationStates.take(indexArray(pairs.map(\.relation)), axis: 0)
        let positional = pairs.flatMap { pair -> [Float] in
            let delta = Float(pair.tail.0) - Float(pair.head.0)
            return [delta == 0 ? 0 : (delta > 0 ? 1 : -1), abs(delta) / Float(max(1, n))]
        }
        let features = concatenated([hs, he, ts, te, rel, MLXArray(positional, [p, 2])], axis: 1)
        var score = mlpOutput(geluErf(mlpHidden(features))).reshaped(p)
        if let hp = headProjection, let tp = tailProjection, let gp = gateProjection, let cl = contentLinear {
            let prefix = concatenated([MLXArray.zeros([1, hidden]), text.cumsum(axis: 0)], axis: 0)
            func pool(_ spans: [(Int, Int)]) -> MLXArray {
                let starts = indexArray(spans.map { min($0.0, n) }), ends = indexArray(spans.map { min($0.1, n) })
                let widths = MLXArray(spans.map { Float(max(1, $0.1 - $0.0)) }, [spans.count, 1])
                return (prefix.take(ends, axis: 0) - prefix.take(starts, axis: 0)) / widths
            }
            let head = hp(pool(pairs.map(\.head))), tail = tp(pool(pairs.map(\.tail)))
            let gated = ((head * sigmoid(gp(rel))) * tail).sum(axis: 1) * Float(1 / sqrt(Double(hidden)))
            let linear = cl(concatenated([head, tail, rel], axis: 1)).reshaped(p)
            score = score + gated + linear
        }
        return score.asArray(Float.self)
    }
}
