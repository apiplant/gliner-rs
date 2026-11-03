import Foundation
import MLX

/// DeBERTa-v2 eval encoder with shared-key disentangled relative attention.
public final class DebertaV2 {
    private struct Layer {
        let query: TensorLinear, key: TensorLinear, value: TensorLinear
        let attentionDense: TensorLinear, attentionNorm: TensorLayerNorm
        let intermediate: TensorLinear, output: TensorLinear, outputNorm: TensorLayerNorm
        let positionQuery: MLXArray, positionKey: MLXArray
    }
    private let embedding: MLXArray
    private let embeddingNorm: TensorLayerNorm
    private let layers: [Layer]
    private let config: DebertaConfig
    private let dtype: DType

    public init(weights: WeightStore, config: DebertaConfig, dtype: DType = .float32) throws {
        try config.validate()
        guard [.float32, .float16].contains(dtype) else {
            throw GLiNERError.unsupported("DeBERTa encoder dtype must be float32 or float16")
        }
        guard config.hiddenSize > 0, config.numAttentionHeads > 0,
              config.hiddenSize % config.numAttentionHeads == 0, config.posEbdSize() > 0,
              config.numHiddenLayers >= 0, config.intermediateSize > 0 else {
            throw GLiNERError.invalidConfiguration("Invalid DeBERTa dimensions")
        }
        self.config = config; self.dtype = dtype
        let w = weights.prefixed("encoder", dtype: dtype)
        let h = config.hiddenSize, heads = config.numAttentionHeads, hd = h / heads
        embedding = try w.get("embeddings.word_embeddings.weight", shape: [config.vocabSize, h])
        embeddingNorm = try TensorLayerNorm(w, "embeddings.LayerNorm", h, eps: Float(config.layerNormEps))
        let rel = try w.get("encoder.rel_embeddings.weight", shape: [2 * config.posEbdSize(), h])
        let relNorm = try TensorLayerNorm(w, "encoder.LayerNorm", h, eps: Float(config.layerNormEps))
        let normalizedRel = relNorm(rel).expandedDimensions(axis: 0)
        var loaded: [Layer] = []
        for i in 0..<config.numHiddenLayers {
            let p = "encoder.layer.\(i)"
            let q = try TensorLinear(w, p + ".attention.self.query_proj", h, h)
            let k = try TensorLinear(w, p + ".attention.self.key_proj", h, h)
            let pq = q(normalizedRel).reshaped(1, -1, heads, hd).swappedAxes(1, 2)
            let pk = k(normalizedRel).reshaped(1, -1, heads, hd).swappedAxes(1, 2)
            eval(pq, pk)
            loaded.append(try Layer(query: q, key: k,
                value: TensorLinear(w, p + ".attention.self.value_proj", h, h),
                attentionDense: TensorLinear(w, p + ".attention.output.dense", h, h),
                attentionNorm: TensorLayerNorm(w, p + ".attention.output.LayerNorm", h, eps: Float(config.layerNormEps)),
                intermediate: TensorLinear(w, p + ".intermediate.dense", h, config.intermediateSize),
                output: TensorLinear(w, p + ".output.dense", config.intermediateSize, h),
                outputNorm: TensorLayerNorm(w, p + ".output.LayerNorm", h, eps: Float(config.layerNormEps)),
                positionQuery: pq, positionKey: pk))
        }
        layers = loaded
    }

    /// Float32 bucketing matches the reference, including its unbounded far buckets.
    static func logBucketPosition(_ relative: Int, bucketSize: Int, maxPosition: Int) -> Int {
        let mid = bucketSize / 2
        guard mid > 0, maxPosition > mid + 1 else { return relative }
        let distance = relative < mid && relative > -mid ? mid - 1 : abs(relative)
        if distance <= mid { return relative }
        let bucket = ceil(log(Float(distance) / Float(mid)) / log(Float(maxPosition - 1) / Float(mid)) * Float(mid - 1)) + Float(mid)
        return Int(bucket * (relative < 0 ? -1 : 1))
    }

    public func forward(inputIds: MLXArray, attentionMask: MLXArray) throws -> MLXArray {
        guard inputIds.ndim == 2, inputIds.shape == attentionMask.shape,
              inputIds.dim(0) > 0, inputIds.dim(1) > 0 else {
            throw GLiNERError.invalidInput("DeBERTa input IDs and mask must have the same nonempty [B,T] shape")
        }
        let b = inputIds.dim(0), t = inputIds.dim(1), heads = config.numAttentionHeads
        let hd = config.hiddenSize / heads, span = config.posEbdSize()
        let mask = attentionMask.asType(dtype)
        var hidden = embeddingNorm(embedding.take(inputIds.asType(.int32), axis: 0)) * mask.expandedDimensions(axis: 2)
        var c2p = [Int32](), p2c = [Int32]()
        c2p.reserveCapacity(t * t); p2c.reserveCapacity(t * t)
        for i in 0..<t { for j in 0..<t {
            var r = i - j
            if config.positionBuckets > 0 && config.maxRelativePositionsResolved() > 0 {
                r = Self.logBucketPosition(r, bucketSize: config.positionBuckets, maxPosition: config.maxRelativePositionsResolved())
            }
            c2p.append(Int32(min(max(r + span, 0), 2 * span - 1)))
            p2c.append(Int32(min(max(-r + span, 0), 2 * span - 1)))
        }}
        let ci = broadcast(MLXArray(c2p, [1, 1, t, t]), to: [b, heads, t, t])
        let pi = broadcast(MLXArray(p2c, [1, 1, t, t]), to: [b, heads, t, t])
        let pairMask = (mask.expandedDimensions(axis: 1) * mask.expandedDimensions(axis: 2)).expandedDimensions(axis: 1).asType(.bool)
        let floor = MLXArray(dtype == .float16 ? Float(-65504) : -Float.greatestFiniteMagnitude).asType(dtype)
        let inverseScale = Float(1 / sqrt(Double(hd * 3)))
        func split(_ x: MLXArray) -> MLXArray { x.reshaped(b, t, heads, hd).swappedAxes(1, 2) }
        for layer in layers {
            let q = split(layer.query(hidden)), k = split(layer.key(hidden)), v = split(layer.value(hidden))
            let content = matmul(q, k.swappedAxes(-1, -2))
            let c = takeAlong(matmul(q, layer.positionKey.swappedAxes(-1, -2)), ci, axis: 3)
            let p = takeAlong(matmul(k, layer.positionQuery.swappedAxes(-1, -2)), pi, axis: 3).swappedAxes(2, 3)
            let scores = which(pairMask, (content + c + p) * inverseScale, floor)
            let context = matmul(softmax(scores, axis: -1), v).swappedAxes(1, 2).reshaped(b, t, config.hiddenSize)
            let attended = layer.attentionNorm(layer.attentionDense(context) + hidden)
            hidden = layer.outputNorm(layer.output(geluErf(layer.intermediate(attended))) + attended)
        }
        return hidden
    }
}
