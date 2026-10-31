import XCTest
import MLX
@testable import GLiNERMLX

final class DebertaTests: XCTestCase {
    func testRelativePositionBuckets() {
        for r in -128...128 {
            XCTAssertEqual(DebertaV2.logBucketPosition(r, bucketSize: 256, maxPosition: 512), r)
        }
        XCTAssertEqual(DebertaV2.logBucketPosition(511, bucketSize: 256, maxPosition: 512), 255)
        XCTAssertEqual(DebertaV2.logBucketPosition(-511, bucketSize: 256, maxPosition: 512), -255)
        XCTAssertGreaterThan(DebertaV2.logBucketPosition(2000, bucketSize: 256, maxPosition: 512), 255)
    }

    func testEmbeddingNormalizationAndPaddingInBatchedEncoder() throws {
        let config = DebertaConfig(hiddenSize: 4, numAttentionHeads: 2, numHiddenLayers: 0,
            intermediateSize: 8, layerNormEps: 1e-5, maxPositionEmbeddings: 8,
            maxRelativePositions: 8, positionBuckets: 4, vocabSize: 3,
            relativeAttention: true, shareAttKey: true, posAttType: ["c2p", "p2c"],
            positionBiasedInput: false, normRelEbd: "layer_norm", hiddenAct: "gelu")
        let weights: [String: MLXArray] = [
            "encoder.embeddings.word_embeddings.weight": MLXArray([Float(1), 2, 3, 4, 4, 1, 2, 3, 0, 0, 0, 0], [3, 4]),
            "encoder.embeddings.LayerNorm.weight": MLXArray.ones([4]),
            "encoder.embeddings.LayerNorm.bias": MLXArray.zeros([4]),
            "encoder.encoder.rel_embeddings.weight": MLXArray.zeros([8, 4]),
            "encoder.encoder.LayerNorm.weight": MLXArray.ones([4]),
            "encoder.encoder.LayerNorm.bias": MLXArray.zeros([4]),
        ]
        let encoder = try DebertaV2(weights: WeightStore(weights), config: config)
        let output = try encoder.forward(inputIds: MLXArray([Int32(0), 1, 1, 0], [2, 2]),
                                         attentionMask: MLXArray([Int32(1), 0, 1, 1], [2, 2]))
        XCTAssertEqual(output.shape, [2, 2, 4])
        let actual = output.asArray(Float.self)
        let denominator = Float(1.25 + 1e-5).squareRoot()
        let a: [Float] = [-1.5, -0.5, 0.5, 1.5].map { $0 / denominator }
        let b: [Float] = [1.5, -1.5, -0.5, 0.5].map { $0 / denominator }
        let expected = a + [0, 0, 0, 0] + b + a
        for (actual, expected) in zip(actual, expected) { XCTAssertEqual(actual, expected, accuracy: 1e-5) }
    }

    func testWeightShapeErrorsAreReportedBeforeExecution() throws {
        let store = WeightStore(["layer.weight": MLXArray.zeros([2, 3]), "layer.bias": MLXArray.zeros([2])])
        XCTAssertThrowsError(try store.validateLinear("layer", input: 4, output: 2))
        XCTAssertThrowsError(try store.get("absent"))
    }

    func testClassificationUsesReluAndCheckpointOutputIndex() throws {
        let store = WeightStore([
            "classifier.0.weight": MLXArray([Float(1), 0, 0, 1, -1, 0, 0, -1], [4, 2]),
            "classifier.0.bias": MLXArray.zeros([4]),
            "classifier.3.weight": MLXArray([Float(1), 2, 3, 4], [1, 4]),
            "classifier.3.bias": MLXArray([Float(0.5)]),
        ])
        let classifier = try ClassificationHead(weights: store, hidden: 2)
        let logits = try classifier.forward(MLXArray([Float(2), -3, -2, 3], [2, 2]))
        XCTAssertEqual(logits, [14.5, 12.5])
    }
}

extension DebertaTests {
    /// An independent scalar oracle exercises both relative-attention terms,
    /// nonzero projection biases, masking, residuals, GELU and layer norms.
    func testOneLayerMatchesScalarDisentangledAttention() throws {
        let h = 4, heads = 2, hd = 2, intermediate = 6, span = 4, t = 3
        let config = DebertaConfig(hiddenSize: h, numAttentionHeads: heads, numHiddenLayers: 1,
            intermediateSize: intermediate, layerNormEps: 1e-5, maxPositionEmbeddings: 8,
            maxRelativePositions: 8, positionBuckets: span, vocabSize: 3,
            relativeAttention: true, shareAttKey: true, posAttType: ["c2p", "p2c"],
            positionBiasedInput: false, normRelEbd: "layer_norm", hiddenAct: "gelu")
        var arrays: [String: MLXArray] = [:], raw: [String: [Float]] = [:]
        func tensor(_ name: String, shape: [Int], seed: Int) {
            let count = shape.reduce(1, *)
            let values = (0..<count).map { sin(Float($0 * 7 + seed)) * 0.3 }
            raw[name] = values; arrays["encoder." + name] = MLXArray(values, shape)
        }
        func linear(_ name: String, input: Int, output: Int, seed: Int) {
            tensor(name + ".weight", shape: [output, input], seed: seed)
            tensor(name + ".bias", shape: [output], seed: seed + 1)
        }
        func ln(_ name: String) {
            raw[name + ".weight"] = [1.1, 0.9, 1.2, 0.8]
            raw[name + ".bias"] = [0.01, -0.02, 0.03, -0.04]
            arrays["encoder." + name + ".weight"] = MLXArray(raw[name + ".weight"]!)
            arrays["encoder." + name + ".bias"] = MLXArray(raw[name + ".bias"]!)
        }
        tensor("embeddings.word_embeddings.weight", shape: [3, h], seed: 1)
        tensor("encoder.rel_embeddings.weight", shape: [2 * span, h], seed: 2)
        ln("embeddings.LayerNorm"); ln("encoder.LayerNorm")
        let p = "encoder.layer.0."
        linear(p + "attention.self.query_proj", input: h, output: h, seed: 3)
        linear(p + "attention.self.key_proj", input: h, output: h, seed: 4)
        linear(p + "attention.self.value_proj", input: h, output: h, seed: 5)
        linear(p + "attention.output.dense", input: h, output: h, seed: 6)
        linear(p + "intermediate.dense", input: h, output: intermediate, seed: 7)
        linear(p + "output.dense", input: intermediate, output: h, seed: 8)
        ln(p + "attention.output.LayerNorm"); ln(p + "output.LayerNorm")
        func rows(_ values: [Float], width: Int) -> [[Float]] {
            stride(from: 0, to: values.count, by: width).map { Array(values[$0..<$0 + width]) }
        }
        func project(_ x: [[Float]], _ name: String) -> [[Float]] {
            let w = raw[name + ".weight"]!, b = raw[name + ".bias"]!, input = x[0].count
            return x.map { row in b.indices.map { o in
                var value = Float(0)
                for k in 0..<input { value += row[k] * w[o * input + k] }
                return value + b[o]
            }}
        }
        func normalize(_ x: [[Float]], _ name: String) -> [[Float]] {
            let w = raw[name + ".weight"]!, b = raw[name + ".bias"]!
            return x.map { row in
                let mean = row.reduce(Float(0), +) / Float(row.count)
                let variance = row.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(row.count)
                return row.indices.map { (row[$0] - mean) / sqrt(variance + 1e-5) * w[$0] + b[$0] }
            }
        }
        func add(_ a: [[Float]], _ b: [[Float]]) -> [[Float]] { zip(a, b).map { x, y in zip(x, y).map(+) } }
        let mask: [Float] = [1, 1, 0]
        let embedded = normalize(rows(raw["embeddings.word_embeddings.weight"]!, width: h), "embeddings.LayerNorm")
        let hidden = embedded.enumerated().map { i, row in row.map { $0 * mask[i] } }
        let relative = normalize(rows(raw["encoder.rel_embeddings.weight"]!, width: h), "encoder.LayerNorm")
        let pq = project(relative, p + "attention.self.query_proj"), pk = project(relative, p + "attention.self.key_proj")
        let q = project(hidden, p + "attention.self.query_proj"), k = project(hidden, p + "attention.self.key_proj")
        let v = project(hidden, p + "attention.self.value_proj")
        var context = Array(repeating: Array(repeating: Float(0), count: h), count: t)
        for head in 0..<heads { for i in 0..<t {
            var scores = Array(repeating: Float(0), count: t)
            for j in 0..<t {
                let bucket = DebertaV2.logBucketPosition(i - j, bucketSize: span, maxPosition: 8)
                let r = min(max(bucket + span, 0), 2 * span - 1)
                var content: Float = 0, c2p: Float = 0, p2c: Float = 0
                for d in 0..<hd {
                    let offset = head * hd + d
                    content += q[i][offset] * k[j][offset]
                    c2p += q[i][offset] * pk[r][offset]
                    p2c += k[j][offset] * pq[r][offset]
                }
                scores[j] = mask[i] * mask[j] == 0 ? -Float.greatestFiniteMagnitude : ((content + c2p) + p2c) * Float(1 / sqrt(Double(hd * 3)))
            }
            let peak = scores.max()!, expScores = scores.map { exp($0 - peak) }, total = expScores.reduce(Float(0), +)
            for d in 0..<hd { for j in 0..<t { context[i][head * hd + d] += expScores[j] / total * v[j][head * hd + d] } }
        }}
        let attended = normalize(add(project(context, p + "attention.output.dense"), hidden), p + "attention.output.LayerNorm")
        let activated = project(attended, p + "intermediate.dense").map { row in row.map { 0.5 * $0 * (1 + erf($0 / sqrt(Float(2)))) } }
        let expected = normalize(add(project(activated, p + "output.dense"), attended), p + "output.LayerNorm").flatMap { $0 }
        let encoder = try DebertaV2(weights: WeightStore(arrays), config: config)
        let output = try encoder.forward(inputIds: MLXArray([Int32(0), 1, 2], [1, t]),
                                         attentionMask: MLXArray([Int32(1), 1, 0], [1, t])).asArray(Float.self)
        for (actual, expected) in zip(output, expected) { XCTAssertEqual(actual, expected, accuracy: 5e-5) }
    }
}
