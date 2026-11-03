import XCTest
import MLX
@testable import GLiNERMLX

final class BoundaryTests: XCTestCase {
    func testDedupKeepsBestOccurrenceAndDropsInvalid() {
        let keys = [5, 3, 5, 7], scores: [Float] = [1, 2, 3, 9], valid = [true, true, true, false]
        XCTAssertEqual(deduplicateBoundaryPool(keys: keys, scores: scores, valid: valid, capacity: 10, n: 4), [5, 3])
        XCTAssertEqual(deduplicateBoundaryPool(keys: keys, scores: scores, valid: valid, capacity: 1, n: 4), [5])
    }
    func testDedupTiesBreakByKey() {
        XCTAssertEqual(deduplicateBoundaryPool(keys: [9, 2, 4], scores: [1, 1, 1], valid: [true, true, true], capacity: 10, n: 4), [2, 4, 9])
    }
    func testTopKPreservesOriginalIndexForTies() {
        XCTAssertEqual(stableScoreOrder([3, 3, 5, -1, 5]), [2, 4, 0, 1, 3])
        XCTAssertEqual(deduplicateBoundaryPool(keys: [], scores: [], valid: [], capacity: 8, n: 3), [])
    }
}

extension BoundaryTests {
    private func smallConfig() -> BoundaryHeadConfig {
        BoundaryHeadConfig(boundaryDim: 4, boundaryAttentionHeads: 2, boundaryAttentionLayers: 1,
            boundaryAttentionWindow: 1, boundaryRefinementLayers: 1, boundaryFfnMultiplier: 1,
            pairDim: 4, contentDim: 2, contentSoftMaxPool: false, enableSpanContent: true,
            useInsideEvidence: true, candidatePool: "shared", candidateAttentionLayers: 0,
            queryAttentionLayers: 0, poolBoundaryTopK: 4, poolSize: 4, minPoolPerQuery: 1,
            enableAbstention: true, abstentionThreshold: 0.5, adaptiveThreshold: false,
            pairTemperature: 1, classificationTemperature: 1, relationTemperature: 1,
            enableRelations: true, directionalRelationStates: true, relationBiaffineContent: true,
            relationHeadsPerType: 4, relationTailsPerType: 4, relationPairCap: 16,
            relationArgumentProposalThreshold: 0.1, overlapPolicy: "none", enableRotaryEndpoints: true,
            rotaryBase: 10000, multiheadPairCompatHeads: 2, queryConditionedInsideWeight: true,
            endpointDifferenceFeatures: true, rerankerEndpointCompat: true, enableRecords: true,
            recordDim: 4, recordTemperature: 1)
    }
    func testCompleteBoundaryForwardAndExplicitReranking() throws {
        var values: [String: MLXArray] = [:]
        func linear(_ name: String, _ input: Int, _ output: Int) {
            values["boundary_head." + name + ".weight"] = .zeros([output, input])
            values["boundary_head." + name + ".bias"] = .zeros([output])
        }
        func norm(_ name: String, _ size: Int) {
            values["boundary_head." + name + ".weight"] = .ones([size])
            values["boundary_head." + name + ".bias"] = .zeros([size])
        }
        values["boundary_head.boundary_encoder.bos_state"] = .zeros([4])
        values["boundary_head.boundary_encoder.eos_state"] = .zeros([4])
        linear("boundary_encoder.left_projection", 4, 4); linear("boundary_encoder.right_projection", 4, 4)
        linear("boundary_encoder.output_projection", 8, 4); norm("boundary_encoder.layer_norm", 4)
        norm("boundary_encoder.attention_blocks.0.norm", 4)
        linear("boundary_encoder.attention_blocks.0.qkv_projection", 4, 12)
        linear("boundary_encoder.attention_blocks.0.output_projection", 4, 4)
        norm("boundary_encoder.refinement_blocks.0.norm", 4)
        linear("boundary_encoder.refinement_blocks.0.input_projection", 4, 8)
        linear("boundary_encoder.refinement_blocks.0.output_projection", 4, 4)
        for endpoint in ["start", "end"] {
            linear("boundary_query_head.\(endpoint)_boundary_projection", 4, 4)
            linear("boundary_query_head.\(endpoint)_query_projection", 4, 4)
            linear("shared_pool_builder.\(endpoint)_projection", 4, 4)
            linear("shared_pool_scorer.\(endpoint)_projection", 4, 4)
        }
        linear("boundary_query_head.inside_text_projection", 4, 4)
        linear("boundary_query_head.inside_query_projection", 4, 4)
        linear("shared_pool_scorer.length_projection", 3, 4)
        linear("shared_pool_scorer.prior_projection", 1, 4)
        linear("shared_pool_scorer.content_pooler.value_projection", 4, 2)
        norm("shared_pool_scorer.content_pooler.layer_norm", 2)
        linear("shared_pool_scorer.content_projection", 2, 4)
        norm("shared_pool_scorer.candidate_norm", 4)
        linear("shared_pool_scorer.query_projection", 4, 4)
        linear("shared_pool_scorer.film", 4, 8)
        linear("shared_pool_scorer.film_output.0", 4, 64)
        linear("shared_pool_scorer.film_output.3", 64, 1)
        values["boundary_head.shared_pool_scorer.film_output.3.bias"] = MLXArray([Float(0.75)])
        linear("null_projection", 4, 1)
        values["boundary_head.null_projection.bias"] = MLXArray([Float(0.25)])
        linear("candidate_encoder", 8, 4)
        values["boundary_head.candidate_encoder.bias"] = MLXArray([Float(1), 2, 3, 4])
        linear("boundary_proposer.start_pair_projection", 4, 4)
        linear("boundary_proposer.end_key_projection", 4, 4)
        linear("boundary_proposer.start_query_projection", 4, 2)
        linear("pair_scorer.start_endpoint_projection", 4, 4)
        linear("pair_scorer.end_endpoint_projection", 4, 4)
        linear("pair_scorer.query_gate", 4, 2)
        linear("pair_scorer.length_query_projection", 4, 3)
        linear("pair_scorer.inside_weight", 4, 1)
        linear("pair_scorer.endpoint_difference_projection", 8, 1)
        linear("pair_scorer.content_pooler.value_projection", 4, 2)
        norm("pair_scorer.content_pooler.layer_norm", 2)
        linear("pair_scorer.content_query_projection", 4, 2)
        linear("pair_scorer.content_bias", 2, 1)
        linear("pair_scorer.compat_mix", 2, 1)
        let head = try BoundaryHead(weights: WeightStore(values), hidden: 4, config: smallConfig())
        let text = MLXArray.ones([3, 4]), query = MLXArray.ones([2, 4])
        let output = try head.forward(text: text, query: query)
        XCTAssertEqual(output.spans, [CandidateSpan(start: 0, end: 1), CandidateSpan(start: 0, end: 2),
                                     CandidateSpan(start: 0, end: 3), CandidateSpan(start: 1, end: 2)])
        XCTAssertEqual(output.logits, Array(repeating: Array(repeating: 0.75, count: 4), count: 2))
        XCTAssertEqual(output.noMatchLogits, [0.25, 0.25])
        let states = try XCTUnwrap(head.candidateStates(output))
        XCTAssertEqual(states.shape, [4, 4])
        XCTAssertEqual(states.asArray(Float.self), Array(repeating: [Float(1), 2, 3, 4], count: 4).flatMap { $0 })
        let explicit = try head.scoreExplicitSpans(output: output, text: text, query: query, queryIndex: 1,
                                                  spans: [CandidateSpan(start: 0, end: 3), CandidateSpan(start: 1, end: 2)])
        XCTAssertEqual(explicit, [0, 0])
        XCTAssertEqual(try head.scorePairs(text: text, query: MLXArray.ones([4]), starts: [0], ends: [2]), [0])
    }

    func testRelationDirectionalAndBiaffineBranches() throws {
        var values: [String: MLXArray] = [:]
        func linear(_ name: String, _ input: Int, _ output: Int, bias: Float = 0) {
            values["relation_scorer." + name + ".weight"] = .zeros([output, input])
            values["relation_scorer." + name + ".bias"] = .full([output], values: MLXArray(bias))
        }
        linear("mlp.0", 26, 4); linear("mlp.3", 4, 1, bias: 0.4)
        linear("head_content_projection", 4, 4); linear("tail_content_projection", 4, 4)
        linear("relation_content_gate", 8, 4); linear("content_linear", 16, 1, bias: 0.6)
        let scorer = try RelationScorer(weights: WeightStore(values), hidden: 4, config: smallConfig())
        let pairs = [RelationPair(relation: 0, head: (0, 1), tail: (2, 3)), RelationPair(relation: 1, head: (2, 3), tail: (0, 1))]
        let scores = try scorer.forward(text: .ones([3, 4]), relationStates: .ones([2, 8]), pairs: pairs)
        XCTAssertEqual(scores, [1, 1])
    }
}
