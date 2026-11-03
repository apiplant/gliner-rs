import Foundation

public enum GLiNERError: Error, LocalizedError {
    case invalidConfiguration(String)
    case unsupported(String)
    case invalidInput(String)
    case missingWeight(String)
    case invalidWeights(String)
    case tokenization(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .unsupported(let message), .invalidInput(let message),
             .missingWeight(let message), .invalidWeights(let message), .tokenization(let message): message
        }
    }
}

/// Extractor settings read from a checkpoint's `config.json`.
public struct ExtractorConfig: Codable, Equatable {
    public var architecture: String
    public var tokenPooling: String
    public var boundaryHead: BoundaryHeadConfig

    public init(architecture: String, tokenPooling: String = "first", boundaryHead: BoundaryHeadConfig) {
        self.architecture = architecture
        self.tokenPooling = tokenPooling
        self.boundaryHead = boundaryHead
    }

    private enum CodingKeys: String, CodingKey { case architecture, tokenPooling, boundaryHead }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        architecture = try c.decode(String.self, forKey: .architecture)
        tokenPooling = try c.decodeIfPresent(String.self, forKey: .tokenPooling) ?? "first"
        boundaryHead = try c.decode(BoundaryHeadConfig.self, forKey: .boundaryHead)
    }

    public static func decode(json: Data) throws -> ExtractorConfig {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Self.self, from: json)
    }

    public func validate() throws {
        guard architecture == "boundary" else {
            throw GLiNERError.unsupported("Only the 'boundary' architecture is supported; got '\(architecture)'.")
        }
        guard tokenPooling == "first" else {
            throw GLiNERError.unsupported("Only token_pooling='first' is supported; got '\(tokenPooling)'.")
        }
        try boundaryHead.validate()
    }
}

/// Inference-relevant subset of GLiNER2's boundary-head settings.
public struct BoundaryHeadConfig: Codable, Equatable {
    public var boundaryDim: Int
    public var boundaryAttentionHeads: Int
    public var boundaryAttentionLayers: Int
    public var boundaryAttentionWindow: Int
    public var boundaryRefinementLayers: Int
    public var boundaryFfnMultiplier: Double
    public var pairDim: Int
    public var contentDim: Int
    public var contentSoftMaxPool: Bool
    public var enableSpanContent: Bool
    public var useInsideEvidence: Bool
    public var candidatePool: String
    public var candidateAttentionLayers: Int
    public var queryAttentionLayers: Int
    public var poolBoundaryTopK: Int
    public var poolSize: Int
    public var minPoolPerQuery: Int
    public var enableAbstention: Bool
    public var abstentionThreshold: Float
    public var adaptiveThreshold: Bool
    public var pairTemperature: Float
    public var classificationTemperature: Float
    public var relationTemperature: Float
    public var enableRelations: Bool
    public var directionalRelationStates: Bool
    public var relationBiaffineContent: Bool
    public var relationHeadsPerType: Int
    public var relationTailsPerType: Int
    public var relationPairCap: Int
    public var relationArgumentProposalThreshold: Float
    public var overlapPolicy: String
    public var enableRotaryEndpoints: Bool
    public var rotaryBase: Float
    public var multiheadPairCompatHeads: Int
    public var queryConditionedInsideWeight: Bool
    public var endpointDifferenceFeatures: Bool
    public var rerankerEndpointCompat: Bool
    public var enableRecords: Bool
    public var recordDim: Int
    public var recordTemperature: Float
    public var recordInstanceQueries: Int

    public init(boundaryDim: Int, boundaryAttentionHeads: Int, boundaryAttentionLayers: Int,
                boundaryAttentionWindow: Int, boundaryRefinementLayers: Int, boundaryFfnMultiplier: Double,
                pairDim: Int, contentDim: Int, contentSoftMaxPool: Bool, enableSpanContent: Bool,
                useInsideEvidence: Bool, candidatePool: String, candidateAttentionLayers: Int,
                queryAttentionLayers: Int, poolBoundaryTopK: Int, poolSize: Int, minPoolPerQuery: Int,
                enableAbstention: Bool, abstentionThreshold: Float, adaptiveThreshold: Bool,
                pairTemperature: Float, classificationTemperature: Float, relationTemperature: Float,
                enableRelations: Bool, directionalRelationStates: Bool, relationBiaffineContent: Bool,
                relationHeadsPerType: Int, relationTailsPerType: Int, relationPairCap: Int,
                relationArgumentProposalThreshold: Float, overlapPolicy: String, enableRotaryEndpoints: Bool,
                rotaryBase: Float, multiheadPairCompatHeads: Int, queryConditionedInsideWeight: Bool,
                endpointDifferenceFeatures: Bool, rerankerEndpointCompat: Bool, enableRecords: Bool,
                recordDim: Int, recordTemperature: Float, recordInstanceQueries: Int = 32) {
        self.boundaryDim = boundaryDim; self.boundaryAttentionHeads = boundaryAttentionHeads
        self.boundaryAttentionLayers = boundaryAttentionLayers; self.boundaryAttentionWindow = boundaryAttentionWindow
        self.boundaryRefinementLayers = boundaryRefinementLayers; self.boundaryFfnMultiplier = boundaryFfnMultiplier
        self.pairDim = pairDim; self.contentDim = contentDim; self.contentSoftMaxPool = contentSoftMaxPool
        self.enableSpanContent = enableSpanContent; self.useInsideEvidence = useInsideEvidence
        self.candidatePool = candidatePool; self.candidateAttentionLayers = candidateAttentionLayers
        self.queryAttentionLayers = queryAttentionLayers; self.poolBoundaryTopK = poolBoundaryTopK
        self.poolSize = poolSize; self.minPoolPerQuery = minPoolPerQuery; self.enableAbstention = enableAbstention
        self.abstentionThreshold = abstentionThreshold; self.adaptiveThreshold = adaptiveThreshold
        self.pairTemperature = pairTemperature; self.classificationTemperature = classificationTemperature
        self.relationTemperature = relationTemperature; self.enableRelations = enableRelations
        self.directionalRelationStates = directionalRelationStates; self.relationBiaffineContent = relationBiaffineContent
        self.relationHeadsPerType = relationHeadsPerType; self.relationTailsPerType = relationTailsPerType
        self.relationPairCap = relationPairCap; self.relationArgumentProposalThreshold = relationArgumentProposalThreshold
        self.overlapPolicy = overlapPolicy; self.enableRotaryEndpoints = enableRotaryEndpoints; self.rotaryBase = rotaryBase
        self.multiheadPairCompatHeads = multiheadPairCompatHeads
        self.queryConditionedInsideWeight = queryConditionedInsideWeight
        self.endpointDifferenceFeatures = endpointDifferenceFeatures; self.rerankerEndpointCompat = rerankerEndpointCompat
        self.enableRecords = enableRecords; self.recordDim = recordDim; self.recordTemperature = recordTemperature
        self.recordInstanceQueries = recordInstanceQueries
    }

    private enum CodingKeys: String, CodingKey {
        case boundaryDim, boundaryAttentionHeads, boundaryAttentionLayers, boundaryAttentionWindow
        case boundaryRefinementLayers, boundaryFfnMultiplier, pairDim, contentDim, contentSoftMaxPool
        case enableSpanContent, useInsideEvidence, candidatePool, candidateAttentionLayers, queryAttentionLayers
        case poolBoundaryTopK, poolSize, minPoolPerQuery, enableAbstention, abstentionThreshold, adaptiveThreshold
        case pairTemperature, classificationTemperature, relationTemperature, enableRelations
        case directionalRelationStates, relationBiaffineContent, relationHeadsPerType, relationTailsPerType
        case relationPairCap, relationArgumentProposalThreshold, overlapPolicy, enableRotaryEndpoints, rotaryBase
        case multiheadPairCompatHeads, queryConditionedInsideWeight, endpointDifferenceFeatures
        case rerankerEndpointCompat, enableRecords, recordDim, recordTemperature, recordInstanceQueries
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        boundaryDim = try c.decode(Int.self, forKey: .boundaryDim)
        boundaryAttentionHeads = try c.decode(Int.self, forKey: .boundaryAttentionHeads)
        boundaryAttentionLayers = try c.decode(Int.self, forKey: .boundaryAttentionLayers)
        boundaryAttentionWindow = try c.decode(Int.self, forKey: .boundaryAttentionWindow)
        boundaryRefinementLayers = try c.decode(Int.self, forKey: .boundaryRefinementLayers)
        boundaryFfnMultiplier = try c.decode(Double.self, forKey: .boundaryFfnMultiplier)
        pairDim = try c.decode(Int.self, forKey: .pairDim); contentDim = try c.decode(Int.self, forKey: .contentDim)
        contentSoftMaxPool = try c.decode(Bool.self, forKey: .contentSoftMaxPool)
        enableSpanContent = try c.decode(Bool.self, forKey: .enableSpanContent)
        useInsideEvidence = try c.decode(Bool.self, forKey: .useInsideEvidence)
        candidatePool = try c.decode(String.self, forKey: .candidatePool)
        candidateAttentionLayers = try c.decode(Int.self, forKey: .candidateAttentionLayers)
        queryAttentionLayers = try c.decode(Int.self, forKey: .queryAttentionLayers)
        poolBoundaryTopK = try c.decode(Int.self, forKey: .poolBoundaryTopK); poolSize = try c.decode(Int.self, forKey: .poolSize)
        minPoolPerQuery = try c.decode(Int.self, forKey: .minPoolPerQuery)
        enableAbstention = try c.decode(Bool.self, forKey: .enableAbstention)
        abstentionThreshold = try c.decode(Float.self, forKey: .abstentionThreshold)
        adaptiveThreshold = try c.decode(Bool.self, forKey: .adaptiveThreshold)
        pairTemperature = try c.decode(Float.self, forKey: .pairTemperature)
        classificationTemperature = try c.decode(Float.self, forKey: .classificationTemperature)
        relationTemperature = try c.decode(Float.self, forKey: .relationTemperature)
        enableRelations = try c.decode(Bool.self, forKey: .enableRelations)
        directionalRelationStates = try c.decode(Bool.self, forKey: .directionalRelationStates)
        relationBiaffineContent = try c.decode(Bool.self, forKey: .relationBiaffineContent)
        relationHeadsPerType = try c.decode(Int.self, forKey: .relationHeadsPerType)
        relationTailsPerType = try c.decode(Int.self, forKey: .relationTailsPerType)
        relationPairCap = try c.decode(Int.self, forKey: .relationPairCap)
        relationArgumentProposalThreshold = try c.decode(Float.self, forKey: .relationArgumentProposalThreshold)
        overlapPolicy = try c.decode(String.self, forKey: .overlapPolicy)
        enableRotaryEndpoints = try c.decode(Bool.self, forKey: .enableRotaryEndpoints)
        rotaryBase = try c.decode(Float.self, forKey: .rotaryBase)
        multiheadPairCompatHeads = try c.decode(Int.self, forKey: .multiheadPairCompatHeads)
        queryConditionedInsideWeight = try c.decode(Bool.self, forKey: .queryConditionedInsideWeight)
        endpointDifferenceFeatures = try c.decode(Bool.self, forKey: .endpointDifferenceFeatures)
        rerankerEndpointCompat = try c.decode(Bool.self, forKey: .rerankerEndpointCompat)
        enableRecords = try c.decode(Bool.self, forKey: .enableRecords)
        recordDim = try c.decode(Int.self, forKey: .recordDim); recordTemperature = try c.decode(Float.self, forKey: .recordTemperature)
        recordInstanceQueries = try c.decodeIfPresent(Int.self, forKey: .recordInstanceQueries) ?? 32
    }

    public func validate() throws {
        guard candidatePool == "shared" else { throw GLiNERError.unsupported("Only candidate_pool='shared' is supported; got '\(candidatePool)'.") }
        guard candidateAttentionLayers == 0 && queryAttentionLayers == 0 else {
            throw GLiNERError.unsupported("Candidate/query attention layers in the shared pool scorer are unsupported.")
        }
        guard !contentSoftMaxPool else { throw GLiNERError.unsupported("content_soft_max_pool is unsupported.") }
        guard !adaptiveThreshold else { throw GLiNERError.unsupported("adaptive_threshold decoding is unsupported.") }
        guard enableRotaryEndpoints && queryConditionedInsideWeight && endpointDifferenceFeatures && rerankerEndpointCompat && enableSpanContent else {
            throw GLiNERError.unsupported("Unsupported pair reranker configuration; expected rotary endpoints, query-conditioned inside weight, endpoint difference and compatibility features, and span content.")
        }
        guard multiheadPairCompatHeads > 0, pairDim % multiheadPairCompatHeads == 0,
              boundaryDim % 2 == 0, pairDim % 2 == 0 else {
            throw GLiNERError.invalidConfiguration("pair_dim/boundary_dim are incompatible with rotary endpoints or compatibility heads.")
        }
    }
}

/// DeBERTa-v2 encoder settings.
public struct DebertaConfig: Codable, Equatable {
    public var hiddenSize: Int
    public var numAttentionHeads: Int
    public var numHiddenLayers: Int
    public var intermediateSize: Int
    public var layerNormEps: Double
    public var maxPositionEmbeddings: Int
    public var maxRelativePositions: Int
    public var positionBuckets: Int
    public var vocabSize: Int
    public var relativeAttention: Bool
    public var shareAttKey: Bool
    public var posAttType: [String]
    public var positionBiasedInput: Bool
    public var normRelEbd: String
    public var hiddenAct: String
    public var convKernelSize: Int
    public var typeVocabSize: Int

    public init(hiddenSize: Int, numAttentionHeads: Int, numHiddenLayers: Int, intermediateSize: Int,
                layerNormEps: Double, maxPositionEmbeddings: Int, maxRelativePositions: Int = -1,
                positionBuckets: Int = -1, vocabSize: Int, relativeAttention: Bool = false,
                shareAttKey: Bool = false, posAttType: [String] = [], positionBiasedInput: Bool = true,
                normRelEbd: String = "", hiddenAct: String = "", convKernelSize: Int = 0,
                typeVocabSize: Int = 0) {
        self.hiddenSize = hiddenSize; self.numAttentionHeads = numAttentionHeads; self.numHiddenLayers = numHiddenLayers
        self.intermediateSize = intermediateSize; self.layerNormEps = layerNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings; self.maxRelativePositions = maxRelativePositions
        self.positionBuckets = positionBuckets; self.vocabSize = vocabSize; self.relativeAttention = relativeAttention
        self.shareAttKey = shareAttKey; self.posAttType = posAttType; self.positionBiasedInput = positionBiasedInput
        self.normRelEbd = normRelEbd; self.hiddenAct = hiddenAct; self.convKernelSize = convKernelSize
        self.typeVocabSize = typeVocabSize
    }

    private enum CodingKeys: String, CodingKey {
        case hiddenSize, numAttentionHeads, numHiddenLayers, intermediateSize, layerNormEps, maxPositionEmbeddings
        case maxRelativePositions, positionBuckets, vocabSize, relativeAttention, shareAttKey, posAttType
        case positionBiasedInput, normRelEbd, hiddenAct, convKernelSize, typeVocabSize
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        layerNormEps = try c.decode(Double.self, forKey: .layerNormEps)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        maxRelativePositions = try c.decodeIfPresent(Int.self, forKey: .maxRelativePositions) ?? -1
        positionBuckets = try c.decodeIfPresent(Int.self, forKey: .positionBuckets) ?? -1
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        relativeAttention = try c.decodeIfPresent(Bool.self, forKey: .relativeAttention) ?? false
        shareAttKey = try c.decodeIfPresent(Bool.self, forKey: .shareAttKey) ?? false
        posAttType = try c.decodeIfPresent([String].self, forKey: .posAttType) ?? []
        positionBiasedInput = try c.decodeIfPresent(Bool.self, forKey: .positionBiasedInput) ?? true
        normRelEbd = try c.decodeIfPresent(String.self, forKey: .normRelEbd) ?? ""
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? ""
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 0
        typeVocabSize = try c.decodeIfPresent(Int.self, forKey: .typeVocabSize) ?? 0
    }

    public static func decode(json: Data) throws -> DebertaConfig {
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Self.self, from: json)
    }

    public func maxRelativePositionsResolved() -> Int { maxRelativePositions < 1 ? maxPositionEmbeddings : maxRelativePositions }
    public func posEbdSize() -> Int { positionBuckets > 0 ? positionBuckets : maxRelativePositionsResolved() }

    public func validate() throws {
        guard relativeAttention, shareAttKey, !positionBiasedInput, convKernelSize == 0, typeVocabSize == 0 else {
            throw GLiNERError.unsupported("Unsupported DeBERTa-v2 variant (expected relative attention, shared keys, no convolution, and no absolute positions).")
        }
        guard Set(posAttType) == Set(["c2p", "p2c"]) && posAttType.count == 2 else {
            throw GLiNERError.invalidConfiguration("Expected pos_att_type [p2c, c2p]; got \(posAttType).")
        }
        guard normRelEbd.contains("layer_norm") else { throw GLiNERError.invalidConfiguration("Expected norm_rel_ebd=layer_norm.") }
        guard hiddenAct == "gelu" else { throw GLiNERError.invalidConfiguration("Expected hidden_act=gelu; got '\(hiddenAct)'.") }
    }
}
