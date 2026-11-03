import Foundation
import MLX

/// A local GLiNER2.5 Boundary/DeBERTa checkpoint running on MLX.
/// Call a model serially, for example from a dedicated actor. No network access occurs here.
public final class GLiNERModel {
    public let config: ExtractorConfig
    public let encoderConfig: DebertaConfig
    public private(set) var processor: Processor
    public let precision: ModelPrecision
    public let maxInputTokens: Int
    public private(set) var runtimeConfiguration: RuntimeConfiguration
    private let encoder: DebertaV2
    private let boundaryHead: BoundaryHead
    private let classifier: ClassificationHead
    private let relationScorer: RelationScorer?
    private let recordHead: RecordHead?

    public init(modelDirectory: URL, precision: ModelPrecision = .float32, maxInputTokens: Int = 4096,
                runtimeConfiguration: RuntimeConfiguration = .platformDefault) throws {
        try GLiNERRuntime.configure(runtimeConfiguration)
        guard maxInputTokens > 0 else { throw GLiNERError.invalidInput("maxInputTokens must be positive") }
        let configData = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let rawConfig = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        guard rawConfig?["boundary_head"] != nil else {
            throw GLiNERError.unsupported("This port supports GLiNER2.5 boundary checkpoints; legacy span checkpoints are not supported.")
        }
        config = try ExtractorConfig.decode(json: configData); try config.validate()
        let encoderData = try Data(contentsOf: modelDirectory.appendingPathComponent("encoder_config/config.json"))
        let rawEncoder = try JSONSerialization.jsonObject(with: encoderData) as? [String: Any]
        guard (rawEncoder?["model_type"] as? String ?? "deberta-v2") == "deberta-v2" else {
            throw GLiNERError.unsupported("This port currently supports the DeBERTa-v2 encoder, not \(rawEncoder?["model_type"] ?? "unknown").")
        }
        encoderConfig = try DebertaConfig.decode(json: encoderData); try encoderConfig.validate()
        processor = try autoreleasepool {
            Processor(tokenizer: try UnigramTokenizer(url: modelDirectory.appendingPathComponent("tokenizer.json")))
        }
        self.precision = precision; self.maxInputTokens = maxInputTokens; self.runtimeConfiguration = runtimeConfiguration
        let dtype: DType = precision == .float16 ? .float16 : .float32
        let loaded = try loadArrays(url: modelDirectory.appendingPathComponent("model.safetensors"))
        // Convert at load time, once. Retaining the original F32 embedding after conversion doubles memory.
        let arrays = loaded.mapValues { $0 }
        var converted: [String: MLXArray] = [:]
        for (name, value) in arrays { converted[name] = value.asType(name.hasPrefix("encoder.") ? dtype : .float32) }
        eval(Array(converted.values))
        let weights = WeightStore(converted), hidden = encoderConfig.hiddenSize, h = config.boundaryHead
        GLiNERRuntime.markComputationStarted()
        encoder = try DebertaV2(weights: weights, config: encoderConfig, dtype: dtype)
        boundaryHead = try BoundaryHead(weights: weights, hidden: hidden, config: h)
        classifier = try ClassificationHead(weights: weights, hidden: hidden)
        relationScorer = h.enableRelations ? try RelationScorer(weights: weights, hidden: hidden, config: h) : nil
        recordHead = h.enableRecords ? try RecordHead(weights: weights, hidden: hidden, recordDim: h.recordDim,
                                                     instanceQueries: h.recordInstanceQueries) : nil
    }

    public var wordSplitter: WordSplitter {
        get { processor.wordSplitter }
        set { processor.wordSplitter = newValue }
    }

    /// Update buffer reuse without reloading weights. TF32 still requires a new process.
    public func updateRuntimeConfiguration(_ configuration: RuntimeConfiguration) throws {
        try GLiNERRuntime.configure(configuration)
        runtimeConfiguration = configuration
    }

    private func requestScope<T>(_ operation: () throws -> T) throws -> T {
        try GLiNERRuntime.configure(runtimeConfiguration)
        defer { if runtimeConfiguration.clearCacheAfterRequest { GLiNERRuntime.releaseUnusedMemory() } }
        return try autoreleasepool(invoking: operation)
    }

    public func prepare(_ text: String, schema: Schema, maxWords: Int? = nil) throws -> PreparedInput {
        if let maxWords, maxWords < 0 { throw GLiNERError.invalidInput("maxWords must be nonnegative") }
        var expanded = schema; expanded.ensureAttributeEntities()
        let input = try processor.prepare(text: text, schema: expanded, maxWords: maxWords)
        try checkInput(input)
        return input
    }

    private func checkInput(_ input: PreparedInput) throws {
        guard !input.inputIds.isEmpty else { throw GLiNERError.invalidInput("The prepared input is empty") }
        guard input.inputIds.count <= maxInputTokens else {
            throw GLiNERError.invalidInput("Input contains \(input.inputIds.count) tokens including the schema; limit is \(maxInputTokens). Use extractLong or a smaller schema.")
        }
        guard input.inputIds.allSatisfy({ $0 >= 0 && Int($0) < encoderConfig.vocabSize }) else {
            throw GLiNERError.invalidInput("Tokenizer produced a token outside the encoder vocabulary")
        }
    }

    /// Exposed for reproducible numeric comparison with the Rust reference. Returns [tokens, hidden].
    public func encode(_ input: PreparedInput) throws -> MLXArray {
        try checkInput(input)
        let n = input.inputIds.count
        let states = try encoder.forward(inputIds: MLXArray(input.inputIds).reshaped([1, n]),
                                         attentionMask: MLXArray.ones([1, n], dtype: .int32))
        let out = states.squeezed(axis: 0).asType(.float32); eval(out)
        return out
    }

    private func decodeRecords(_ pass: Pass, group: Int, spec: StructureSpec) throws -> [JSONValue] {
        guard let recordHead, let forward = pass.forward, let qs = pass.queryStates,
              let candidates = try boundaryHead.candidateStates(forward) else { return [] }
        let mode: RecordMode = spec.mode == .latent ? .latent : spec.mode == .anchorless ? .anchorless : .natural
        var fields: [RecordField] = []
        for (fi, field) in spec.fields.enumerated() {
            guard let q = pass.query(group, fi) else { return [] }
            let anchor = mode == .natural && (spec.anchor.map { $0 == field.name } ?? (fi == 0))
            let cardinality: Cardinality = field.cardinality ?? (anchor ? .requiredOne : field.dtype == .str ? .optionalOne : .zeroOrMore)
            fields.append(RecordField(query: q, cardinality: cardinality, isAnchor: anchor, exclusive: field.exclusive ?? false))
        }
        if fields.isEmpty || (mode == .natural && !fields.contains(where: { $0.isAnchor })) { return [] }
        let fieldQueries = qs.take(indexArray(fields.map(\.query)), axis: 0)
        let objects: [Float], instances: MLXArray, spans: [CandidateSpan]?
        switch mode {
        case .natural:
            objects = forward.logits[fields.first { $0.isAnchor }!.query]; instances = candidates; spans = forward.spans
        case .latent:
            objects = try recordHead.latentSeedScores(candidates); instances = candidates; spans = forward.spans
        case .anchorless:
            instances = try recordHead.anchorlessInstances(candidates); objects = try recordHead.objectScores(instances); spans = nil
        }
        let assign = try recordHead.assignLogits(instanceStates: instances, candidateStates: candidates, fieldQueries: fieldQueries)
        let records = decodeRecordGroup(mode: mode, fields: fields, objectLogits: objects, spans: spans, assign: assign,
                                        threshold: pass.options.threshold, temperature: config.boundaryHead.recordTemperature)
        let anchors: [(Int, Int)?] = records.map { record in
            guard let candidate = record.anchor, let mention = pass.mention(forward.spans[candidate]) else { return nil }
            return (mention.start, mention.end)
        }
        var results: [JSONValue] = []
        for (index, record) in records.enumerated() {
            var instance: [String: JSONValue] = [:]
            for (fi, field) in spec.fields.enumerated() {
                instance[field.name] = try formatRecordField(pass, field: field, recordField: fields[fi], selected: record.fields[fi] ?? [],
                                                           recordIndex: index, anchors: anchors)
            }
            if instance.values.contains(where: { !$0.isEmptyValue }) { results.append(.object(instance)) }
        }
        return results
    }

    private func formatRecordField(_ pass: Pass, field: FieldSpec, recordField: RecordField, selected: [(Int, Float)],
                                   recordIndex: Int, anchors: [(Int, Int)?]) throws -> JSONValue {
        guard let forward = pass.forward else { return .null }
        let scalar = recordField.cardinality == .optionalOne || recordField.cardinality == .requiredOne || field.dtype == .str
        let allowsAbsent = recordField.cardinality == .optionalOne || recordField.cardinality == .zeroOrMore
        let threshold = pass.options.threshold
        var mentions: [Mention] = []
        for (candidate, assignment) in selected {
            guard var mention = pass.mention(forward.spans[candidate]) else { continue }
            let p = pairProbability(forward.logits[recordField.query][candidate]); mention.score = min(p, assignment)
            if (allowsAbsent && p < threshold) || field.threshold.map({ mention.score < $0 }) == true || !field.validators.allSatisfy({ $0.validate(mention.text) }) { continue }
            mentions.append(mention)
        }
        mentions = finalizeMentions(mentions, scalar: scalar, policy: pass.policy)
        if let choices = field.choices {
            if let anchor = anchors[recordIndex] {
                let (hasLiteral, local) = recordLocalChoiceMentions(text: pass.input.text, choices: choices, anchors: anchors)
                var preferred = local[recordIndex] ?? []
                if scalar && !preferred.isEmpty {
                    func distance(_ mention: (String, Int, Int)) -> Int { max(0, anchor.0 - mention.2, mention.1 - anchor.1) }
                    var best = 0
                    for i in preferred.indices where distance(preferred[i]) < distance(preferred[best]) { best = i }
                    preferred = [preferred[best]]
                }
                if !preferred.isEmpty {
                    return try decodeChoiceField(pass, query: recordField.query, choices: choices, scalar: scalar,
                                                 threshold: field.threshold ?? threshold, preferred: preferred, includeSpans: true)
                }
                if hasLiteral { return scalar ? .null : .array([]) }
            }
            let matched = mentions.compactMap { m -> Mention? in
                guard let choice = choices.first(where: { rustLowercased($0) == rustLowercased(m.text) }) else { return nil }
                return Mention(text: choice, score: m.score, start: m.start, end: m.end)
            }
            if !matched.isEmpty { return formatStructureField(matched, scalar: scalar, options: pass.options) }
            return try decodeChoiceField(pass, query: recordField.query, choices: choices, scalar: scalar,
                                         threshold: field.threshold ?? threshold, includeSpans: true)
        }
        return formatStructureField(mentions, scalar: scalar, options: pass.options)
    }

    private func decodeLegacyStructure(_ pass: Pass, group: Int, spec: StructureSpec) throws -> JSONValue? {
        guard let forward = pass.forward else { return nil }
        var instance: [String: JSONValue] = [:]
        for (fi, field) in spec.fields.enumerated() {
            guard let q = pass.query(group, fi) else { continue }
            let scalar = field.dtype == .str, threshold = field.threshold ?? pass.options.threshold
            if let choices = field.choices {
                instance[field.name] = try decodeChoiceField(pass, query: q, choices: choices, scalar: scalar, threshold: threshold)
            } else {
                let mentions = resolveOverlaps(thresholded(forward, query: q, threshold: threshold), policy: pass.policy).compactMap { s -> Mention? in
                    guard var m = pass.mention(CandidateSpan(start: s.start, end: s.end)), field.validators.allSatisfy({ $0.validate(m.text) }) else { return nil }
                    m.score = s.score; return m
                }
                instance[field.name] = formatStructureField(mentions, scalar: scalar, options: pass.options)
            }
        }
        return instance.values.contains(where: { !$0.isEmptyValue }) ? .object(instance) : nil
    }

    private func decodeChoiceField(_ pass: Pass, query: Int, choices: [String], scalar: Bool, threshold: Float,
                                   preferred: [(String, Int, Int)]? = nil, includeSpans: Bool = false) throws -> JSONValue {
        let empty: JSONValue = scalar ? .null : .array([])
        guard let forward = pass.forward, let ts = pass.textStates, let qs = pass.queryStates else { return empty }
        var present: [(String, Int)] = []
        for choice in choices where !present.contains(where: { $0.0 == choice }) {
            if let i = pass.input.prefixTokens.firstIndex(where: { rustLowercased($0) == rustLowercased(choice) }) { present.append((choice, i)) }
        }
        if present.isEmpty { return empty }
        let probs = try boundaryHead.scoreExplicitSpans(output: forward, text: ts, query: qs, queryIndex: query,
                                                        spans: present.map { CandidateSpan(start: $0.1, end: $0.1 + 1) }).map(pairProbability)
        if let preferred {
            let chosen = preferred.compactMap { item -> JSONValue? in
                guard let i = present.firstIndex(where: { $0.0 == item.0 }) else { return nil }
                var opts = pass.options; opts.includeSpans = includeSpans
                return Mention(text: item.0, score: probs[i], start: item.1, end: item.2).formatted(opts)
            }
            if !chosen.isEmpty { return scalar ? chosen[0] : .array(chosen) }
        }
        func plain(_ i: Int) -> JSONValue {
            pass.options.includeConfidence ? .object(["text": .string(present[i].0), "confidence": .number(Double(probs[i]))]) : .string(present[i].0)
        }
        if !scalar { return .array(probs.indices.filter { probs[$0] >= threshold }.map(plain)) }
        let best = firstArgmax(probs); return probs[best] < threshold ? .null : plain(best)
    }

    private func decodeRelations(_ pass: Pass, schema: Schema) throws -> JSONValue {
        var result: [String: JSONValue] = [:]
        for relation in schema.relations { result[relation.name] = .array([]) }
        guard let scorer = relationScorer, let forward = pass.forward, let ts = pass.textStates, let qs = pass.queryStates else { return .object(result) }
        let h = config.boundaryHead
        var relationStates: [MLXArray] = [], specs: [RelationSpec] = [], pairs: [RelationPair] = []
        for (gi, group) in pass.input.groups.enumerated() where group.kind == .relations {
            guard let head = pass.query(gi, 0), let tail = pass.query(gi, 1) else { continue }
            let headState = qs.take(MLXArray(Int32(head)), axis: 0), tailState = qs.take(MLXArray(Int32(tail)), axis: 0)
            relationStates.append(h.directionalRelationStates ? concatenated([headState, tailState], axis: 0) : (headState + tailState) * 0.5)
            specs.append(schema.relations[group.specIndex])
            func arguments(_ q: Int) -> [ScoredSpan] {
                zip(forward.spans, forward.logits[q]).map { s, x in ScoredSpan(start: s.start, end: s.end, score: scalarSigmoid(x)) }
            }
            var proposed = proposeRelationPairs(heads: arguments(head), tails: arguments(tail), headsPerType: h.relationHeadsPerType,
                                                tailsPerType: h.relationTailsPerType, pairCap: h.relationPairCap,
                                                proposalThreshold: h.relationArgumentProposalThreshold)
            for i in proposed.indices { proposed[i].relation = specs.count - 1 }; pairs += proposed
        }
        if pairs.isEmpty { return .object(result) }
        let logits = try scorer.forward(text: ts, relationStates: stacked(relationStates, axis: 0), pairs: pairs)
        var edges: [String: [RelationEdge]] = [:]
        for (pair, logit) in zip(pairs, logits) {
            let spec = specs[pair.relation], score = scalarSigmoid(logit / h.relationTemperature)
            guard score >= spec.threshold ?? pass.options.threshold,
                  let head = pass.mention(CandidateSpan(start: pair.headStart, end: pair.headEnd)),
                  let tail = pass.mention(CandidateSpan(start: pair.tailStart, end: pair.tailEnd)) else { continue }
            edges[spec.name, default: []].append(RelationEdge(headStart: head.start, headEnd: head.end, tailStart: tail.start,
                                                             tailEnd: tail.end, score: score, headText: head.text, tailText: tail.text))
        }
        for (name, values) in edges {
            result[name] = .array(deduplicateRelationEdges(values).map { edge in
                if !pass.options.includeSpans && !pass.options.includeConfidence { return .array([.string(edge.headText), .string(edge.tailText)]) }
                return .object([
                    "head": Mention(text: edge.headText, score: edge.score, start: edge.headStart, end: edge.headEnd).formatted(pass.options),
                    "tail": Mention(text: edge.tailText, score: edge.score, start: edge.tailStart, end: edge.tailEnd).formatted(pass.options)
                ])
            })
        }
        return .object(result)
    }

    public func extract(_ text: String, schema: Schema, options: ExtractOptions = .init()) throws -> JSONValue {
        try requestScope {
        try validate(schema: schema, options: options)
        var expanded = schema; expanded.ensureAttributeEntities()
        let input = try prepare(text, schema: expanded, maxWords: options.maxWords)
        return try extractPrepared(input, states: encode(input), schema: expanded, options: options)
        }
    }

    public func extractEntities(_ text: String, labels: [String], options: ExtractOptions = .init()) throws -> JSONValue {
        try extract(text, schema: Schema(entities: labels), options: options)
    }
    public func classifyText(_ text: String, tasks: [ClassificationSpec], options: ExtractOptions = .init()) throws -> JSONValue {
        var schema = Schema(); schema.classifications = tasks
        return try extract(text, schema: schema, options: options)
    }
    public func extractRelations(_ text: String, relations: [String], options: ExtractOptions = .init()) throws -> JSONValue {
        var schema = Schema(); schema.relations = relations.map { RelationSpec(name: $0) }
        return try extract(text, schema: schema, options: options)
    }
    public func extractJSON(_ text: String, structures: [StructureSpec], options: ExtractOptions = .init()) throws -> JSONValue {
        var schema = Schema(); schema.structures = structures
        return try extract(text, schema: schema, options: options)
    }

    public func classificationProbabilities(_ text: String, tasks: [ClassificationSpec]) throws -> [(String, [(String, Float)])] {
        try requestScope {
        var schema = Schema(); schema.classifications = tasks
        try validate(schema: schema, options: .init())
        let input = try prepare(text, schema: schema), states = try encode(input)
        return try input.groups.filter { $0.kind == .classifications }.map { group in
            let spec = tasks[group.specIndex]
            let logits = try classifier.forward(states.take(indexArray(group.markerPositions), axis: 0))
                .map { $0 / config.boundaryHead.classificationTemperature }
            return (spec.task, Array(zip(spec.labels, probabilities(spec, logits))))
        }
        }
    }

    public func extractBatch(_ texts: [String], schema: Schema, options: ExtractOptions = .init()) throws -> [JSONValue] {
        try requestScope {
        if texts.isEmpty { return [] }
        try validate(schema: schema, options: options)
        var expanded = schema; expanded.ensureAttributeEntities()
        let inputs = try texts.map { try prepare($0, schema: expanded, maxWords: options.maxWords) }
        let width = inputs.map { $0.inputIds.count }.max()!, count = inputs.count
        var ids = [Int32](repeating: 0, count: count * width), mask = ids
        for (i, input) in inputs.enumerated() { for (j, id) in input.inputIds.enumerated() { ids[i * width + j] = id; mask[i * width + j] = 1 } }
        let states = try encoder.forward(inputIds: MLXArray(ids).reshaped([count, width]),
                                         attentionMask: MLXArray(mask).reshaped([count, width])).asType(.float32)
        eval(states)
        return try inputs.enumerated().map { i, input in
            let row = states.take(MLXArray(Int32(i)), axis: 0).take(indexArray(Array(0..<input.inputIds.count)), axis: 0)
            return try extractPrepared(input, states: row, schema: expanded, options: options)
        }
        }
    }

    private func validate(schema: Schema, options: ExtractOptions) throws {
        guard options.threshold.isFinite, (0...1).contains(options.threshold) else { throw GLiNERError.invalidInput("threshold must be between 0 and 1") }
        guard !(schema.entities.isEmpty && schema.structures.isEmpty && schema.relations.isEmpty && schema.classifications.isEmpty) else {
            throw GLiNERError.invalidInput("Provide at least one entity, classification, relation or structure")
        }
        guard schema.classifications.allSatisfy({ !$0.labels.isEmpty }) else { throw GLiNERError.invalidInput("Classification tasks need at least one label") }
        if !schema.relations.isEmpty && relationScorer == nil { throw GLiNERError.unsupported("This checkpoint has no relation scorer") }
    }

    private struct Pass {
        let input: PreparedInput
        let slots: [(Int, Int)]
        let forward: BoundaryOutput?
        let textStates: MLXArray?
        let queryStates: MLXArray?
        let policy: OverlapPolicy
        let options: ExtractOptions
        func query(_ group: Int, _ field: Int) -> Int? { slots.firstIndex { $0.0 == group && $0.1 == field } }
        func mention(_ span: CandidateSpan) -> Mention? {
            let start = span.start - input.prefixTokens.count, end = span.end - input.prefixTokens.count
            guard start >= 0, start < end, end <= input.words.count else { return nil }
            let first = input.words[start], last = input.words[end - 1], bytes = Array(input.text.utf8)
            let surface = String(decoding: bytes[first.byteStart..<last.byteEnd], as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !surface.isEmpty else { return nil }
            return Mention(text: surface, score: 0, start: first.charStart, end: last.charEnd)
        }
    }

    private func extractPrepared(_ input: PreparedInput, states: MLXArray, schema: Schema, options: ExtractOptions) throws -> JSONValue {
        let policy = options.overlapPolicy ?? OverlapPolicy(rawValue: config.boundaryHead.overlapPolicy) ?? .flat
        var slots: [(Int, Int)] = [], positions: [Int] = []
        for (gi, group) in input.groups.enumerated() where group.kind != .classifications {
            for (fi, pos) in group.markerPositions.enumerated() { slots.append((gi, fi)); positions.append(pos) }
        }
        var ts: MLXArray?, qs: MLXArray?, forward: BoundaryOutput?
        if !slots.isEmpty && !input.wordPositions.isEmpty {
            ts = states.take(indexArray(input.wordPositions), axis: 0)
            qs = states.take(indexArray(positions), axis: 0)
            forward = try boundaryHead.forward(text: ts!, query: qs!)
        }
        let pass = Pass(input: input, slots: slots, forward: forward, textStates: ts, queryStates: qs, policy: policy, options: options)
        var out: [String: JSONValue] = [:]
        for (gi, group) in input.groups.enumerated() where group.kind == .structures {
            let spec = schema.structures[group.specIndex]
            if recordHead != nil && spec.mode != .legacy {
                let records = try decodeRecords(pass, group: gi, spec: spec)
                if !records.isEmpty { out[spec.name] = .array(records) }
            } else if let record = try decodeLegacyStructure(pass, group: gi, spec: spec) { out[spec.name] = .array([record]) }
        }
        if let gi = input.groups.firstIndex(where: { $0.kind == .entities }) { out["entities"] = try decodeEntities(pass, schema: schema, group: gi) }
        for group in input.groups where group.kind == .classifications && !group.markerPositions.isEmpty {
            let spec = schema.classifications[group.specIndex]
            let logits = try classifier.forward(states.take(indexArray(group.markerPositions), axis: 0))
                .map { $0 / config.boundaryHead.classificationTemperature }
            let probs = probabilities(spec, logits)
            let entry: (Int) -> JSONValue = { i in options.includeConfidence
                ? .object(["label": .string(spec.labels[i]), "confidence": .number(Double(probs[i]))]) : .string(spec.labels[i]) }
            if spec.multiLabel {
                var chosen = probs.indices.filter { probs[$0] >= spec.clsThreshold }
                if chosen.isEmpty { chosen = [firstArgmax(probs)] }
                out[spec.task] = .array(chosen.map(entry))
            } else { out[spec.task] = entry(firstArgmax(probs)) }
        }
        if !schema.relations.isEmpty { out["relation_extraction"] = try decodeRelations(pass, schema: schema) }
        return .object(out)
    }

    private func probabilities(_ spec: ClassificationSpec, _ logits: [Float]) -> [Float] {
        let independent = spec.activation == .sigmoid || (spec.activation == .auto && spec.multiLabel)
        return independent ? logits.map(scalarSigmoid) : scalarSoftmax(logits)
    }
    private func pairProbability(_ value: Float) -> Float { scalarSigmoid(value / config.boundaryHead.pairTemperature) }
    private func thresholded(_ forward: BoundaryOutput, query: Int, threshold: Float) -> [ScoredSpan] {
        zip(forward.spans, forward.logits[query]).compactMap { span, logit in
            let p = pairProbability(logit)
            return p >= threshold ? ScoredSpan(start: span.start, end: span.end, score: p) : nil
        }
    }

    private func decodeEntities(_ pass: Pass, schema: Schema, group: Int) throws -> JSONValue {
        var decoded: [(EntitySpec, [Mention])] = [], tokenSpans: [[Int]: CandidateSpan] = [:]
        for (fi, spec) in schema.entities.enumerated() {
            var mentions: [Mention] = []
            if let q = pass.query(group, fi), let forward = pass.forward {
                let abstained = forward.noMatchLogits.map { scalarSigmoid($0[q]) > config.boundaryHead.abstentionThreshold } ?? false
                if !abstained {
                    for span in resolveOverlaps(thresholded(forward, query: q, threshold: spec.threshold ?? pass.options.threshold), policy: pass.policy) {
                        let token = CandidateSpan(start: span.start, end: span.end)
                        if var m = pass.mention(token), spec.validators.allSatisfy({ $0.validate(m.text) }) {
                            m.score = span.score; mentions.append(m); tokenSpans[[m.start, m.end]] = token
                        }
                    }
                }
            }
            if !spec.isAttribute { decoded.append((spec, mentions)) }
        }
        let attributes = try entityAttributes(pass, schema: schema, group: group, decoded: decoded, tokenSpans: tokenSpans)
        var out: [String: JSONValue] = [:]
        for (spec, mentions) in decoded where out[spec.name] == nil {
            let values = mentions.map { $0.formatted(pass.options, attributes: attributes[[$0.start, $0.end]] ?? [:]) }
            out[spec.name] = spec.dtype == .str ? values.first ?? .null : .array(uniqueJSON(values))
        }
        return .object(out)
    }

    private func entityAttributes(_ pass: Pass, schema: Schema, group: Int, decoded: [(EntitySpec, [Mention])],
                                  tokenSpans: [[Int]: CandidateSpan]) throws -> [[Int]: [String: JSONValue]] {
        guard !schema.entityAttributes.isEmpty, let forward = pass.forward, let ts = pass.textStates, let qs = pass.queryStates else { return [:] }
        var spans: [[Int]] = []
        for (_, mentions) in decoded { for m in mentions where !spans.contains([m.start, m.end]) { spans.append([m.start, m.end]) } }
        if spans.isEmpty { return [:] }
        let tokens = spans.map { tokenSpans[$0]! }
        var out: [[Int]: [String: JSONValue]] = [:]
        for named in schema.entityAttributes {
            let spec = named.group
            var labels: [(String, Int)] = []
            for label in spec.labels {
                let prompt = spec.qualifyLabels ? "\(named.name): \(label)" : label
                if let field = schema.entities.firstIndex(where: { $0.name == prompt }), let q = pass.query(group, field) { labels.append((label, q)) }
            }
            if labels.isEmpty { continue }
            let logits = try labels.map { _, q in
                try boundaryHead.scoreExplicitSpans(output: forward, text: ts, query: qs, queryIndex: q, spans: tokens)
                    .map { $0 / config.boundaryHead.pairTemperature }
            }
            for (entity, mentions) in decoded {
                if let applies = spec.appliesTo, !applies.contains(entity.name) { continue }
                for mention in mentions {
                    let key = [mention.start, mention.end], column = spans.firstIndex(of: key)!
                    let values = logits.map { $0[column] }
                    let value: JSONValue
                    if spec.multiLabel {
                        value = .array(values.indices.compactMap { i in
                            let p = scalarSigmoid(values[i]); return p >= spec.threshold ? .object(["label": .string(labels[i].0), "confidence": .number(Double(p))]) : nil
                        })
                    } else {
                        let probs = scalarSoftmax(values), best = firstArgmax(probs)
                        value = .object(["label": .string(labels[best].0), "confidence": .number(Double(probs[best]))])
                    }
                    out[key, default: [:]][named.name] = value
                }
            }
        }
        return out
    }
}
