import Foundation

private indirect enum OrderedJSON {
    case object([(String, OrderedJSON)]), array([OrderedJSON]), string(String), number(Double), bool(Bool), null
    subscript(_ key: String) -> OrderedJSON? { if case .object(let pairs) = self { return pairs.last { $0.0 == key }?.1 }; return nil }
    var string: String? { if case .string(let s) = self { return s }; return nil }
    var bool: Bool? { if case .bool(let b) = self { return b }; return nil }
    var number: Double? { if case .number(let n) = self { return n }; return nil }
    var array: [OrderedJSON]? { if case .array(let a) = self { return a }; return nil }
    var object: [(String,OrderedJSON)]? { if case .object(let o) = self { return o }; return nil }
}
private struct OrderedParser {
    let bytes: [UInt8]; var i = 0
    init(_ data: Data) throws {
        _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        guard String(data: data, encoding: .utf8) != nil else { throw GLiNERError.invalidInput("Cases JSON must be UTF-8") }
        let raw = Array(data)
        bytes = raw.starts(with: [0xEF, 0xBB, 0xBF]) ? Array(raw.dropFirst(3)) : raw
    }
    mutating func spaces() { while i < bytes.count && [9,10,13,32].contains(bytes[i]) { i += 1 } }
    mutating func parseString() throws -> String {
        let start = i; i += 1; var escaped = false
        while i < bytes.count {
            let byte = bytes[i]; i += 1
            if byte == 34 && !escaped { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<i])) }
            if byte == 92 && !escaped { escaped = true } else { escaped = false }
        }
        throw GLiNERError.invalidInput("Unterminated JSON string")
    }
    mutating func value() throws -> OrderedJSON {
        spaces(); guard i < bytes.count else { throw GLiNERError.invalidInput("Unexpected end of JSON") }
        switch bytes[i] {
        case 34: return .string(try parseString())
        case 123:
            i += 1; spaces(); var result: [(String,OrderedJSON)] = []
            if bytes[i] == 125 { i += 1; return .object(result) }
            while true {
                spaces(); let key = try parseString(); spaces(); i += 1
                let val = try value(); result.append((key,val)); spaces()
                if bytes[i] == 125 { i += 1; break }; i += 1
            }
            return .object(result)
        case 91:
            i += 1; spaces(); var result: [OrderedJSON] = []
            if bytes[i] == 93 { i += 1; return .array(result) }
            while true { result.append(try value()); spaces(); if bytes[i] == 93 { i += 1; break }; i += 1 }
            return .array(result)
        case 116: i += 4; return .bool(true)
        case 102: i += 5; return .bool(false)
        case 110: i += 4; return .null
        default:
            let start = i
            while i < bytes.count && ![9,10,13,32,44,93,125].contains(bytes[i]) { i += 1 }
            guard let n = Double(String(decoding: bytes[start..<i], as: UTF8.self)), n.isFinite else { throw GLiNERError.invalidInput("Invalid JSON number") }
            return .number(n)
        }
    }
}
public struct InferenceRequest {
    public let text: String, schema: Schema, options: ExtractOptions, wordSplitter: WordSplitter, isLong: Bool
    public init(json: Data) throws {
        var parser = try OrderedParser(json); try self.init(node: parser.value())
    }
    public static func loadCases(url: URL) throws -> [InferenceRequest] {
        var parser = try OrderedParser(Data(contentsOf: url)); let root = try parser.value()
        guard let cases = root.array else { throw GLiNERError.invalidInput("Cases JSON must be an array") }
        return try cases.map { try InferenceRequest(node: $0) }
    }
    private init(node: OrderedJSON) throws {
        guard node.object != nil else { throw GLiNERError.invalidInput("Inference case must be an object") }
        isLong = node["long"]?.bool ?? false
        if isLong { text = (0..<40).map { "In report \($0), analyst Jane Smith of Goldman Sachs said Tesla shipped the Model Y from Berlin while Elon Musk visited Austin." }.joined(separator: " ") }
        else { guard let value = node["text"]?.string else { throw GLiNERError.invalidInput("Case requires a string text") }; text = value }
        wordSplitter = (node["char_split"]?.bool ?? false) ? .char : .whitespace
        let threshold = Float(node["threshold"]?.number ?? 0.5)
        options = ExtractOptions(threshold: threshold, includeSpans: node["include_spans"]?.bool ?? true, includeConfidence: node["include_confidence"]?.bool ?? true)
        var schema = Schema()
        if let structures = node["json"] {
            guard let pairs = structures.object else { throw GLiNERError.invalidInput("json must be an object") }
            let modeName = node["structure_mode"]?.string ?? ((node["legacy"]?.bool ?? false) ? "legacy" : "auto")
            guard let mode = StructureMode(rawValue: modeName) else { throw GLiNERError.invalidInput("Unknown structure_mode \(modeName)") }
            for (name, value) in pairs {
                guard let fields = value.array, fields.allSatisfy({ $0.string != nil }) else { throw GLiNERError.invalidInput("Fields for structure \(name) must be an array of strings") }
                var spec = StructureSpec.parse(name: name, fields: fields.compactMap(\.string)); spec.mode = mode
                if let anchors = node["anchors"]?.object { spec.anchor = anchors.last { $0.0 == name }?.1.string }
                if let anchor = node["anchor"]?.string { spec.anchor = anchor }
                schema.structures.append(spec)
            }
        }
        if let entities = node["entities"] {
            if let values = entities.array {
                guard values.allSatisfy({ $0.string != nil }) else { throw GLiNERError.invalidInput("entities must contain strings") }
                for value in values { schema = schema.addingEntity(EntitySpec(name: value.string!)) }
            } else if let pairs = entities.object {
                for (name, description) in pairs { guard let value = description.string else { throw GLiNERError.invalidInput("Entity description for \(name) must be a string") }; schema = schema.addingEntity(EntitySpec(name: name, description: value)) }
            } else { throw GLiNERError.invalidInput("entities must be an array or object") }
        }
        if let relations = node["relations"] {
            guard let array = relations.array, array.allSatisfy({ $0.string != nil }) else { throw GLiNERError.invalidInput("relations must be an array of strings") }
            schema.relations = array.map { RelationSpec(name: $0.string!) }
        }
        if let attributes = node["entity_attributes"] {
            guard let pairs = attributes.object else { throw GLiNERError.invalidInput("entity_attributes must be an object") }
            var groups: [NamedAttributeGroup] = []
            for (name, value) in pairs {
                guard value.object != nil else { throw GLiNERError.invalidInput("Attribute group \(name) must be an object") }
                let labels = try Self.strings(value["labels"], field: "attribute labels")
                let applies = value["applies_to"] == nil ? nil : try Self.strings(value["applies_to"], field: "applies_to")
                groups.append(NamedAttributeGroup(name: name, group: AttributeGroup(labels: labels, multiLabel: value["multi_label"]?.bool ?? false, threshold: Float(value["threshold"]?.number ?? 0.5), appliesTo: applies, qualifyLabels: value["qualify_labels"]?.bool ?? false)))
            }
            schema = schema.addingEntityAttributes(groups)
        }
        if let classifications = node["classifications"] {
            guard let tasks = classifications.array else { throw GLiNERError.invalidInput("classifications must be an array") }
            for task in tasks {
                guard let name = task["task"]?.string else { throw GLiNERError.invalidInput("classification needs a string task") }
                let labels = try Self.strings(task["labels"], field: "classification labels")
                let activationName = task["activation"]?.string ?? "auto"
                guard let activation = ClassActivation(rawValue: activationName) else { throw GLiNERError.invalidInput("Unknown classification activation \(activationName)") }
                var descriptions: [LabelDescription] = []
                if let desc = task["label_descriptions"] {
                    if let pairs = desc.object { for (label, d) in pairs { guard let text = d.string else { throw GLiNERError.invalidInput("Label description must be a string") }; descriptions.append(LabelDescription(label: label, description: text)) } }
                    else if let list = desc.array { for item in list { guard let label = item["label"]?.string, let text = item["description"]?.string else { throw GLiNERError.invalidInput("Invalid label description") }; descriptions.append(LabelDescription(label: label, description: text)) } }
                    else { throw GLiNERError.invalidInput("Invalid label_descriptions") }
                }
                var examples: [ClassificationExample] = []
                if let ex = task["examples"] { guard let list = ex.array else { throw GLiNERError.invalidInput("examples must be an array") }; for item in list { guard let input = item["input"]?.string, let output = item["output"]?.string else { throw GLiNERError.invalidInput("Invalid classification example") }; examples.append(ClassificationExample(input: input, output: output)) } }
                schema.classifications.append(ClassificationSpec(task: name, labels: labels, labelDescriptions: descriptions, multiLabel: task["multi_label"]?.bool ?? false, clsThreshold: Float(task["cls_threshold"]?.number ?? 0.5), activation: activation, prompt: task["prompt"]?.string, examples: examples))
            }
        }
        self.schema = schema
    }
    private static func strings(_ value: OrderedJSON?, field: String) throws -> [String] {
        guard let value = value else { return [] }; guard let items = value.array, items.allSatisfy({ $0.string != nil }) else { throw GLiNERError.invalidInput("\(field) must be an array of strings") }; return items.map { $0.string! }
    }
}
