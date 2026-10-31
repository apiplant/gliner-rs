import Foundation

public enum WordSplitter: String, Codable, CaseIterable { case whitespace, char }
public enum OverlapPolicy: String, Codable, CaseIterable { case flat, nested, allow, longest }
public enum ClassActivation: String, Codable, CaseIterable { case auto, sigmoid, softmax }
public enum Cardinality: String, Codable, CaseIterable { case optionalOne = "optional_one", requiredOne = "required_one", zeroOrMore = "zero_or_more", oneOrMore = "one_or_more" }
public enum FieldDtype: String, Codable, CaseIterable { case list, str }
public enum EntityDtype: String, Codable, CaseIterable { case list, str }
public enum StructureMode: String, Codable, CaseIterable { case auto, legacy, latent, anchorless }
public enum ValidatorMode: String, Codable, CaseIterable { case full, partial }

public struct RegexValidator: Codable, Equatable {
    public var pattern: String
    public var mode: ValidatorMode
    public var exclude: Bool
    public var caseInsensitive: Bool

    public init(pattern: String, mode: ValidatorMode = .full, exclude: Bool = false, caseInsensitive: Bool = true) {
        self.pattern = pattern; self.mode = mode; self.exclude = exclude; self.caseInsensitive = caseInsensitive
    }

    public func validate(_ text: String) -> Bool {
        let expression = mode == .full ? "^(?:\(pattern))$" : pattern
        let options: NSRegularExpression.Options = caseInsensitive ? [.caseInsensitive] : []
        let matched: Bool
        if let regex = try? NSRegularExpression(pattern: expression, options: options) {
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            matched = regex.firstMatch(in: text, range: range) != nil
        } else {
            matched = false
        }
        return exclude ? !matched : matched
    }
}

public struct EntitySpec: Codable, Equatable {
    public var name: String
    public var description: String?
    public var threshold: Float?
    public var dtype: EntityDtype
    public var validators: [RegexValidator]
    /// Attribute labels are query slots, but do not appear in the entity output.
    public var isAttribute: Bool

    public init(name: String, description: String? = nil, threshold: Float? = nil,
                dtype: EntityDtype = .list, validators: [RegexValidator] = [], isAttribute: Bool = false) {
        self.name = name; self.description = description; self.threshold = threshold
        self.dtype = dtype; self.validators = validators; self.isAttribute = isAttribute
    }
}

public struct AttributeGroup: Codable, Equatable {
    public var labels: [String]
    public var multiLabel: Bool
    public var threshold: Float
    public var appliesTo: [String]?
    public var qualifyLabels: Bool

    public init(labels: [String], multiLabel: Bool = false, threshold: Float = 0.5,
                appliesTo: [String]? = nil, qualifyLabels: Bool = false) {
        self.labels = labels; self.multiLabel = multiLabel; self.threshold = threshold
        self.appliesTo = appliesTo; self.qualifyLabels = qualifyLabels
    }
}

public struct NamedAttributeGroup: Codable, Equatable {
    public var name: String
    public var group: AttributeGroup
    public init(name: String, group: AttributeGroup) { self.name = name; self.group = group }
}

public struct RelationSpec: Codable, Equatable {
    public var name: String
    public var description: String?
    public var threshold: Float?
    public init(name: String, description: String? = nil, threshold: Float? = nil) {
        self.name = name; self.description = description; self.threshold = threshold
    }
}

public struct LabelDescription: Codable, Equatable {
    public var label: String
    public var description: String
    public init(label: String, description: String) { self.label = label; self.description = description }
}

public struct ClassificationExample: Codable, Equatable {
    public var input: String
    public var output: String
    public init(input: String, output: String) { self.input = input; self.output = output }
}

public struct ClassificationSpec: Codable, Equatable {
    public var task: String
    public var labels: [String]
    public var labelDescriptions: [LabelDescription]
    public var multiLabel: Bool
    public var clsThreshold: Float
    public var activation: ClassActivation
    public var prompt: String?
    public var examples: [ClassificationExample]

    public init(task: String, labels: [String], labelDescriptions: [LabelDescription] = [],
                multiLabel: Bool = false, clsThreshold: Float = 0.5, activation: ClassActivation = .auto,
                prompt: String? = nil, examples: [ClassificationExample] = []) {
        self.task = task; self.labels = labels; self.labelDescriptions = labelDescriptions
        self.multiLabel = multiLabel; self.clsThreshold = clsThreshold; self.activation = activation
        self.prompt = prompt; self.examples = examples
    }
}

public struct FieldSpec: Codable, Equatable {
    public var name: String
    public var dtype: FieldDtype
    public var choices: [String]?
    public var description: String?
    public var threshold: Float?
    public var cardinality: Cardinality?
    public var exclusive: Bool?
    public var validators: [RegexValidator]

    public init(name: String, dtype: FieldDtype = .list, choices: [String]? = nil,
                description: String? = nil, threshold: Float? = nil, cardinality: Cardinality? = nil,
                exclusive: Bool? = nil, validators: [RegexValidator] = []) {
        self.name = name; self.dtype = dtype; self.choices = choices; self.description = description
        self.threshold = threshold; self.cardinality = cardinality; self.exclusive = exclusive
        self.validators = validators
    }

    /// Parses `name[::str|list][::[a|b|c]][::description]`; choices imply `str` unless dtype is explicit.
    public static func parse(_ spec: String) -> FieldSpec {
        let parts = spec.components(separatedBy: "::")
        var field = FieldSpec(name: parts.first ?? "")
        var dtypeExplicit = false
        for part in parts.dropFirst() {
            switch part {
            case "str": field.dtype = .str; dtypeExplicit = true
            case "list": field.dtype = .list; dtypeExplicit = true
            default:
                if part.hasPrefix("["), part.hasSuffix("]"), part.count >= 2 {
                    field.choices = String(part.dropFirst().dropLast()).components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
                    if !dtypeExplicit { field.dtype = .str }
                } else { field.description = part }
            }
        }
        field.cardinality = field.dtype == .str ? .requiredOne : .zeroOrMore
        field.exclusive = true
        return field
    }
}

public struct StructureSpec: Codable, Equatable {
    public var name: String
    public var fields: [FieldSpec]
    public var mode: StructureMode
    public var anchor: String?

    public init(name: String, fields: [FieldSpec], mode: StructureMode = .auto, anchor: String? = nil) {
        self.name = name; self.fields = fields; self.mode = mode; self.anchor = anchor
    }

    public static func parse(name: String, fields: [String]) -> StructureSpec {
        StructureSpec(name: name, fields: fields.map(FieldSpec.parse))
    }
}

public struct Schema: Codable, Equatable {
    public var structures: [StructureSpec]
    public var entities: [EntitySpec]
    public var relations: [RelationSpec]
    public var classifications: [ClassificationSpec]
    public var entityAttributes: [NamedAttributeGroup]

    public init() {
        structures = []; entities = []; relations = []; classifications = []; entityAttributes = []
    }

    public init(entities names: [String]) {
        self.init()
        for name in names { appendEntity(EntitySpec(name: name)) }
    }

    public func addingEntity(_ spec: EntitySpec) -> Schema {
        var copy = self; copy.appendEntity(spec); return copy
    }

    public func addingRelation(_ spec: RelationSpec) -> Schema { var copy = self; copy.relations.append(spec); return copy }
    public func addingClassification(_ spec: ClassificationSpec) -> Schema { var copy = self; copy.classifications.append(spec); return copy }
    public func addingStructure(_ spec: StructureSpec) -> Schema { var copy = self; copy.structures.append(spec); return copy }

    /// Registers groups and adds sorted, deduplicated phantom entity queries for their labels.
    public func addingEntityAttributes(_ groups: [NamedAttributeGroup]) -> Schema {
        var copy = self
        copy.entityAttributes.append(contentsOf: groups)
        copy.ensureAttributeEntities()
        return copy
    }

    /// Injects the phantom query entities implied by `entityAttributes`, in prompt-stable sorted order.
    public mutating func ensureAttributeEntities() {
        var promptLabels = Set<String>()
        for named in entityAttributes {
            for label in named.group.labels {
                promptLabels.insert(named.group.qualifyLabels ? "\(named.name): \(label)" : label)
            }
        }
        for label in promptLabels.sorted() where !entities.contains(where: { $0.name == label }) {
            entities.append(EntitySpec(name: label, isAttribute: true))
        }
    }

    private mutating func appendEntity(_ spec: EntitySpec) {
        guard let index = entities.firstIndex(where: { $0.name == spec.name }) else { entities.append(spec); return }
        entities[index].dtype = spec.dtype
        entities[index].threshold = spec.threshold
        if spec.description != nil { entities[index].description = spec.description }
        if !spec.validators.isEmpty { entities[index].validators = spec.validators }
    }
}
