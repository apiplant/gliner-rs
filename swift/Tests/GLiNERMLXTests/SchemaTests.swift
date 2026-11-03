import Foundation
import XCTest
@testable import GLiNERMLX

final class SchemaTests: XCTestCase {
    func testCompactFieldParsingMatchesRustDefaults() {
        let choiceField = FieldSpec.parse("status::[shipped| pending ]::Order status")
        XCTAssertEqual(choiceField.name, "status")
        XCTAssertEqual(choiceField.dtype, .str)
        XCTAssertEqual(choiceField.choices, ["shipped", "pending"])
        XCTAssertEqual(choiceField.description, "Order status")
        XCTAssertEqual(choiceField.cardinality, .requiredOne)
        XCTAssertEqual(choiceField.exclusive, true)

        let listField = FieldSpec.parse("tags::list::[a|b]")
        XCTAssertEqual(listField.dtype, .list)
        XCTAssertEqual(listField.cardinality, .zeroOrMore)
        XCTAssertEqual(FieldSpec.parse("name").dtype, .list)
    }

    func testRegexValidatorFullPartialAndExclusion() {
        XCTAssertTrue(RegexValidator(pattern: "acme-[0-9]+").validate("ACME-42"))
        XCTAssertFalse(RegexValidator(pattern: "[0-9]+").validate("id-42"))
        XCTAssertTrue(RegexValidator(pattern: "[0-9]+", mode: .partial).validate("id-42"))
        XCTAssertTrue(RegexValidator(pattern: "[0-9]+", exclude: true).validate("acme"))
        XCTAssertFalse(RegexValidator(pattern: "[", mode: .partial).validate("anything"))
    }

    func testAttributePhantomsAreSortedDeduplicatedAndQualified() {
        let schema = Schema(entities: ["Company"])
            .addingEntityAttributes([
                NamedAttributeGroup(name: "status", group: AttributeGroup(labels: ["active", "inactive"], qualifyLabels: true)),
                NamedAttributeGroup(name: "kind", group: AttributeGroup(labels: ["active", "person"]))
            ])

        XCTAssertEqual(schema.entities.map(\.name), ["Company", "active", "person", "status: active", "status: inactive"])
        XCTAssertTrue(schema.entities.filter(\.isAttribute).allSatisfy { $0.name != "Company" })
        XCTAssertEqual(schema.entityAttributes.map(\.name), ["status", "kind"])
    }

    func testSchemaAdditionsAndClassificationCodable() throws {
        let spec = ClassificationSpec(task: "sentiment", labels: ["positive", "negative"],
                                      labelDescriptions: [LabelDescription(label: "positive", description: "good")],
                                      multiLabel: true, clsThreshold: 0.4, activation: .sigmoid,
                                      prompt: "How does it feel?", examples: [ClassificationExample(input: "great", output: "positive")])
        let schema = Schema()
            .addingClassification(spec)
            .addingStructure(.parse(name: "person", fields: ["name::str", "role::[lead|member]"]))
            .addingRelation(RelationSpec(name: "works_for"))
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(Schema.self, from: data)
        XCTAssertEqual(decoded, schema)
        XCTAssertEqual(decoded.classifications[0].examples[0].output, "positive")
    }
}
