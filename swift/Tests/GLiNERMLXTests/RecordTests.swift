import XCTest
@testable import GLiNERMLX
final class RecordTests: XCTestCase {
    func testHungarian() { XCTAssertEqual(linearSumAssignment([[4,1,3],[2,0,5]]), [1,0]) }
    func testChoices() {
        let offsets = findChoiceMentions(text: "Books and e-books, BOOKS; notebooks", choice: "books")
        XCTAssertEqual(offsets.map { [$0.0,$0.1] }, [[0,5],[12,17],[19,24]])
        let (found, records) = recordLocalChoiceMentions(text: "Amazon sells books. Walmart sells groceries and books.", choices: ["books", "groceries"], anchors: [(0,6),(20,27)])
        XCTAssertTrue(found); XCTAssertEqual(records[0]?.map(\.0), ["books"]); XCTAssertEqual(records[1]?.map(\.0), ["groceries", "books"])
    }
    func testRequiredExclusiveAssignment() {
        let fields = [RecordField(query: 0, cardinality: .requiredOne, isAnchor: false, exclusive: true)]
        let records = decodeRecordGroup(mode: .anchorless, fields: fields, objectLogits: [4,3], spans: nil, assign: [[[0,4,3],[0,4,0]]], threshold: 0.5, temperature: 1)
        XCTAssertEqual(records.count, 2); XCTAssertEqual(records[0].fields[0]?.first?.0, 1); XCTAssertEqual(records[1].fields[0]?.first?.0, 0)
    }
}
