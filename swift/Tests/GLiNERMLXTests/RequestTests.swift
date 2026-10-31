import Foundation
import XCTest
@testable import GLiNERMLX
final class RequestTests: XCTestCase {
    func testPreservesObjectOrderAndUnicode() throws {
        let json = #"{"text":"Zoë 🧑🏽‍🚀","entities":{"zeta":"ultimo","alpha":"primo"},"json":{"z":["x::str"],"a":["y::list"]},"classifications":[{"task":"sentimento","labels":["a","b"],"label_descriptions":{"b":"secondo","a":"primo"}}]}"#
        let request = try InferenceRequest(json: Data(json.utf8))
        XCTAssertEqual(request.text, "Zoë 🧑🏽‍🚀")
        XCTAssertEqual(request.schema.entities.map(\.name), ["zeta","alpha"])
        XCTAssertEqual(request.schema.structures.map(\.name), ["z","a"])
        XCTAssertEqual(request.schema.classifications[0].labelDescriptions.map(\.label), ["b","a"])
        XCTAssertTrue(request.options.includeSpans); XCTAssertTrue(request.options.includeConfidence)
    }
    func testInvalidJSON() {
        XCTAssertThrowsError(try InferenceRequest(json: Data("{\"text\": ".utf8)))
        XCTAssertThrowsError(try InferenceRequest(json: Data("{\"text\": 3}".utf8)))
    }
}
