import XCTest
@testable import GLiNERMLX
final class ChunkingTests: XCTestCase {
    func testChunksAndValidation() throws {
        let chunks = try splitTextIntoChunks(text: "é uno due tre", options: .init(chunkSize: 2, chunkOverlap: 1))
        XCTAssertEqual(chunks, [TextChunk(text: "é uno", startChar: 0), TextChunk(text: "uno due", startChar: 2), TextChunk(text: "due tre", startChar: 6)])
        XCTAssertThrowsError(try splitTextIntoChunks(text: "x", options: .init(chunkSize: 2, chunkOverlap: 2)))
    }
    func testRemapAndMerge() {
        let mention: JSONValue = .object(["text": .string("wrong"), "start": .number(0), "end": .number(3), "confidence": .number(0.8)])
        let mapped = remapChunkSpans(mention, startChar: 2, originalText: "é uno due")
        XCTAssertEqual(mapped["text"], .string("uno")); XCTAssertEqual(mapped["start"], .number(2))
        let lower: JSONValue = .object(["text": .string("uno"), "start": .number(2), "end": .number(5), "confidence": .number(0.3)])
        let results: [JSONValue] = [.object(["entities": .object(["word": .array([lower])])]), .object(["entities": .object(["word": .array([mapped])])])]
        let merged = mergeChunkResults(results)
        XCTAssertEqual(merged["entities"]?["word"], .array([mapped]))
        XCTAssertEqual(stripSpanMetadata(merged, includeConfidence: false, includeSpans: false)["entities"]?["word"], .array([.string("uno")]))
    }
    func testRelationAndClassificationMerge() {
        let h: JSONValue = .object(["text": .string("A"),"start": .number(0),"end": .number(1),"confidence": .number(0.7)])
        let t: JSONValue = .object(["text": .string("B"),"start": .number(2),"end": .number(3),"confidence": .number(0.7)])
        let relation: JSONValue = .object(["head": h, "tail": t])
        let chunk: JSONValue = .object(["relation_extraction": .object(["related": .array([relation])]), "sentiment": .object(["label": .string("positive"),"confidence": .number(0.8)])])
        let merged = mergeChunkResults([chunk,chunk])
        XCTAssertEqual(merged["relation_extraction"]?["related"], .array([relation]))
        XCTAssertEqual(stripSpanMetadata(merged, includeConfidence: false, includeSpans: false)["sentiment"], .string("positive"))
    }
}
