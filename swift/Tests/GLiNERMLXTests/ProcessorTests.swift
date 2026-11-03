import XCTest
@testable import GLiNERMLX

final class ProcessorTests: XCTestCase {
    func testUnicodeLowercase() {
        XCTAssertEqual(rustLowercased("ΟΔΥΣΣΕΎΣ"), "οδυσσεύς")
        XCTAssertEqual(rustLowercased("ΟΣ\u{301}"), "ος\u{301}")
        XCTAssertEqual(rustLowercased("ΟΣΑ"), "οσα")
        XCTAssertEqual(rustLowercased("İ I É"), "i\u{307} i é")
    }
    func testUnicodeOffsets() {
        let words = WordSplitter.whitespace.split("héllo wörld", lower: true)
        XCTAssertEqual(words.map(\.text), ["héllo", "wörld"])
        XCTAssertEqual(words[1].byteStart, 7); XCTAssertEqual(words[1].byteEnd, 13)
        XCTAssertEqual(words[1].charStart, 6); XCTAssertEqual(words[1].charEnd, 11)
        XCTAssertEqual(WordSplitter.char.split("北京 GLiNER2").map(\.text), ["北", "京", "GLiNER2"])
    }
    func testOverlap() {
        let items = [ScoredSpan(start: 0, end: 3, score: 0.9), ScoredSpan(start: 0, end: 1, score: 0.6), ScoredSpan(start: 2, end: 3, score: 0.6)]
        XCTAssertEqual(resolveOverlaps(items, policy: .flat), Array(items.dropFirst()))
    }
    func testTokenizerViterbiAndUnknownFusion() throws {
        let json = #"{"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,"vocab":[["<unk>",0],["▁",-1],["a",-2],["b",-2],["ab",-1]]},"normalizer":null,"pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true},"added_tokens":[{"id":9,"content":"[P]","normalized":false}]}"#
        let tokenizer = try UnigramTokenizer(data: Data(json.utf8))
        XCTAssertEqual(try tokenizer.encode("ab"), [1,4])
        XCTAssertEqual(try tokenizer.encode("xyz"), [1,0])
        XCTAssertEqual(try tokenizer.encode("[P]"), [9])
    }
}
