import Foundation

@main
struct TokenizeApple {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count == 2 else {
            fputs("usage: tokenize_apple MODEL_DIR CASES_JSON\n", stderr)
            throw GLiNERError.invalidInput("Expected MODEL_DIR and CASES_JSON")
        }
        let modelDirectory = URL(fileURLWithPath: arguments[0], isDirectory: true)
        let casesURL = URL(fileURLWithPath: arguments[1])
        let tokenizer = try UnigramTokenizer(url: modelDirectory.appendingPathComponent("tokenizer.json"))
        let cases = try InferenceRequest.loadCases(url: casesURL)
        var processor = Processor(tokenizer: tokenizer)

        for request in cases {
            processor.wordSplitter = request.wordSplitter
            let prepared = try processor.prepare(text: request.text, schema: request.schema)
            let markers = prepared.groups.flatMap(\.markerPositions)
            let words: [[String: Any]] = prepared.words.map { word in
                [
                    "text": word.text,
                    "char_start": word.charStart,
                    "char_end": word.charEnd,
                    "byte_start": word.byteStart,
                    "byte_end": word.byteEnd,
                ]
            }
            let row: [String: Any] = [
                "text": prepared.text,
                "input_ids": prepared.inputIds,
                "word_positions": prepared.wordPositions,
                "marker_positions": markers,
                "prefix_tokens": prepared.prefixTokens,
                "words": words,
            ]
            let data = try JSONSerialization.data(withJSONObject: row, options: [.fragmentsAllowed])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data([0x0A]))
        }
    }
}
