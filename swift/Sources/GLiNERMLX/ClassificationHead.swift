import MLX

/// Eval-time label-marker classification MLP (Linear -> ReLU -> Linear).
public struct ClassificationHead {
    private let hidden: TensorLinear
    private let output: TensorLinear
    private let hiddenSize: Int
    public init(weights: WeightStore, hidden: Int, prefix: String = "classifier", span: Bool = false) throws {
        let w = weights.prefixed(prefix, dtype: .float32)
        self.hidden = try TensorLinear(w, "0", hidden, 2 * hidden)
        output = try TensorLinear(w, span ? "2" : "3", 2 * hidden, 1)
        hiddenSize = hidden
    }
    public func forward(_ states: MLXArray) throws -> [Float] {
        guard states.ndim == 2, states.dim(1) == hiddenSize else {
            throw GLiNERError.invalidInput("Classification states must have shape [K,\(hiddenSize)]")
        }
        if states.dim(0) == 0 { return [] }
        return output(maximum(hidden(states.asType(.float32)), 0)).asArray(Float.self)
    }
}
public typealias Classifier = ClassificationHead
