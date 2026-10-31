import Foundation
import MLX

/// Flat checkpoint access. Prefixes follow the original safetensors names.
public struct WeightStore {
    private let values: [String: MLXArray]
    private let prefix: String
    private let dtype: DType?

    public init(_ values: [String: MLXArray]) {
        self.values = values; self.prefix = ""; self.dtype = nil
    }
    private init(values: [String: MLXArray], prefix: String, dtype: DType?) {
        self.values = values; self.prefix = prefix; self.dtype = dtype
    }
    public func prefixed(_ name: String, dtype: DType? = nil) -> WeightStore {
        WeightStore(values: values, prefix: fullName(name), dtype: dtype ?? self.dtype)
    }
    private func fullName(_ name: String) -> String {
        prefix.isEmpty ? name : (name.isEmpty ? prefix : prefix + "." + name)
    }
    public func get(_ name: String, shape: [Int]? = nil) throws -> MLXArray {
        let key = fullName(name)
        guard let value = values[key] else { throw GLiNERError.missingWeight(key) }
        if let shape, value.shape != shape {
            throw GLiNERError.invalidWeights("\(key): expected \(shape), got \(value.shape)")
        }
        return dtype.map { value.asType($0) } ?? value
    }
    public func validateLinear(_ name: String, input: Int, output: Int) throws {
        _ = try get(name + ".weight", shape: [output, input])
        _ = try get(name + ".bias", shape: [output])
    }
    public func validateLayerNorm(_ name: String, size: Int) throws {
        _ = try get(name + ".weight", shape: [size]); _ = try get(name + ".bias", shape: [size])
    }
    public func linear(_ x: MLXArray, _ name: String) throws -> MLXArray {
        let w = try get(name + ".weight")
        guard w.ndim == 2, x.dim(-1) == w.dim(1) else {
            throw GLiNERError.invalidWeights("\(fullName(name)): incompatible linear input \(x.shape), weight \(w.shape)")
        }
        let b = try get(name + ".bias", shape: [w.dim(0)])
        return matmul(x, w.transposed()) + b
    }
    public func layerNorm(_ x: MLXArray, _ name: String, eps: Float = 1e-5) throws -> MLXArray {
        let w = try get(name + ".weight", shape: [x.dim(-1)])
        let b = try get(name + ".bias", shape: [x.dim(-1)])
        return MLXFast.layerNorm(x, weight: w, bias: b, eps: eps)
    }
    public func embedding(_ ids: MLXArray, _ name: String) throws -> MLXArray {
        let w = try get(name + ".weight")
        guard w.ndim == 2 else { throw GLiNERError.invalidWeights("\(name): embedding must be rank 2") }
        return w.take(ids, axis: 0)
    }
}

struct TensorLinear {
    let weight: MLXArray
    let bias: MLXArray
    init(_ weights: WeightStore, _ prefix: String, _ input: Int, _ output: Int) throws {
        weight = try weights.get(prefix + ".weight", shape: [output, input])
        bias = try weights.get(prefix + ".bias", shape: [output])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { matmul(x, weight.transposed()) + bias }
}

struct TensorLayerNorm {
    let weight: MLXArray
    let bias: MLXArray
    let eps: Float
    init(_ weights: WeightStore, _ prefix: String, _ size: Int, eps: Float = 1e-5) throws {
        weight = try weights.get(prefix + ".weight", shape: [size])
        bias = try weights.get(prefix + ".bias", shape: [size]); self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight, bias: bias, eps: eps)
    }
}

func geluErf(_ x: MLXArray) -> MLXArray {
    // Candle's exact GELU uses erf(x * FRAC_1_SQRT_2), then (erf + 1) * 0.5 * x.
    // Keep half-precision activation intermediates in float32 before the final cast.
    let value = x.asType(.float32)
    let result = ((erf(value * Float(1 / sqrt(2.0))) + 1) * 0.5) * value
    return result.asType(x.dtype)
}
func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
func tensorRows(_ x: MLXArray) -> [[Float]] {
    let values = x.asType(.float32).asArray(Float.self)
    let width = x.dim(1)
    guard width > 0 else { return Array(repeating: [], count: x.dim(0)) }
    return stride(from: 0, to: values.count, by: width).map { Array(values[$0..<$0 + width]) }
}
func indexArray(_ positions: [Int]) -> MLXArray { MLXArray(positions.map(Int32.init)) }
func tensorSlice(_ x: MLXArray, axis: Int, start: Int, count: Int) -> MLXArray {
    x.take(indexArray(Array(start..<start + count)), axis: axis)
}
