// MPNNLayers.swift — message-passing layers, mirroring port/mlx_packer.py
// (enc_layer / dec_layer / dec_layer_j). Shared by the design + packer models.
import MLX

func encLayer(_ w: Weights, _ p: String, _ hV0: MLXArray, _ hE0: MLXArray,
              _ eIdx: MLXArray, _ mask: MLXArray, _ maskAttend: MLXArray,
              scale: Float = 30.0) -> (MLXArray, MLXArray) {
    var hV = hV0, hE = hE0
    var hEV = catNeighborsNodes(hV, hE, eIdx)
    var hVexp = broadcast(hV.expandedDimensions(axis: -2),
                          to: Array(hEV.shape.dropLast()) + [lastDim(hV)])
    hEV = concatenated([hVexp, hEV], axis: -1)
    var hm = linear(w, p + ".W3", geluExact(linear(w, p + ".W2", geluExact(linear(w, p + ".W1", hEV)))))
    hm = maskAttend.expandedDimensions(axis: -1) * hm
    let dh = sum(hm, axis: -2) / scale
    hV = layerNorm(w, p + ".norm1", hV + dh)
    hV = layerNorm(w, p + ".norm2", hV + ffn(w, p + ".dense", hV))
    hV = mask.expandedDimensions(axis: -1) * hV

    hEV = catNeighborsNodes(hV, hE, eIdx)
    hVexp = broadcast(hV.expandedDimensions(axis: -2),
                      to: Array(hEV.shape.dropLast()) + [lastDim(hV)])
    hEV = concatenated([hVexp, hEV], axis: -1)
    hm = linear(w, p + ".W13", geluExact(linear(w, p + ".W12", geluExact(linear(w, p + ".W11", hEV)))))
    hE = layerNorm(w, p + ".norm3", hE + hm)
    return (hV, hE)
}

// dec_layer / dec_layer_j share the same body (h_V expands over the -2 axis to match h_E).
func decLayer(_ w: Weights, _ p: String, _ hV0: MLXArray, _ hE: MLXArray,
              maskV: MLXArray? = nil, maskAttend: MLXArray? = nil,
              scale: Float = 30.0) -> MLXArray {
    var hV = hV0
    let hVexp = broadcast(hV.expandedDimensions(axis: -2),
                          to: Array(hE.shape.dropLast()) + [lastDim(hV)])
    let hEV = concatenated([hVexp, hE], axis: -1)
    var hm = linear(w, p + ".W3", geluExact(linear(w, p + ".W2", geluExact(linear(w, p + ".W1", hEV)))))
    if let ma = maskAttend { hm = ma.expandedDimensions(axis: -1) * hm }
    let dh = sum(hm, axis: -2) / scale
    hV = layerNorm(w, p + ".norm1", hV + dh)
    hV = layerNorm(w, p + ".norm2", hV + ffn(w, p + ".dense", hV))
    if let mv = maskV { hV = mv.expandedDimensions(axis: -1) * hV }
    return hV
}

@inline(__always)
func decLayerJ(_ w: Weights, _ p: String, _ hV: MLXArray, _ hE: MLXArray,
               maskV: MLXArray? = nil, maskAttend: MLXArray? = nil) -> MLXArray {
    decLayer(w, p, hV, hE, maskV: maskV, maskAttend: maskAttend)
}
