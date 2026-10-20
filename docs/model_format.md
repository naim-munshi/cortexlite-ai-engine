# CLMODEL1 model format

CortexLite uses a minimal whitespace-delimited text format to keep the C parser independent of external libraries.

```
CLMODEL1
<input_dim> <hidden_dim> <output_dim>
<h1 x input_dim floats in row-major order>
<hidden_dim bias floats>
<output_dim x hidden_dim floats in row-major order>
<output_dim bias floats>
```

The loader validates the magic string, dimension bounds, allocation success, finite numeric values and unexpected trailing content. Floating-point values are exported with nine significant decimal digits from PyTorch to preserve 32-bit float round-trip fidelity. Training data and label semantics are intentionally not encoded in the model file.

The network is `softmax(W2 * relu(W1 * x + b1) + b2)`. The ReLU hidden activations are exported as the embeddings used by the C++ search layer. These toy embeddings are not semantic LLM embeddings.
