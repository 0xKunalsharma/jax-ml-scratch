# JAX-ML-Scratch

Mathematical implementations of machine learning models built from scratch using JAX, with an emphasis on understanding the underlying mathematics rather than relying on high-level model abstractions.

### Progress Log

**[16 October 2026]** Implemented `mlp_mnist.py`, a multilayer perceptron trained on the MNIST handwritten digit dataset. Includes manual parameter initialization, forward propagation, cross-entropy loss, gradient-based optimization, and evaluation on held-out test data.

**[21 October 2026]** Transformer_fineweb.ipynb — single-layer, single-head, post-LN transformer trained on 1M FineWeb tokens (GPT-2 tokenizer) with a hand-rolled Muon optimiser (Newton-Schulz orthogonalisation). Final train loss ~4.9 after 100k steps. Greedy samples included in the notebook.
