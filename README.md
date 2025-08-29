# Parameter-Efficient Fine-Tuning (PEFT) Algorithm Comparison

Official code repository for ICCV 2025: Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri–Rao Product

Paper: [Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri–Rao Product](https://arxiv.org/pdf/2508.00230)

## Toy experiment quick start
Install dependencies and run the script

```bash
pip install torch matplotlib numpy open_clip_torch
python approximate_weight.py
```

## KRAdapter Linear layer

We are working on a hugginface peft implementation to allow users to easily experiment with KRAdapter. In the meantime, users can run our toy experiements on matrix appoximation or use the code below to use the Kathri-Rao product

```python
import torch
import torch.nn as nn
from typing import Any
import math

class KRAdapterLinearLayer(nn.Module):
    def __init__(self, original_layer, scaling=2): #No rank needed in KRAdapter as it is inferred from the base layer shape
        super().__init__()
        in_features, out_features = original_layer.in_features, original_layer.out_features
        min_shape, max_shape = min(in_features, out_features), max(out_features, in_features)
        self.s = (out_features, in_features)
        self.r = int(math.sqrt(min_shape))
        self.scaling = scaling
        self.original_layer = original_layer
        self.merged = False

        self.d = 0
        while self.r * (self.r + self.d) < max_shape:
            self.d += 1
        
        self.min_shape = min_shape
        self.weight = nn.Parameter(torch.zeros(min_shape, self.r))
        self.v_weight = nn.Parameter(torch.zeros(min_shape, self.r + self.d))

        nn.init.kaiming_uniform_(self.v_weight, a=5)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)

        update = self.get_update()
        return x @ update * self.scaling + self.original_layer(x)

    def get_update(self):
        update = (self.weight.unsqueeze(1) * self.v_weight.unsqueeze(-1)).view(self.weight.shape[0], -1)
        if torch.argmin(torch.tensor(self.s)) != torch.argmin(torch.tensor(update.shape)):
            update = update.T
        update = update[:self.s[0], :self.s[1]]
        return update.T

    @torch.no_grad
    def merge(self):
        if not self.merged:
            self.original_layer.weight.data += self.get_update().data * self.scaling
            self.merged = True
    
    @torch.no_grad
    def unmerge(self):
        if self.merged:
            self.original_layer.weigh.data -= self.get_update().data * self.scaling
            self.merged = False


if __name__ == "__main__":
    size = (1024, 768)
    original_layer = nn.Linear(*size)
    for p in original_layer.parameters():
        p.requires_grad = False

    kra_layer = KRAdapterLinearLayer(original_layer)

    x = torch.randn(64, 4, size[0])
    out = kra_layer(x)
```

## Overview

Parameter-efficient fine-tuning methods have become crucial for adapting large pre-trained models while minimizing computational overhead. This experimental framework compares several state-of-the-art PEFT algorithms:

- **LoRA (Low-Rank Adaptation)**: Decomposes weight updates into low-rank matrices
- **SinLoRA**: LoRA variant with sinusoidal activation functions
- **KRAdapter**: Kronecker-factorized rank adapter with structured parameterization
- **RandLoRA**: LoRA with fixed random basis matrices and learnable coefficients
- **Kronecker Adapter**: Uses Kronecker product structure for parameter efficiency

## Scientific Background

### Problem Formulation

Given a target weight matrix **W** ∈ ℝ^(d×k), PEFT methods aim to learn a parameter-efficient update **ΔW** such that:

**W_adapted** = **W** + **ΔW**

where **ΔW** is represented using significantly fewer parameters than the full matrix **W**.

### Algorithms

#### LoRA (Low-Rank Adaptation)
- **Parameterization**: ΔW = **B** @ **A**, where **B** ∈ ℝ^(d×r), **A** ∈ ℝ^(r×k)
- **Parameters**: r(d + k) where r << min(d,k)
- **Key insight**: Many weight updates have low intrinsic rank

#### SinLoRA
- **Parameterization**: ΔW = sin(ω × **B** @ **A**) × α
- **Parameters**: r(d + k) + 2 (frequency ω and scaling α)

#### RandLoRA
- **Parameterization**: Fixed random basis with learnable combination weights
- **Parameters**: Reduced through pre-computed random projections

#### Kronecker Adapter
- **Parameterization**: ΔW = **A** ⊗ **B** (Kronecker product)
- **Parameters**: r² + (d/r)(k/r) where r is the Kronecker rank

#### KRAdapter (Ours)
- **Parameterization**: Kathri-Rao factorization with adaptive dimensions
- **Parameters**: Optimized based on matrix dimensions and rank constraints

### Evaluation Metrics

1. **Nuclear Reconstruction Error**: RMS error between singular values of target and approximated matrices
2. **Effective Rank**: Entropy-based measure of rank: exp(-Σ pᵢ log pᵢ) where pᵢ are normalized singular values
3. **Parameter Efficiency**: Number of trainable parameters per algorithm

## Usage

### Configuration Options

Edit the configuration parameters in the `main()` function:

```python
# Matrix dimensions (reduce for faster execution)
matrix_size = (1024, 768)  # or (128, 128) for quick testing

# Training parameters
training_iterations = 300
learning_rate = 1e-1
```

### Custom Matrix Types

The framework supports various matrix types for evaluation:

- **Random noise**: Standard Gaussian matrices
- **Sparse noise**: 90% sparse random matrices  
- **PCA whitened**: Decorrelated random matrices
- **Low rank**: Matrices with truncated singular value spectrum
- **CLIP weights**: Attention weights from vision-language models
- **Frequency patterns**: Sinusoidal matrices with controlled frequency content

## Output and Interpretation

### Generated Files

1. **`experimental_results.json`**: Complete numerical results including singular values, errors, and statistics
2. **`peft_comparison.png`**: Comparative visualization of algorithm performance
3. **Console output**: Real-time training progress and summary statistics

## Experimental Design

### Matrix Test Suite

The evaluation uses eight distinct matrix types to assess algorithm robustness:

1. **Random Noise**: Tests general approximation capability
2. **Sparse Noise**: Evaluates performance on sparse structures  
3. **PCA Whitened**: Assesses handling of decorrelated data
4. **Low Rank**: Validates low-rank approximation quality
5. **CLIP Language**: Real-world transformer attention weights
6. **CLIP Vision**: Vision transformer attention patterns
7. **High Frequency**: High-frequency sinusoidal patterns (1-10 kHz)
8. **Low Frequency**: Low-frequency sinusoidal patterns (1-100 Hz)

### Parameter Budget

All algorithms are constrained to use at least the same number of parameters as KRAdapters:
```
Budget = 2 × min(d,k) × √max(d,k)
```

This ensures fair comparison across different parameterization strategies.

## Research Applications

This framework is particularly useful for:

- **Model Compression**: Identifying optimal PEFT methods for specific architectures
- **Transfer Learning**: Evaluating adaptation strategies for new domains  
- **Spectral Analysis**: Understanding how algorithms handle different frequency characteristics
- **Architecture Design**: Informing parameter-efficient layer designs

## Implementation Notes

### Reproducibility

- Fixed random seeds ensure consistent results across runs
- All algorithms use identical initialization strategies (Kaiming uniform)
- Training procedures are standardized across methods

### Computational Considerations

- **Matrix Size**: Use (128, 128) for rapid prototyping, (1024, 768) or larger for full evaluation
- **Device Support**: CPU and GPU compatible (set `DEVICE` variable)

### CLIP Integration

The framework includes CLIP model weights for realistic evaluation:
- Uses OpenAI's ViT-L/14 architecture
- Extracts attention projection weights from language and vision encoders

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce matrix size or use CPU device
2. **CLIP Loading Failures**: Install `open-clip-torch` or disable CLIP matrices
3. **Convergence Issues**: Adjust learning rate or increase iterations
4. **Import Errors**: Ensure all dependencies are installed with correct versions

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{2025_KRAdapters_ICCV,
  title={Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri--Rao Product},
  author={Albert, Paul and Zhang, Frederic Z and Saratchandran, Hemanth and Hengel, Anton van den and Abbasnejad, Ehsan},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```

This code is provided for research purposes. Please refer to individual dependencies for their respective licenses.
