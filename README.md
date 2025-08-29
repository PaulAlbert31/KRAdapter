# Parameter-Efficient Fine-Tuning (PEFT) Algorithm Comparison

This repository contains code for comparing various Parameter-Efficient Fine-Tuning (PEFT) algorithms for weight matrix approximation. The study evaluates how different PEFT methods reconstruct various types of weight matrices, providing insights into their effectiveness across different spectral characteristics.

## Quick start
Install dependencies and run the script

```bash
pip install torch matplotlib numpy open_clip_torch
python approximate_weight.py
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
