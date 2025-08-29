"""
Parameter-Efficient Fine-Tuning (PEFT) Algorithm Comparison for Weight Matrix Approximation

This module implements and compares various PEFT algorithms including LoRA, SinLoRA, 
KRAdapter, RandLoRA, and Kronecker Adapters for approximating different types of weight matrices.

Author: Research Team
Date: 2024
"""

import torch
import math
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import open_clip
import json
from typing import Tuple, Optional, Dict, Any, List
import warnings
import os

# Import configuration
from config import *

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class BaseModule(nn.Module):
    """
    Base class for all parameter-efficient adaptation modules.
    
    This class provides the common interface for different PEFT methods,
    allowing them to update a base weight matrix with learned parameters.
    """
    
    def __init__(self, size: Tuple[int, int]):
        """
        Initialize the base module.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
        """
        super(BaseModule, self).__init__()
        self.size = size

    def update(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update the weight matrix with the adaptation.
        
        Args:
            w: Base weight matrix (optional)
            
        Returns:
            Updated weight matrix
        """
        w_update = self.get_update(w)
        if hasattr(self, 'B1') and hasattr(self, 'A1'):
            w_update = w_update + self.B1 @ self.A1
        return w_update

    def get_update(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get the update to be applied to the base weight matrix.
        
        Args:
            w: Base weight matrix (optional)
            
        Returns:
            Weight update matrix
        """
        return w if w is not None else torch.zeros(self.size)


class LoRA(BaseModule):
    """
    Low-Rank Adaptation (LoRA) module.
    
    LoRA approximates weight updates using the product of two low-rank matrices:
    ΔW = B @ A, where B ∈ R^(d×r) and A ∈ R^(r×k), with r << min(d,k)
    """
    
    def __init__(self, size: Tuple[int, int], rank: int):
        """
        Initialize LoRA module.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
            rank: Rank of the low-rank decomposition
        """
        super(LoRA, self).__init__(size)
        self.rank = rank
        self.B = nn.Parameter(torch.zeros(size[0], rank))
        self.A = nn.Parameter(torch.zeros(rank, size[1]))
        
        # Initialize with Kaiming uniform initialization
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
    
    def get_update(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the LoRA update: B @ A"""
        return self.B @ self.A


class SinLoRA(LoRA):
    """
    Sinusoidal LoRA variant that applies sine activation to the low-rank product.
    
    This variant uses: ΔW = sin(ω * B @ A) * α
    where ω is the frequency parameter and α is the scaling factor.
    """
    
    def __init__(self, size: Tuple[int, int], rank: int, frequency: float = 20.0, scaling: float = 1.0):
        """
        Initialize SinLoRA module.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
            rank: Rank of the low-rank decomposition
            frequency: Frequency parameter for sine activation
            scaling: Scaling factor for the output
        """
        super(SinLoRA, self).__init__(size, rank)
        self.frequency = frequency
        self.scaling = scaling

    def get_update(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the SinLoRA update: sin(ω * B @ A) * α"""
        return torch.sin(self.frequency * self.B @ self.A) * self.scaling


class FullFineTuning(BaseModule):
    """
    Full fine-tuning baseline that learns the entire weight matrix.
    """
    
    def __init__(self, size: Tuple[int, int]):
        """
        Initialize full fine-tuning module.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
        """
        super(FullFineTuning, self).__init__(size)
        self.w = nn.Parameter(torch.zeros(size[0], size[1]))
        
    def get_update(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the full weight matrix"""
        return self.w


class KroneckerAdapter(BaseModule):
    """
    Kronecker Adapter using Kronecker product structure for parameter efficiency.
    
    This method represents the weight update as a Kronecker product of two smaller matrices.
    """
    
    def __init__(self, size: Tuple[int, int], rank: int):
        """
        Initialize Kronecker Adapter.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
            rank: Effective rank parameter
        """
        super(KroneckerAdapter, self).__init__(size)
        self.scaling = 1.0
        
        # Calculate dimensions for Kronecker factors
        d1 = 0 if size[0] % rank == 0 else 1
        d2 = 0 if size[1] % rank == 0 else 1
        
        self.B = nn.Parameter(torch.zeros(size[0] // rank, size[1] // rank))
        self.A = nn.Parameter(torch.zeros(rank + d1, rank + d2))
        
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def get_update(self, w: torch.Tensor) -> torch.Tensor:
        """Get the Kronecker product update"""
        if w is None:
            raise ValueError("Kronecker Adapter requires base weight matrix")
            
        # Compute Kronecker product
        update = (self.A[:, None, :, None] * self.B[None, :, None, :]).view(
            self.A.size(0) * self.B.size(0), 
            self.A.size(1) * self.B.size(1)
        )
        
        # Truncate to match target size
        update = update[:w.shape[0], :w.shape[1]]
        return update * self.scaling


class KRAdapter(BaseModule):
    """
    Kronecker-factorized Rank Adapter.
    
    A parameter-efficient method that uses Kronecker-structured low-rank decomposition.
    """
    
    def __init__(self, size: Tuple[int, int], rank: int = 2):
        """
        Initialize KR Adapter.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
            rank: Rank parameter for factorization
        """
        super(KRAdapter, self).__init__(size)
        self.scaling = 1.0
        
        # Calculate effective rank
        effective_rank = int(max(size) ** (1 / rank))
        rank2 = max(size) // effective_rank
        d1 = 0 if max(size) % effective_rank == 0 else 1
        
        self.A = nn.Parameter(torch.zeros(min(size), 1, rank2 + d1))
        self.B = nn.Parameter(torch.zeros(min(size), effective_rank, 1))
        
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def get_update(self, w: torch.Tensor) -> torch.Tensor:
        """Get the KR Adapter update"""
        if w is None:
            raise ValueError("KR Adapter requires base weight matrix")
            
        update = (self.A * self.B).flatten(start_dim=1)
        
        # Transpose if dimensions don't match
        if update.shape[0] != w.shape[0]:
            update = update.T
            
        # Truncate to match target size
        update = update[:, :w.shape[1]]
        return update * self.scaling


class RandLoRA(BaseModule):
    """
    Random LoRA with fixed random basis matrices.
    
    This method uses pre-computed random basis matrices that are frozen during training,
    while only learning the combination coefficients.
    """
    
    def __init__(self, size: Tuple[int, int], rank: int):
        """
        Initialize Random LoRA.
        
        Args:
            size: Tuple of (output_dim, input_dim) for the weight matrix
            rank: Rank of the decomposition
        """
        super(RandLoRA, self).__init__(size)
        self.scaling = 1.0
        
        # Calculate number of basis matrices
        n_basis = math.ceil(min(size) / rank) + 1
        
        # Create frozen random basis matrices
        self.basis_a = nn.Parameter(
            torch.zeros(n_basis, size[0], rank), 
            requires_grad=False
        )
        self.basis_b = nn.Parameter(
            torch.zeros(1, rank, size[1]), 
            requires_grad=False
        )
        
        # Initialize basis matrices
        torch.nn.init.kaiming_uniform_(self.basis_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.basis_b, a=math.sqrt(5))
        
        # Normalize basis matrices
        self.basis_a.data = self.basis_a.data / self.basis_a.data.std()
        self.basis_b.data = self.basis_b.data / self.basis_b.data.std()
        
        # Learnable combination coefficients
        self.A = nn.Parameter(torch.ones(n_basis, rank) / n_basis)
        self.B = nn.Parameter(torch.zeros(n_basis, size[1]))

    def get_update(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the Random LoRA update"""
        # Combine basis matrices with learned coefficients
        basis_combined = self.basis_a.permute(1, 0, 2).flatten(start_dim=1)
        coeff_combined = (self.A[:, :, None] * self.basis_b * self.B[:, None, :]).flatten(end_dim=-2)
        
        update = basis_combined @ coeff_combined
        return update * self.scaling


# Utility Functions

@torch.no_grad
def get_spectral_energy(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the spectral energy (sum of squared singular values) of a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Spectral energy value
    """
    _, S, _ = torch.linalg.svd(matrix)
    return torch.sum(S ** 2)


@torch.no_grad
def get_svd_error(s_pred: torch.Tensor, s_target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean squared error between predicted and target singular values.
    
    Args:
        s_pred: Predicted singular values
        s_target: Target singular values
        
    Returns:
        RMS error between singular value spectra
    """
    return ((s_pred - s_target) ** 2).sqrt().mean()


@torch.no_grad
def get_effective_rank(matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Calculate the effective rank of a matrix using entropy of normalized singular values.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Tuple of (singular_values, effective_rank)
    """
    matrix = matrix.float()
    _, S, _ = torch.linalg.svd(matrix)
    
    # Normalize singular values to get probability distribution
    S_normalized = S / S.sum()
    S_normalized = S_normalized + 1e-10  # Add small epsilon for numerical stability
    
    # Calculate effective rank using entropy: exp(-Σ p_i * log(p_i))
    entropy = -(S_normalized * torch.log(S_normalized)).sum()
    effective_rank = torch.exp(entropy).item()
    
    return S, effective_rank


def create_synthetic_matrices(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    """
    Create various types of synthetic matrices for testing PEFT algorithms.
    
    Args:
        size: Tuple of (height, width) for matrices
        
    Returns:
        Dictionary of synthetic matrices
    """
    matrices = {}
    
    # Random noise
    matrices['Random noise'] = torch.randn(size)
    
    # Sparse noise (90% dropout)
    sparse_matrix = torch.randn(size)
    matrices['Sparse noise'] = F.dropout(sparse_matrix, p=0.9)
    
    # PCA whitened noise
    matrices['PCA whitened noise'] = pca_whitening(size)
    
    # Low rank matrix
    low_rank_matrix = torch.randn(size)
    U, S, V = torch.linalg.svd(low_rank_matrix, full_matrices=False)
    S[len(S) // 4:] = 0  # Zero out 75% of singular values
    matrices['Low rank'] = U @ torch.diag(S) @ V
    
    # High frequency patterns
    matrices['High freq'] = create_frequency_matrix([1000, 10000], size)
    
    # Low frequency patterns  
    matrices['Low freq'] = create_frequency_matrix([1, 100], size)
    
    return matrices


def create_clip_matrices(size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    """
    Create matrices from CLIP model weights.
    
    Args:
        size: Target size for matrices
        
    Returns:
        Dictionary containing CLIP-based matrices
    """
    matrices = {}
    
    try:
        # Load CLIP model
        model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        state_dict = model.state_dict()
        
        # Get language model attention weights
        lang_keys = [k for k in state_dict.keys() if 'in_proj_weight' in k and 'visual' not in k]
        if lang_keys:
            weight = state_dict[lang_keys[-1]]
            matrices['CLIP language'] = weight[:size[0], :size[1]]
        
        # Get vision model attention weights
        vision_keys = [k for k in state_dict.keys() if 'in_proj_weight' in k and 'visual' in k]
        if vision_keys:
            weight = state_dict[vision_keys[-1]]
            matrices['CLIP vision'] = weight[:size[0], :size[1]]
            
    except Exception as e:
        warnings.warn(f"Could not load CLIP weights: {e}")
    
    return matrices


def pca_whitening(size: Tuple[int, int], eps: float = 1e-5) -> torch.Tensor:
    """
    Generate PCA whitened random data.
    
    Args:
        size: Shape of output matrix
        eps: Small value for numerical stability
        
    Returns:
        PCA whitened matrix
    """
    data_matrix = torch.randn(size)
    mean = torch.mean(data_matrix, dim=0)
    centered_data = data_matrix - mean
    cov_matrix = torch.cov(centered_data.T)
    
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    x_pca = torch.matmul(centered_data, eigenvectors)
    whitened_data = x_pca / torch.sqrt(eigenvalues + eps)
    
    return whitened_data


def create_frequency_matrix(freq_range: List[float], size: Tuple[int, int]) -> torch.Tensor:
    """
    Create a matrix with sinusoidal patterns at specified frequencies.
    
    Args:
        freq_range: [min_freq, max_freq] for frequency range
        size: Shape of output matrix
        
    Returns:
        Matrix with frequency patterns
    """
    low_freq, high_freq = freq_range
    s0, s1 = size
    
    freq_matrix = torch.empty(0, s1)
    time = torch.arange(0, 1, 1 / s1)
    
    for f in np.linspace(low_freq, high_freq, s0):
        # Create base sinusoid with random phase
        offset = random.random()
        freq_vector = torch.sin(2 * torch.pi * f * time + offset).unsqueeze(0)
        
        # Add random harmonics
        for _ in range(random.randint(0, 5)):
            rand_f = torch.rand(1) * (high_freq - low_freq) + low_freq
            freq_vector += torch.sin(2 * torch.pi * rand_f * time).unsqueeze(0)
            
        freq_matrix = torch.cat((freq_matrix, freq_vector), dim=0)
    
    return freq_matrix


def compute_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Frobenius norm loss between predicted and target matrices.
    
    Args:
        predicted: Predicted matrix
        target: Target matrix
        
    Returns:
        Frobenius norm loss
    """
    return torch.norm(predicted - target, p='fro')


def train_model(target_matrix: torch.Tensor, model: BaseModule, 
               n_iterations: int, learning_rate: float = 1e-1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Train a PEFT model to approximate a target matrix.
    
    Args:
        target_matrix: Target weight matrix to approximate
        model: PEFT model to train
        n_iterations: Number of training iterations
        learning_rate: Learning rate for optimization
        
    Returns:
        Tuple of (final_approximation, final_loss)
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    target_matrix = target_matrix.detach()
    
    for iteration in range(n_iterations):
        approximation = model.update(target_matrix)
        loss = compute_loss(approximation, target_matrix)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if iteration % 50 == 0:
            print(f"Iteration {iteration}: Loss = {loss.item():.4f}")
    
    return approximation.detach(), loss


def run_experiment(size: Tuple[int, int] = (1024, 768), n_iterations: int = 300,
                  learning_rate: float = 1e-1) -> Dict[str, Any]:
    """
    Run the complete PEFT approximation experiment.
    
    Args:
        size: Size of weight matrices to test
        n_iterations: Number of training iterations per model
        learning_rate: Learning rate for optimization
        
    Returns:
        Dictionary containing experimental results
    """
    print("Starting PEFT Algorithm Comparison Experiment")
    print(f"Matrix size: {size}")
    print(f"Training iterations: {n_iterations}")
    print("=" * 60)
    
    # Create test matrices
    synthetic_matrices = create_synthetic_matrices(size)
    clip_matrices = create_clip_matrices(size)
    all_matrices = {**synthetic_matrices, **clip_matrices}
    
    # Algorithm configurations
    n_params_budget = 2 * min(size) * math.sqrt(max(size))  # Parameter budget
    lora_rank = int(n_params_budget / (size[0] + size[1])) + 1
    
    algorithms = {
        'LoRA': lambda: LoRA(size, rank=lora_rank),
        'SinLoRA': lambda: SinLoRA(size, rank=lora_rank),
        'KRAdapter': lambda: KRAdapter(size),
        'RandLoRA': lambda: RandLoRA(size, rank=lora_rank // 4),
        'Kronecker': lambda: KroneckerAdapter(size, rank=11)
    }
    
    results = {}
    
    # Test each matrix type
    for matrix_name, target_matrix in all_matrices.items():
        print(f"\nTesting with {matrix_name} matrix...")
        target_matrix = target_matrix.to(DEVICE).half()
        
        # Get target statistics
        target_svd, target_eff_rank = get_effective_rank(target_matrix.float())
        results[matrix_name] = {
            'Target': {
                'Singular values': target_svd.tolist(),
                'Nuclear error': 0.0,
                'Effective rank': target_eff_rank
            }
        }
        
        # Test each algorithm
        for algo_name, algo_constructor in algorithms.items():
            try:
                print(f"  Training {algo_name}...")
                model = algo_constructor()
                
                # Count parameters
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"    Parameters: {n_params:,}")
                
                # Train model
                approximation, loss = train_model(target_matrix, model, n_iterations, learning_rate)
                
                # Evaluate results
                pred_svd, pred_eff_rank = get_effective_rank(approximation.float())
                nuclear_error = get_svd_error(pred_svd, target_svd).item()
                
                results[matrix_name][algo_name] = {
                    'Singular values': pred_svd.tolist(),
                    'Nuclear error': nuclear_error,
                    'Effective rank': pred_eff_rank,
                    'Parameters': n_params,
                    'Final loss': loss.item()
                }
                
                print(f"    Nuclear error: {nuclear_error:.3f}")
                
            except Exception as e:
                print(f"    Error with {algo_name}: {e}")
                continue
    
    return results


def create_visualizations(results: Dict[str, Any], save_plots: bool = True):
    """
    Create visualizations of the experimental results with error relative to LoRA.
    
    Args:
        results: Results dictionary from run_experiment
        save_plots: Whether to save plots to files
    """
    matrix_names = list(results.keys())
    algorithm_names = ['LoRA', 'SinLoRA', 'KRAdapter', 'RandLoRA', 'Kronecker']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create nuclear error comparison plot (relative to LoRA)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(matrix_names))
    bar_width = 0.15
    
    for i, algo_name in enumerate(algorithm_names):
        if algo_name in ['LoRA', 'SinLoRA', 'KRAdapter', 'RandLoRA', 'Kronecker']:
            relative_errors = []
            absolute_errors = []
            
            for matrix_name in matrix_names:
                if algo_name in results[matrix_name] and 'LoRA' in results[matrix_name]:
                    algo_error = results[matrix_name][algo_name]['Nuclear error']
                    lora_error = results[matrix_name]['LoRA']['Nuclear error']
                    
                    # Calculate relative error (as percentage of LoRA's error)
                    if lora_error > 0:
                        relative_error = (algo_error / lora_error) * 100
                    else:
                        relative_error = 100 if algo_name == 'LoRA' else 0
                    
                    relative_errors.append(relative_error)
                    absolute_errors.append(algo_error)
                else:
                    relative_errors.append(0)
                    absolute_errors.append(0)
            
            bars = ax.bar(x + i * bar_width, relative_errors, bar_width, 
                         label=algo_name, color=colors[i % len(colors)], alpha=0.8)
            
            # Add value labels on bars (showing absolute errors for reference)
            for j, (bar, abs_error) in enumerate(zip(bars, absolute_errors)):
                height = bar.get_height()
                if height > 0:
                    # Show absolute error value for context
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(relative_errors)*0.01,
                           f'{abs_error:.2f}', ha='center', va='bottom', 
                           rotation=45, fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Matrix Type', fontsize=12)
    ax.set_ylabel('Relative Nuclear Reconstruction Error (%)\n(LoRA = 100%)', fontsize=12)
    ax.set_title('PEFT Algorithm Performance Comparison (Relative to LoRA)', fontsize=14)
    ax.set_xticks(x + bar_width * 2)
    ax.set_xticklabels(matrix_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 100% for LoRA reference
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(matrix_names)-0.5, 105, 'LoRA baseline (100%)', 
            fontsize=10, alpha=0.7, ha='right')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('peft_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table with both absolute and relative errors
    print("\n" + "="*100)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*100)
    print(f"{'Matrix Type':<20} {'LoRA':<12} {'SinLoRA':<12} {'KRAdapter':<12} {'RandLoRA':<12} {'Kronecker':<12}")
    print(f"{'(Absolute/Relative)':<20} {'(Abs/Rel%)':<12} {'(Abs/Rel%)':<12} {'(Abs/Rel%)':<12} {'(Abs/Rel%)':<12} {'(Abs/Rel%)':<12}")
    print("-"*100)
    
    for matrix_name in matrix_names:
        row = f"{matrix_name:<20}"
        lora_error = results[matrix_name].get('LoRA', {}).get('Nuclear error', 1.0)
        
        for algo_name in algorithm_names:
            if algo_name in results[matrix_name]:
                error = results[matrix_name][algo_name]['Nuclear error']
                relative_error = (error / lora_error) * 100 if lora_error > 0 else 0
                row += f"{error:.2f}/{relative_error:.0f}% "
                # Pad to 12 characters
                row = row.ljust(len(row) + (12 - len(f"{error:.2f}/{relative_error:.0f}% ")))
            else:
                row += f"{'N/A':<12} "
        print(row)


def main():
    """Main execution function using configuration file."""
    print("=" * 80)
    print("PARAMETER-EFFICIENT FINE-TUNING ALGORITHM COMPARISON")
    print("=" * 80)
    print(f"Configuration loaded from config.py")
    print(f"Device: {DEVICE}")
    print(f"Matrix size: {MATRIX_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print("")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_CONFIG['output_directory'], exist_ok=True)
    
    # Run experiment using configuration
    results = run_experiment(
        size=MATRIX_SIZE,
        n_iterations=TRAINING_CONFIG['n_iterations'],
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    
    # Save results if configured
    if OUTPUT_CONFIG['save_results']:
        output_file = os.path.join(
            OUTPUT_CONFIG['output_directory'], 
            OUTPUT_CONFIG['results_filename']
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Create visualizations if configured
    if OUTPUT_CONFIG['save_plots']:
        create_visualizations(results, save_plots=True)
    else:
        create_visualizations(results, save_plots=False)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Matrix types evaluated: {sum(MATRIX_TYPES.values())}")
    print(f"Algorithms evaluated: {sum(ALGORITHMS.values())}")
    if OUTPUT_CONFIG['save_results']:
        print(f"Results saved to: {OUTPUT_CONFIG['results_filename']}")
    if OUTPUT_CONFIG['save_plots']:
        print(f"Plots saved to: {OUTPUT_CONFIG['plot_filename']}")


if __name__ == "__main__":
    main()
