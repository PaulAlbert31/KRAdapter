"""
Configuration file for PEFT Algorithm Comparison Experiments

This file contains all configurable parameters for the experimental framework.
Researchers can modify these settings without changing the main code.
"""

from typing import Tuple, Dict, Any
import torch

# =============================================================================
# EXPERIMENTAL CONFIGURATION
# =============================================================================

# Device Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_MIXED_PRECISION = False  # Enable for memory efficiency on GPU

# Reproducibility
RANDOM_SEED = 42

# =============================================================================
# MATRIX CONFIGURATION  
# =============================================================================

# Matrix dimensions for experiments
# Use smaller sizes for quick testing: (128, 128)
# Use larger sizes for publication results: (1024, 768) or (2048, 1024)
MATRIX_SIZE = (1024, 768)

# Matrix types to evaluate (set to False to skip)
MATRIX_TYPES = {
    'Random noise': True,
    'Sparse noise': True, 
    'PCA whitened noise': True,
    'Low rank': True,
    'CLIP language': True,  # Requires open_clip installation
    'CLIP vision': True,    # Requires open_clip installation  
    'High freq': True,
    'Low freq': True
}

# =============================================================================
# ALGORITHM CONFIGURATION
# =============================================================================

# Algorithms to evaluate (set to False to skip)
ALGORITHMS = {
    'LoRA': True,
    'SinLoRA': True,
    'KRAdapter': True,
    'RandLoRA': True,
    'Kronecker': True
}

# Parameter budget strategy
# Options: 'equal_budget', 'equal_rank', 'custom'
PARAMETER_STRATEGY = 'equal_budget'

# Custom parameter settings (used if PARAMETER_STRATEGY = 'custom')
CUSTOM_PARAMS = {
    'LoRA': {'rank': 64},
    'SinLoRA': {'rank': 64, 'frequency': 20.0, 'scaling': 1.0},
    'KRAdapter': {'rank': 2},
    'RandLoRA': {'rank': 16},
    'Kronecker': {'rank': 11}
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Training parameters
TRAINING_CONFIG = {
    'n_iterations': 300,
    'learning_rate': 1e-1,
    'optimizer': 'AdamW',
    'weight_decay': 0.01,
    'scheduler': None,  # Options: None, 'cosine', 'linear'
    'warmup_steps': 0,
    'gradient_clip': None,  # Set to value for gradient clipping
}

# Logging configuration  
LOGGING_CONFIG = {
    'log_interval': 50,     # Print loss every N iterations
    'save_progress': True,  # Save intermediate results
    'verbose': True         # Print detailed progress
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Evaluation metrics to compute
METRICS = {
    'nuclear_error': True,
    'frobenius_error': True, 
    'effective_rank': True,
    'spectral_energy': True,
    'parameter_count': True
}

# Convergence criteria
CONVERGENCE_CONFIG = {
    'tolerance': 1e-6,      # Loss change threshold for early stopping
    'patience': 50,         # Iterations to wait before early stopping
    'min_iterations': 100   # Minimum iterations before early stopping
}

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

# Output settings
OUTPUT_CONFIG = {
    'save_results': True,
    'results_filename': 'experimental_results.json',
    'save_plots': True,
    'plot_filename': 'peft_comparison.png',
    'plot_dpi': 300,
    'save_individual_plots': False,
    'output_directory': './'
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'color_scheme': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'font_size': 12,
    'title_size': 14,
    'show_error_bars': False,
    'show_parameter_counts': True
}

# =============================================================================
# MATRIX GENERATION PARAMETERS
# =============================================================================

# Sparse matrix configuration
SPARSE_CONFIG = {
    'dropout_rate': 0.9,    # Sparsity level
    'structured': False     # Use structured sparsity
}

# Low-rank matrix configuration  
LOW_RANK_CONFIG = {
    'rank_fraction': 0.25,  # Fraction of singular values to keep
    'noise_level': 0.0      # Added noise level
}

# Frequency matrix configuration
FREQUENCY_CONFIG = {
    'low_freq_range': [1, 100],        # Low frequency range
    'high_freq_range': [1000, 10000],  # High frequency range
    'harmonics': True,                  # Add random harmonics
    'max_harmonics': 5                  # Maximum number of harmonics
}

# PCA whitening configuration
PCA_CONFIG = {
    'regularization': 1e-5,  # Regularization for numerical stability
    'center_data': True      # Center data before PCA
}

# =============================================================================
# CLIP MODEL CONFIGURATION
# =============================================================================

CLIP_CONFIG = {
    'model_name': 'ViT-L-14',
    'pretrained': 'openai',
    'extract_layers': ['language', 'vision'],
    'weight_types': ['in_proj_weight'],  # Attention projection weights
    'fallback_on_error': True
}

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Memory optimization
MEMORY_CONFIG = {
    'chunk_size': None,      # Process matrices in chunks if memory limited
    'clear_cache': True,     # Clear GPU cache between experiments
    'low_memory_mode': False # Use memory-efficient algorithms
}

# Numerical stability
NUMERICAL_CONFIG = {
    'eps': 1e-10,           # Small constant for numerical stability
    'svd_driver': None,     # SVD algorithm ('gesvd', 'gesvdj', None for default)
    'matrix_dtype': torch.float32  # Data type for matrices
}

# Parallel processing
PARALLEL_CONFIG = {
    'use_multiprocessing': False,  # Parallel evaluation across algorithms
    'n_workers': 4,               # Number of parallel workers
    'batch_algorithms': False     # Batch multiple algorithms together
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_parameter_budget(matrix_size: Tuple[int, int]) -> int:
    """Calculate parameter budget based on matrix size."""
    import math
    return int(2 * min(matrix_size) * math.sqrt(max(matrix_size)))

def get_lora_rank(matrix_size: Tuple[int, int], budget: int) -> int:
    """Calculate LoRA rank given parameter budget."""
    return max(1, int(budget / (matrix_size[0] + matrix_size[1])) + 1)

def validate_config() -> bool:
    """Validate configuration settings."""
    # Check if at least one algorithm is enabled
    if not any(ALGORITHMS.values()):
        raise ValueError("At least one algorithm must be enabled")
    
    # Check if at least one matrix type is enabled
    if not any(MATRIX_TYPES.values()):
        raise ValueError("At least one matrix type must be enabled")
    
    # Validate matrix dimensions
    if min(MATRIX_SIZE) < 1:
        raise ValueError("Matrix dimensions must be positive")
    
    return True

# Validate configuration on import
validate_config()
