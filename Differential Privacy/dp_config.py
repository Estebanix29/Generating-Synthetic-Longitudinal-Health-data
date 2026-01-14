
from dataclasses import dataclass, field
from typing import Optional, Literal
import yaml


@dataclass
class DPConfig:
    # Privacy parameters
    target_epsilon: float = 8.0
    target_delta: float = 1e-5
    max_grad_norm: float = 1.0
    
    # Training parameters
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 2e-3
    optimizer: Literal['Adam', 'SGD'] = 'Adam'
    
    # Optimizer-specific parameters
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    sgd_momentum: float = 0.9
    weight_decay: float = 1e-4
    
    # Advanced DP parameters
    accountant: Literal['rdp', 'gdp'] = 'rdp'  # Rényi DP (recommended)
    secure_mode: bool = False
    virtual_batch_size: Optional[int] = None  # For gradient accumulation
    grad_sample_mode: Literal['hooks', 'ew'] = 'hooks'  # 'hooks' or 'ew' (expandedweights)
    
    # Monitoring
    log_every_n_steps: int = 100
    save_every_n_epochs: int = 1
    privacy_alert_threshold: float = 0.9  # Alert when 90% of budget consumed
    
    # Dataset info (for delta calculation)
    dataset_size: int = 160000  # MIMIC-IV size
    
    def __post_init__(self):
        """Validate configuration and compute derived values."""
        # Validate epsilon
        if self.target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be positive, got {self.target_epsilon}")
        
        # Validate delta
        if self.target_delta <= 0 or self.target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {self.target_delta}")
        
        # Recommend delta based on dataset size
        recommended_delta = 1 / (self.dataset_size ** 2)
        if self.target_delta > 1 / self.dataset_size:
            print(f"Warning: delta={self.target_delta} is large for dataset_size={self.dataset_size}")
            print(f"   Recommended: delta ≤ 1/n² = {recommended_delta:.2e}")
        
        # Validate batch size
        if self.batch_size > self.dataset_size:
            raise ValueError(f"batch_size ({self.batch_size}) cannot exceed dataset_size ({self.dataset_size})")
        
        # Set virtual batch size if not specified
        if self.virtual_batch_size is None:
            self.virtual_batch_size = self.batch_size * 2
        
        # Validate gradient accumulation
        if self.virtual_batch_size < self.batch_size:
            raise ValueError(f"virtual_batch_size ({self.virtual_batch_size}) must be >= batch_size ({self.batch_size})")
        
        # Compute sampling rate
        self.sampling_rate = self.batch_size / self.dataset_size
        
        # Compute accumulation steps
        self.accumulation_steps = self.virtual_batch_size // self.batch_size
    
    @property
    def privacy_level(self) -> str:
        """Return human-readable privacy level."""
        if self.target_epsilon <= 1:
            return "Very Strong"
        elif self.target_epsilon <= 3:
            return "Strong"
        elif self.target_epsilon <= 8:
            return "Moderate"
        else:
            return "Practical"
    
    @property
    def expected_utility_loss(self) -> str:
        """Return expected utility loss range."""
        if self.target_epsilon <= 1:
            return "10-20%"
        elif self.target_epsilon <= 3:
            return "5-10%"
        elif self.target_epsilon <= 8:
            return "3-5%"
        else:
            return "2-3%"
    
    def summary(self):
        """Print configuration summary."""
        print("\n" + "-"*60)
        print("Differential Privacy Configuration")
        print("-"*60)
        print(f"\nPrivacy Parameters:")
        print(f"  Epsilon (ε):        {self.target_epsilon}")
        print(f"  Delta (δ):          {self.target_delta:.2e}")
        print(f"  Max Grad Norm (C):  {self.max_grad_norm}")
        print(f"  Privacy Level:      {self.privacy_level}")
        print(f"  Expected Utility Loss: {self.expected_utility_loss}")
        print(f"\nTraining Parameters:")
        print(f"  Batch Size:         {self.batch_size}")
        print(f"  Virtual Batch Size: {self.virtual_batch_size}")
        print(f"  Accumulation Steps: {self.accumulation_steps}")
        print(f"  Epochs:             {self.epochs}")
        print(f"  Learning Rate:      {self.learning_rate}")
        print(f"  Optimizer:          {self.optimizer}")
        print(f"\nPrivacy Amplification:")
        print(f"  Dataset Size:       {self.dataset_size:,}")
        print(f"  Sampling Rate (q):  {self.sampling_rate:.6f}")
        print(f"\nAdvanced Settings:")
        print(f"  Accountant:         {self.accountant.upper()}")
        print(f"  Secure Mode:        {self.secure_mode}")
        print(f"  Grad Sample Mode:   {self.grad_sample_mode}")
        print("-"*60 + "\n")
    
    def save_yaml(self, filepath: str):
        """Save configuration to YAML file."""
        config_dict = {
            'privacy': {
                'target_epsilon': self.target_epsilon,
                'target_delta': self.target_delta,
                'max_grad_norm': self.max_grad_norm,
            },
            'training': {
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'learning_rate': self.learning_rate,
                'optimizer': self.optimizer,
                'weight_decay': self.weight_decay,
            },
            'advanced': {
                'accountant': self.accountant,
                'secure_mode': self.secure_mode,
                'virtual_batch_size': self.virtual_batch_size,
            }
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_yaml(cls, filepath: str):
        """Load configuration from YAML file."""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested dict
        flat_dict = {}
        for section in config_dict.values():
            flat_dict.update(section)
        
        return cls(**flat_dict)


# Preset configurations

def get_moderate_privacy_config(dataset_size: int = 160000) -> DPConfig:
    """
    Moderate Privacy Configuration (ε=8) - RECOMMENDED
    
    Use Case: Research publications, IRB approval, balanced trade-off
    Expected Utility Loss: 3-5%
    Training Time: 2× baseline
    """
    return DPConfig(
        # Privacy: Moderate
        target_epsilon=8.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        
        # Training: Near-normal parameters
        batch_size=256,
        epochs=10,
        learning_rate=2e-3,
        optimizer='Adam',
        
        # Advanced
        accountant='rdp',
        virtual_batch_size=512,
        
        # Dataset
        dataset_size=dataset_size,
    )


def get_high_privacy_config(dataset_size: int = 160000) -> DPConfig:
    """
    High Privacy Configuration (ε=3) - STRONG GUARANTEE
    
    Use Case: Multi-institutional collaboration, public release
    Expected Utility Loss: 5-10%
    Training Time: 2.5× baseline
    """
    return DPConfig(
        # Privacy: Strong
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=0.5,  # Aggressive clipping
        
        # Training: Conservative parameters
        batch_size=512,  # Large batches for privacy amplification
        epochs=8,
        learning_rate=3e-3,  # Higher LR to compensate for noise
        optimizer='SGD',  # More stable with high noise
        sgd_momentum=0.9,
        weight_decay=1e-3,
        
        # Advanced
        accountant='rdp',
        virtual_batch_size=1024,
        
        # Dataset
        dataset_size=dataset_size,
    )


def get_low_privacy_config(dataset_size: int = 160000) -> DPConfig:
    """
    Low Privacy Configuration (ε=10) - MAXIMUM UTILITY
    
    Use Case: Single-institution, internal use, maximum utility with formal guarantee
    Expected Utility Loss: 2-3%
    Training Time: 2× baseline
    """
    return DPConfig(
        # Privacy: Practical
        target_epsilon=10.0,
        target_delta=1e-5,
        max_grad_norm=1.5,  # Mild clipping
        
        # Training: Near baseline
        batch_size=128,
        epochs=15,
        learning_rate=1.5e-3,
        optimizer='Adam',
        
        # Advanced
        accountant='rdp',
        virtual_batch_size=256,
        
        # Dataset
        dataset_size=dataset_size,
    )


def get_maximum_privacy_config(dataset_size: int = 160000) -> DPConfig:
    """
    Maximum Privacy Configuration (ε=1) - VERY STRONG GUARANTEE
    
    Use Case: Public data release, adversarial setting, regulatory requirement
    Expected Utility Loss: 10-20%
    Training Time: 3× baseline
    WARNING: May be unstable, requires careful tuning
    """
    return DPConfig(
        # Privacy: Very Strong
        target_epsilon=1.0,
        target_delta=1e-5,
        max_grad_norm=0.1,  # Very aggressive clipping
        
        # Training: High noise compensation
        batch_size=1024,  # Maximum privacy amplification
        epochs=5,  # Fewer epochs to conserve budget
        learning_rate=5e-3,  # Very high LR
        optimizer='SGD',
        sgd_momentum=0.9,
        weight_decay=1e-2,
        
        # Advanced
        accountant='rdp',
        virtual_batch_size=2048,
        
        # Dataset
        dataset_size=dataset_size,
    )


def create_custom_config(
    privacy_level: Literal['high', 'moderate', 'low'],
    dataset_size: int,
    available_memory_gb: int
) -> DPConfig:
    """
    Create custom DP config based on requirements.
    
    Args:
        privacy_level: 'high', 'moderate', or 'low'
        dataset_size: Number of training examples
        available_memory_gb: GPU memory available
    
    Returns:
        DPConfig: Customized configuration
    """
    # Select epsilon based on privacy level
    epsilon_map = {'high': 3.0, 'moderate': 8.0, 'low': 10.0}
    epsilon = epsilon_map[privacy_level]
    
    # Calculate optimal batch size based on memory
    # Rule of thumb: ~1GB per 32 batch size for transformers
    max_batch_size = int((available_memory_gb * 32) / 2)  # /2 for DP overhead
    
    # Calculate optimal batch size based on dataset size
    # Target sampling rate q = 0.001 to 0.01
    optimal_batch_size = int(dataset_size * 0.002)
    
    # Take minimum of memory and optimal
    batch_size = min(max_batch_size, optimal_batch_size, 512)
    batch_size = max(batch_size, 32)  # Minimum 32
    
    # Adjust other parameters based on privacy level
    if privacy_level == 'high':
        max_grad_norm = 0.5
        learning_rate = 3e-3
        epochs = 8
        optimizer = 'SGD'
    elif privacy_level == 'moderate':
        max_grad_norm = 1.0
        learning_rate = 2e-3
        epochs = 10
        optimizer = 'Adam'
    else:  # low
        max_grad_norm = 1.5
        learning_rate = 1.5e-3
        epochs = 15
        optimizer = 'Adam'
    
    return DPConfig(
        target_epsilon=epsilon,
        target_delta=1e-5,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        optimizer=optimizer,
        dataset_size=dataset_size,
    )


# Configuration registry

PRESET_CONFIGS = {
    'moderate': get_moderate_privacy_config,
    'high': get_high_privacy_config,
    'low': get_low_privacy_config,
    'maximum': get_maximum_privacy_config,
}


def get_preset_config(preset_name: str, dataset_size: int = 160000) -> DPConfig:
    """
    Get a preset configuration by name.
    
    Args:
        preset_name: One of 'moderate', 'high', 'low', 'maximum'
        dataset_size: Size of training dataset
    
    Returns:
        DPConfig: Preset configuration
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Choose from: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name](dataset_size)


if __name__ == "__main__":
    # Demo: Print all presets
    print("\nDifferential Privacy Configuration Presets\n")
    
    for preset_name in ['low', 'moderate', 'high', 'maximum']:
        print(f"\n{'-'*60}")
        print(f"Preset: {preset_name}")
        print(f"{'-'*60}")
        config = get_preset_config(preset_name)
        config.summary()
    
    # Demo: Custom configuration
    print(f"\n{'-'*60}")
    print("Custom configuration")
    print(f"{'-'*60}")
    custom_config = create_custom_config(
        privacy_level='moderate',
        dataset_size=160000,
        available_memory_gb=24
    )
    custom_config.summary()
    
    # Demo: Save/load YAML
    print("\nSaving configuration to YAML...")
    custom_config.save_yaml('config/dp_custom.yaml')
    
    print("\nDemonstration complete.")
