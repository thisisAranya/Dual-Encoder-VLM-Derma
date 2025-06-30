"""
utils.py - Utility functions for Dual Vision Encoder
"""

import torch
import torch.nn as nn
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def setup_logging(output_dir: str, log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    save_dir: str,
    is_best: bool = False,
    additional_info: Optional[Dict] = None
) -> None:
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Save latest checkpoint
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"New best checkpoint saved at epoch {epoch} with loss {loss:.4f}")
    
    # Save epoch checkpoint
    epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, epoch_path)
    
    print(f"Checkpoint saved: {latest_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    
    return checkpoint


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def print_model_info(model: nn.Module) -> None:
    """Print detailed model information"""
    param_counts = count_parameters(model)
    
    print("\n" + "="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Frozen parameters: {param_counts['frozen']:,}")
    print(f"Trainable percentage: {param_counts['trainable']/param_counts['total']*100:.2f}%")
    
    # Print component-wise parameter counts
    print("\nComponent-wise parameter counts:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        trainable_module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name}: {module_params:,} total, {trainable_module_params:,} trainable")
    
    print("="*50 + "\n")


def visualize_fusion_weights(
    fusion_weights: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Encoder Fusion Weights"
) -> None:
    """Visualize fusion weights between encoders"""
    plt.figure(figsize=(8, 6))
    
    encoders = list(fusion_weights.keys())
    weights = list(fusion_weights.values())
    colors = ['#FF6B6B', '#4ECDC4']
    
    # Create pie chart
    plt.pie(weights, labels=encoders, colors=colors, autopct='%1.2f%%', startangle=90)
    plt.title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fusion weights plot saved to: {save_path}")
    
    plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None
) -> None:
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def analyze_model_outputs(
    model,
    test_messages: list,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze model outputs and fusion weights on test data"""
    results = {
        'fusion_weights': {'qwen': [], 'dinov2': []},
        'responses': [],
        'confidence_scores': []
    }
    
    for messages in test_messages:
        try:
            # Get model response with analysis
            output = model.forward(messages, return_intermediates=True)
            weights = model.analyze_fusion_weights(messages)
            
            if 'error' not in weights:
                results['fusion_weights']['qwen'].append(weights['qwen_weight'])
                results['fusion_weights']['dinov2'].append(weights['dinov2_weight'])
                results['confidence_scores'].append(weights['confidence'])
            
            results['responses'].append(output['generated_text'][0])
            
        except Exception as e:
            print(f"Error analyzing message: {e}")
            continue
    
    # Calculate statistics
    if results['fusion_weights']['qwen']:
        stats = {
            'avg_qwen_weight': np.mean(results['fusion_weights']['qwen']),
            'avg_dinov2_weight': np.mean(results['fusion_weights']['dinov2']),
            'std_qwen_weight': np.std(results['fusion_weights']['qwen']),
            'std_dinov2_weight': np.std(results['fusion_weights']['dinov2']),
            'avg_confidence': np.mean(results['confidence_scores']),
            'num_samples': len(results['fusion_weights']['qwen'])
        }
        results['statistics'] = stats
        
        print("\nFusion Weight Analysis:")
        print(f"Average Qwen-VL weight: {stats['avg_qwen_weight']:.3f} ± {stats['std_qwen_weight']:.3f}")
        print(f"Average DINOv2 weight: {stats['avg_dinov2_weight']:.3f} ± {stats['std_dinov2_weight']:.3f}")
        print(f"Average confidence: {stats['avg_confidence']:.3f}")
        print(f"Samples analyzed: {stats['num_samples']}")
    
    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save raw results
        with open(os.path.join(save_dir, 'analysis_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        if results['fusion_weights']['qwen']:
            # Fusion weights distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(results['fusion_weights']['qwen'], bins=20, alpha=0.7, color='blue', label='Qwen-VL')
            plt.hist(results['fusion_weights']['dinov2'], bins=20, alpha=0.7, color='red', label='DINOv2')
            plt.xlabel('Fusion Weight')
            plt.ylabel('Frequency')
            plt.title('Distribution of Fusion Weights')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(results['confidence_scores'], bins=20, alpha=0.7, color='green')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Confidence Scores')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'fusion_analysis.png'), dpi=300, bbox_inches='tight')
            plt.show()
    
    return results


def create_config_template(save_path: str) -> None:
    """Create a template configuration file"""
    config = {
        "project_name": "dual_vision_encoder",
        "output_dir": "./outputs",
        "qwen_model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dinov2_model_name": "facebook/dinov2-base",
        "common_dim": 1024,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "min_lr": 1e-6,
        "num_epochs": 10,
        "batch_size": 4,
        "max_new_tokens": 100,
        "max_grad_norm": 1.0,
        "diversity_loss_weight": 0.1,
        "save_steps": 1,
        "log_steps": 10,
        "num_workers": 4,
        "use_wandb": True,
        "train_data_path": "./data/train.json",
        "val_data_path": "./data/val.json"
    }
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration template saved to: {save_path}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters"""
    required_keys = [
        'qwen_model_name', 'dinov2_model_name', 'common_dim',
        'learning_rate', 'num_epochs', 'batch_size'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Missing required configuration keys: {missing_keys}")
        return False
    
    # Validate ranges
    if config['learning_rate'] <= 0:
        print("Learning rate must be positive")
        return False
    
    if config['batch_size'] <= 0:
        print("Batch size must be positive")
        return False
    
    if config['common_dim'] <= 0:
        print("Common dimension must be positive")
        return False
    
    print("Configuration validation passed")
    return True


def get_device_info() -> Dict[str, Any]:
    """Get information about available devices"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_names': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info['device_names'].append(torch.cuda.get_device_name(i))
    
    return device_info


def print_device_info() -> None:
    """Print device information"""
    info = get_device_info()
    
    print("\n" + "="*40)
    print("DEVICE INFORMATION")
    print("="*40)
    print(f"CUDA Available: {info['cuda_available']}")
    print(f"Number of GPUs: {info['cuda_device_count']}")
    
    if info['cuda_available']:
        print(f"Current Device: {info['current_device']}")
        for i, name in enumerate(info['device_names']):
            print(f"GPU {i}: {name}")
    else:
        print("Running on CPU")
    
    print("="*40 + "\n")


def cleanup_checkpoints(checkpoint_dir: str, keep_last_n: int = 5) -> None:
    """Clean up old checkpoint files, keeping only the last N"""
    if not os.path.exists(checkpoint_dir):
        return
    
    # Find all epoch checkpoints
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
            epoch_num = int(file.split('_')[-1].split('.')[0])
            checkpoint_files.append((epoch_num, file))
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[0])
    
    # Remove old checkpoints
    if len(checkpoint_files) > keep_last_n:
        files_to_remove = checkpoint_files[:-keep_last_n]
        for epoch_num, filename in files_to_remove:
            file_path = os.path.join(checkpoint_dir, filename)
            os.remove(file_path)
            print(f"Removed old checkpoint: {filename}")
        
        print(f"Kept last {keep_last_n} checkpoints")


if __name__ == "__main__":
    # Example usage
    print("Testing utilities...")
    
    # Print device info
    print_device_info()
    
    # Create config template
    create_config_template("./config_template.json")
    
    # Test config validation
    with open("./config_template.json", 'r') as f:
        config = json.load(f)
    
    is_valid = validate_config(config)
    print(f"Config validation result: {is_valid}")
