"""
training_examples.py - Examples and utilities for different training scenarios
"""

import json
import os
from typing import Dict, List, Any
from model import DualVisionEncoder
from train import DualVisionTrainer
from dataset import DualVisionDataset
from torch.utils.data import DataLoader


def create_config_for_scenario(scenario: str, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration for specific training scenarios
    
    Args:
        scenario: Training scenario name
        base_config: Base configuration dictionary
    
    Returns:
        Updated configuration for the scenario
    """
    config = base_config.copy()
    
    scenarios = {
        'vision_finetuning': {
            'description': 'Fine-tune vision encoders while keeping LLM frozen',
            'freeze_config': {
                'freeze_qwen_vision': False,
                'freeze_dinov2': False,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'training_config': {
                'learning_rate': 5e-5,
                'num_epochs': 15,
                'diversity_loss_weight': 0.2
            }
        },
        
        'language_adaptation': {
            'description': 'Adapt language model to new domain while keeping vision frozen',
            'freeze_config': {
                'freeze_qwen_vision': True,
                'freeze_dinov2': True,
                'freeze_llm': False,
                'freeze_fusion_layers': True,
                'freeze_projections': True
            },
            'training_config': {
                'learning_rate': 1e-5,
                'num_epochs': 10,
                'diversity_loss_weight': 0.0
            }
        },
        
        'fusion_learning': {
            'description': 'Learn optimal fusion strategy between encoders',
            'freeze_config': {
                'freeze_qwen_vision': True,
                'freeze_dinov2': True,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'training_config': {
                'learning_rate': 1e-3,
                'num_epochs': 20,
                'diversity_loss_weight': 0.5
            }
        },
        
        'progressive_training': {
            'description': 'Progressive training from fusion to full model',
            'curriculum_training': {
                'enabled': True,
                'stages': [
                    {
                        'mode': 'fusion_only',
                        'epochs': 5,
                        'description': 'Learn fusion strategy'
                    },
                    {
                        'mode': 'encoders_only',
                        'epochs': 10,
                        'description': 'Fine-tune vision understanding'
                    },
                    {
                        'mode': 'full',
                        'epochs': 5,
                        'description': 'End-to-end fine-tuning'
                    }
                ]
            },
            'training_config': {
                'learning_rate': 1e-4,
                'diversity_loss_weight': 0.1
            }
        },
        
        'medical_adaptation': {
            'description': 'Adapt model for medical imaging tasks',
            'freeze_config': {
                'freeze_qwen_vision': False,
                'freeze_dinov2': False,  # Unfreeze DINOv2 for medical images
                'freeze_llm': False,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'training_config': {
                'learning_rate': 2e-5,
                'num_epochs': 25,
                'diversity_loss_weight': 0.15
            }
        },
        
        'efficient_finetuning': {
            'description': 'Efficient fine-tuning with minimal parameters',
            'freeze_config': {
                'freeze_qwen_vision': True,
                'freeze_dinov2': True,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'training_config': {
                'learning_rate': 5e-4,
                'num_epochs': 30,
                'diversity_loss_weight': 0.3
            }
        }
    }
    
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")
    
    scenario_config = scenarios[scenario]
    
    # Update configuration
    if 'freeze_config' in scenario_config:
        config['freeze_config'] = scenario_config['freeze_config']
    
    if 'training_config' in scenario_config:
        config.update(scenario_config['training_config'])
    
    if 'curriculum_training' in scenario_config:
        config['curriculum_training'] = scenario_config['curriculum_training']
    
    config['scenario'] = scenario
    config['scenario_description'] = scenario_config['description']
    
    return config


def save_scenario_config(scenario: str, output_path: str, base_config_path: str = None):
    """Save configuration for a specific training scenario"""
    
    # Load base config or create default
    if base_config_path and os.path.exists(base_config_path):
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
    else:
        # Default base config
        base_config = {
            "project_name": "dual_vision_encoder",
            "output_dir": "./outputs",
            "qwen_model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "dinov2_model_name": "facebook/dinov2-base",
            "common_dim": 1024,
            "dropout": 0.1,
            "batch_size": 4,
            "max_new_tokens": 100,
            "max_grad_norm": 1.0,
            "save_steps": 1,
            "log_steps": 10,
            "num_workers": 4,
            "use_wandb": True,
            "train_data_path": "./data/train.json",
            "val_data_path": "./data/val.json"
        }
    
    # Create scenario-specific config
    scenario_config = create_config_for_scenario(scenario, base_config)
    
    # Save config
    with open(output_path, 'w') as f:
        json.dump(scenario_config, f, indent=2)
    
    print(f"Scenario config saved to: {output_path}")
    print(f"Description: {scenario_config['scenario_description']}")


def create_all_scenario_configs():
    """Create configuration files for all training scenarios"""
    scenarios = [
        'vision_finetuning',
        'language_adaptation', 
        'fusion_learning',
        'progressive_training',
        'medical_adaptation',
        'efficient_finetuning'
    ]
    
    os.makedirs('./configs', exist_ok=True)
    
    for scenario in scenarios:
        output_path = f"./configs/config_{scenario}.json"
        save_scenario_config(scenario, output_path)
    
    print(f"\nCreated {len(scenarios)} scenario configurations in ./configs/")


def analyze_training_requirements(scenario: str) -> Dict[str, Any]:
    """Analyze computational requirements for a training scenario"""
    
    # Load a dummy config to analyze
    base_config = {
        "qwen_model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "dinov2_model_name": "facebook/dinov2-base",
        "common_dim": 1024,
        "batch_size": 4
    }
    
    config = create_config_for_scenario(scenario, base_config)
    
    # Estimate parameter counts (rough estimates)
    param_estimates = {
        'qwen_vision': 1_000_000,    # ~1M parameters
        'dinov2': 86_000_000,        # ~86M parameters  
        'qwen_llm': 2_700_000_000,   # ~2.7B parameters
        'fusion_layers': 50_000,     # ~50K parameters
        'projections': 10_000        # ~10K parameters
    }
    
    freeze_config = config.get('freeze_config', {})
    
    trainable_params = 0
    frozen_params = 0
    
    components = {
        'freeze_qwen_vision': 'qwen_vision',
        'freeze_dinov2': 'dinov2', 
        'freeze_llm': 'qwen_llm',
        'freeze_fusion_layers': 'fusion_layers',
        'freeze_projections': 'projections'
    }
    
    for freeze_key, component in components.items():
        param_count = param_estimates[component]
        if freeze_config.get(freeze_key, False):
            frozen_params += param_count
        else:
            trainable_params += param_count
    
    total_params = trainable_params + frozen_params
    
    # Estimate memory requirements (rough)
    # Assumes float16 training
    model_memory_gb = total_params * 2 / (1024**3)  # 2 bytes per param
    gradient_memory_gb = trainable_params * 2 / (1024**3) 
    optimizer_memory_gb = trainable_params * 8 / (1024**3)  # AdamW states
    
    total_memory_gb = model_memory_gb + gradient_memory_gb + optimizer_memory_gb
    
    # Add batch and activation memory (rough estimate)
    batch_memory_gb = config.get('batch_size', 4) * 0.5  # ~0.5GB per sample
    total_memory_gb += batch_memory_gb
    
    return {
        'scenario': scenario,
        'description': config.get('scenario_description', ''),
        'parameters': {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params,
            'trainable_percentage': (trainable_params / total_params) * 100
        },
        'memory_estimates_gb': {
            'model': round(model_memory_gb, 2),
            'gradients': round(gradient_memory_gb, 2),
            'optimizer': round(optimizer_memory_gb, 2),
            'batch_activations': round(batch_memory_gb, 2),
            'total_estimated': round(total_memory_gb, 2)
        },
        'training_config': {
            'learning_rate': config.get('learning_rate', 'N/A'),
            'epochs': config.get('num_epochs', 'N/A'),
            'batch_size': config.get('batch_size', 'N/A')
        }
    }


def print_scenario_comparison():
    """Print comparison of all training scenarios"""
    scenarios = [
        'vision_finetuning',
        'language_adaptation',
        'fusion_learning', 
        'progressive_training',
        'medical_adaptation',
        'efficient_finetuning'
    ]
    
    print("\n" + "="*80)
    print("TRAINING SCENARIO COMPARISON")
    print("="*80)
    
    for scenario in scenarios:
        analysis = analyze_training_requirements(scenario)
        
        print(f"\n📋 {scenario.upper().replace('_', ' ')}")
        print(f"Description: {analysis['description']}")
        print(f"Trainable params: {analysis['parameters']['trainable']:,} ({analysis['parameters']['trainable_percentage']:.1f}%)")
        print(f"Estimated GPU memory: {analysis['memory_estimates_gb']['total_estimated']} GB")
        print(f"Learning rate: {analysis['training_config']['learning_rate']}")
        print(f"Epochs: {analysis['training_config']['epochs']}")
    
    print("\n" + "="*80)


def quick_start_guide():
    """Print quick start guide for different scenarios"""
    
    guide = """
🚀 DUAL VISION ENCODER - QUICK START GUIDE
==========================================

1. VISION FINE-TUNING (Improve visual understanding)
   python train.py --config configs/config_vision_finetuning.json
   
2. LANGUAGE ADAPTATION (Adapt to new domain)
   python train.py --config configs/config_language_adaptation.json
   
3. FUSION LEARNING (Learn optimal encoder combination)
   python train.py --config configs/config_fusion_learning.json
   
4. PROGRESSIVE TRAINING (Curriculum learning approach)
   python train.py --config configs/config_progressive_training.json
   
5. MEDICAL ADAPTATION (Specialized for medical images)
   python train.py --config configs/config_medical_adaptation.json
   
6. EFFICIENT FINE-TUNING (Minimal parameters, fast training)
   python train.py --config configs/config_efficient_finetuning.json

📊 COMMAND LINE SHORTCUTS:
- python train.py --config config.json --training_mode encoders_only
- python train.py --config config.json --training_mode llm_only  
- python train.py --config config.json --training_mode fusion_only

🔧 CUSTOM FREEZING:
- python train.py --config config.json --freeze_llm --freeze_dinov2
- python train.py --config config.json --freeze_qwen_vision --freeze_fusion

📈 MONITORING:
- Use --use_wandb for experiment tracking
- Check outputs/ for checkpoints and logs
- Use inference.py --compare_encoders for analysis

💡 TIPS:
- Start with fusion_learning to understand encoder behavior
- Use progressive_training for best results
- Try efficient_finetuning for quick experiments
- Use medical_adaptation for domain-specific tasks
"""
    
    print(guide)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training scenario utilities")
    parser.add_argument('--create_configs', action='store_true', help='Create all scenario configs')
    parser.add_argument('--analyze', type=str, help='Analyze specific scenario requirements')
    parser.add_argument('--compare', action='store_true', help='Compare all scenarios')
    parser.add_argument('--guide', action='store_true', help='Show quick start guide')
    
    args = parser.parse_args()
    
    if args.create_configs:
        create_all_scenario_configs()
    
    elif args.analyze:
        analysis = analyze_training_requirements(args.analyze)
        print(json.dumps(analysis, indent=2))
    
    elif args.compare:
        print_scenario_comparison()
    
    elif args.guide:
        quick_start_guide()
    
    else:
        print("Use --help to see available options")
        quick_start_guide()
