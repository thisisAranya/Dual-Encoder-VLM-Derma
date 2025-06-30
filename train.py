import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os
import json
from typing import Dict, List, Optional
import argparse
from datetime import datetime

from model import DualVisionEncoder
from dataset import DualVisionDataset
from utils import setup_logging, save_checkpoint, load_checkpoint
from loss_functions import DualVisionLoss, prepare_batch_for_loss, create_loss_function


class DualVisionTrainer:
    """Trainer class for the Dual Vision Encoder"""
    
    def __init__(
        self,
        model: DualVisionEncoder,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        
        # Apply freezing configuration
        self._setup_freezing()
        
        # Initialize loss function
        self.loss_function = create_loss_function(
            tokenizer=self.model.processor.tokenizer,
            task_type=config.get('task_type', 'general'),
            loss_config=config.get('loss_config', {
                'language_loss_weight': 1.0,
                'diversity_loss_weight': config.get('diversity_loss_weight', 0.1),
                'contrastive_loss_weight': 0.05,
                'consistency_loss_weight': 0.01
            })
        )
        
        # Setup optimizer with only trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup scheduler
        total_steps = len(train_dataloader) * config['num_epochs']
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config['min_lr']
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        if config['use_wandb']:
            wandb.init(
                project=config['project_name'],
                config=config,
                name=f"dual_vision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def _setup_freezing(self):
        """Setup parameter freezing based on configuration"""
        freeze_config = self.config.get('freeze_config', {})
        
        # Get freeze settings
        freeze_qwen_vision = freeze_config.get('freeze_qwen_vision', False)
        freeze_dinov2 = freeze_config.get('freeze_dinov2', True)  # Default: freeze DINOv2
        freeze_llm = freeze_config.get('freeze_llm', False)
        freeze_fusion_layers = freeze_config.get('freeze_fusion_layers', False)
        freeze_projections = freeze_config.get('freeze_projections', False)
        
        # Apply freezing
        if freeze_qwen_vision:
            self._freeze_component(self.model.qwen_vision_encoder, "Qwen Vision Encoder")
        
        if freeze_dinov2:
            self._freeze_component(self.model.dinov2_model, "DINOv2")
        
        if freeze_llm:
            self._freeze_component(self.model.language_model, "Language Model")
            self._freeze_component(self.model.lm_head, "LM Head")
        
        if freeze_fusion_layers:
            self._freeze_component(self.model.gating_network, "Gating Network")
            self._freeze_component(self.model.vision_to_lm, "Vision to LM projection")
        
        if freeze_projections:
            self._freeze_component(self.model.qwen_projection, "Qwen Projection")
            self._freeze_component(self.model.dinov2_projection, "DINOv2 Projection")
        
        # Print parameter statistics
        self._print_parameter_stats()
    
    def _freeze_component(self, component: nn.Module, name: str):
        """Freeze a model component"""
        for param in component.parameters():
            param.requires_grad = False
        print(f"✅ Frozen: {name}")
    
    def _unfreeze_component(self, component: nn.Module, name: str):
        """Unfreeze a model component"""
        for param in component.parameters():
            param.requires_grad = True
        print(f"🔓 Unfrozen: {name}")
    
    def _print_parameter_stats(self):
        """Print detailed parameter statistics"""
        components = {
            'Qwen Vision Encoder': self.model.qwen_vision_encoder,
            'DINOv2': self.model.dinov2_model,
            'Language Model': self.model.language_model,
            'LM Head': self.model.lm_head,
            'Qwen Projection': self.model.qwen_projection,
            'DINOv2 Projection': self.model.dinov2_projection,
            'Gating Network': self.model.gating_network,
            'Vision to LM': self.model.vision_to_lm
        }
        
        print("\n" + "="*60)
        print("PARAMETER FREEZE STATUS")
        print("="*60)
        
        total_params = 0
        trainable_params = 0
        
        for name, component in components.items():
            comp_total = sum(p.numel() for p in component.parameters())
            comp_trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            
            total_params += comp_total
            trainable_params += comp_trainable
            
            status = "🔓 TRAINABLE" if comp_trainable > 0 else "❄️  FROZEN"
            print(f"{name:20s}: {comp_total:>10,} total, {comp_trainable:>10,} trainable {status}")
        
        print("-" * 60)
        print(f"{'TOTAL':20s}: {total_params:>10,} total, {trainable_params:>10,} trainable")
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
        print("="*60 + "\n")
    
    def change_freeze_settings(self, new_freeze_config: Dict[str, bool]):
        """
        Dynamically change freeze settings during training
        
        Args:
            new_freeze_config: New freeze configuration
                - freeze_qwen_vision: bool
                - freeze_dinov2: bool  
                - freeze_llm: bool
                - freeze_fusion_layers: bool
                - freeze_projections: bool
        """
        print("Changing freeze settings...")
        
        # Update config
        self.config['freeze_config'].update(new_freeze_config)
        
        # First, unfreeze everything
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Then apply new freezing
        self._setup_freezing()
        
        # Update optimizer with new trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Reset scheduler
        total_steps = len(self.train_dataloader) * (self.config['num_epochs'] - self.current_epoch)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps, 1),  # Avoid division by zero
            eta_min=self.config['min_lr']
        )
        
        print("Freeze settings updated successfully!")
    
    def get_training_mode_preset(self, mode: str) -> Dict[str, bool]:
        """
        Get predefined freeze configurations for common training scenarios
        
        Args:
            mode: Training mode
                - 'full': Train everything
                - 'encoders_only': Train only vision encoders + fusion, freeze LLM
                - 'llm_only': Train only LLM, freeze vision encoders
                - 'fusion_only': Train only fusion layers, freeze encoders + LLM
                - 'qwen_only': Train only Qwen vision encoder + fusion
                - 'dinov2_only': Train only DINOv2 + fusion
        """
        presets = {
            'full': {
                'freeze_qwen_vision': False,
                'freeze_dinov2': False,
                'freeze_llm': False,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'encoders_only': {
                'freeze_qwen_vision': False,
                'freeze_dinov2': False,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'llm_only': {
                'freeze_qwen_vision': True,
                'freeze_dinov2': True,
                'freeze_llm': False,
                'freeze_fusion_layers': True,
                'freeze_projections': True
            },
            'fusion_only': {
                'freeze_qwen_vision': True,
                'freeze_dinov2': True,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'qwen_only': {
                'freeze_qwen_vision': False,
                'freeze_dinov2': True,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            },
            'dinov2_only': {
                'freeze_qwen_vision': True,
                'freeze_dinov2': False,
                'freeze_llm': True,
                'freeze_fusion_layers': False,
                'freeze_projections': False
            }
        }
        
        if mode not in presets:
            raise ValueError(f"Unknown training mode: {mode}. Available: {list(presets.keys())}")
        
        return presets[mode]
    
    def switch_training_mode(self, mode: str):
        """Switch to a predefined training mode"""
        freeze_config = self.get_training_mode_preset(mode)
        print(f"Switching to training mode: {mode}")
        self.change_freeze_settings(freeze_config)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_fusion_weights = {'qwen': 0, 'dinov2': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass
            try:
                # Prepare batch for loss computation
                prepared_batch = prepare_batch_for_loss(
                    batch, 
                    self.model.processor.tokenizer, 
                    self.device
                )
                
                # Forward pass in training mode
                outputs = self.model.forward(
                    messages=batch['messages'],
                    compute_loss=True,
                    target_ids=prepared_batch['target_ids'],
                    return_intermediates=True
                )
                
                # Calculate loss
                loss = self.loss_function(outputs, prepared_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Logging
                total_loss += loss.item()
                num_batches += 1
                
                # Track fusion weights
                if outputs['has_images'] and 'alpha' in outputs:
                    alpha_value = outputs['alpha'].mean().item()
                    total_fusion_weights['qwen'] += alpha_value
                    total_fusion_weights['dinov2'] += (1 - alpha_value)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
                })
                
                self.global_step += 1
                
                # Log to wandb
                if self.config['use_wandb'] and self.global_step % self.config['log_steps'] == 0:
                    log_dict = {
                        'train/loss': loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': self.current_epoch,
                        'train/step': self.global_step
                    }
                    
                    # Add fusion weights if available
                    if outputs['has_images'] and 'alpha' in outputs:
                        alpha_value = outputs['alpha'].mean().item()
                        log_dict.update({
                            'train/fusion_weight_qwen': alpha_value,
                            'train/fusion_weight_dinov2': 1 - alpha_value
                        })
                    
                    wandb.log(log_dict)
                
            except Exception as e:
                print(f"Error in training step: {e}")
                continue
        
        # Calculate averages
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_fusion_weights = {
            'qwen': total_fusion_weights['qwen'] / num_batches if num_batches > 0 else 0,
            'dinov2': total_fusion_weights['dinov2'] / num_batches if num_batches > 0 else 0
        }
        
        return {
            'loss': avg_loss,
            'fusion_weights': avg_fusion_weights
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_fusion_weights = {'qwen': 0, 'dinov2': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                try:
                    # Prepare batch for loss computation
                    prepared_batch = prepare_batch_for_loss(
                        batch, 
                        self.model.processor.tokenizer, 
                        self.device
                    )
                    
                    # Forward pass in training mode for validation
                    outputs = self.model.forward(
                        messages=batch['messages'],
                        compute_loss=True,
                        target_ids=prepared_batch['target_ids'],
                        return_intermediates=True
                    )
                    
                    # Calculate loss
                    loss = self.loss_function(outputs, prepared_batch)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Track fusion weights
                    if outputs['has_images'] and 'alpha' in outputs:
                        alpha_value = outputs['alpha'].mean().item()
                        total_fusion_weights['qwen'] += alpha_value
                        total_fusion_weights['dinov2'] += (1 - alpha_value)
                        
                except Exception as e:
                    print(f"Error in validation step: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_fusion_weights = {
            'qwen': total_fusion_weights['qwen'] / num_batches if num_batches > 0 else 0,
            'dinov2': total_fusion_weights['dinov2'] / num_batches if num_batches > 0 else 0
        }
        
        return {
            'loss': avg_loss,
            'fusion_weights': avg_fusion_weights
        }
    
    def get_loss_components(self) -> Dict[str, float]:
        """Get detailed loss component breakdown for logging"""
        if hasattr(self.loss_function, 'last_loss_components'):
            return self.loss_function.last_loss_components
        return {} torch.tensor(0.0, device=self.device)
        
        intermediates = outputs['intermediates']
        
        # Get projected features
        qwen_proj = intermediates['tokens1_proj']  # [batch_size, N1, D]
        dinov2_proj = intermediates['tokens2_proj']  # [batch_size, N2, D]
        fused_tokens = intermediates['fused_tokens']  # [batch_size, N, D]
        
        # Alignment loss: fused features should be meaningful combinations
        # One approach: minimize distance between fused features and both projected features
        
        # Mean pool for comparison
        qwen_pooled = qwen_proj.mean(dim=1)  # [batch_size, D]
        dinov2_pooled = dinov2_proj.mean(dim=1)  # [batch_size, D]
        fused_pooled = fused_tokens.mean(dim=1)  # [batch_size, D]
        
        # Fused features should be "between" the two encoder features
        qwen_distance = F.mse_loss(fused_pooled, qwen_pooled, reduction='mean')
        dinov2_distance = F.mse_loss(fused_pooled, dinov2_pooled, reduction='mean')
        
        # Encourage fused features to be a meaningful combination
        alignment_loss = torch.min(qwen_distance, dinov2_distance)
        
        return alignment_loss
    
    def train(self):
        """Main training loop with support for curriculum training"""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        # Check for curriculum training
        curriculum = self.config.get('curriculum_training', {})
        if curriculum.get('enabled', False):
            self._train_with_curriculum(curriculum)
        else:
            self._train_standard()
        
        print("Training completed!")
    
    def _train_standard(self):
        """Standard training loop"""
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Print epoch summary
            self._print_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Log to wandb
            if self.config['use_wandb']:
                self._log_to_wandb(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            self._save_checkpoint_if_needed(epoch, val_metrics['loss'])
    
    def _train_with_curriculum(self, curriculum: Dict):
        """Training with curriculum learning (changing freeze settings over time)"""
        print("Starting curriculum training...")
        
        stages = curriculum['stages']
        total_epochs = self.config['num_epochs']
        
        current_stage = 0
        stage_start_epoch = 0
        
        for epoch in range(total_epochs):
            self.current_epoch = epoch
            
            # Check if we need to switch to next stage
            if current_stage < len(stages) - 1:
                next_stage = stages[current_stage + 1]
                stage_epochs = stages[current_stage].get('epochs', total_epochs // len(stages))
                
                if epoch >= stage_start_epoch + stage_epochs:
                    current_stage += 1
                    stage_start_epoch = epoch
                    
                    # Switch training mode
                    new_mode = stages[current_stage]['mode']
                    print(f"\n🔄 Switching to curriculum stage {current_stage + 1}: {new_mode}")
                    self.switch_training_mode(new_mode)
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate  
            val_metrics = self.validate()
            
            # Print epoch summary with stage info
            stage_info = f" [Stage {current_stage + 1}/{len(stages)}: {stages[current_stage]['mode']}]"
            self._print_epoch_summary(epoch, train_metrics, val_metrics, stage_info)
            
            # Log to wandb
            if self.config['use_wandb']:
                wandb_metrics = {
                    'curriculum/stage': current_stage,
                    'curriculum/stage_name': stages[current_stage]['mode']
                }
                self._log_to_wandb(train_metrics, val_metrics, epoch, wandb_metrics)
            
            # Save checkpoint
            self._save_checkpoint_if_needed(epoch, val_metrics['loss'])
    
    def _print_epoch_summary(self, epoch: int, train_metrics: Dict, val_metrics: Dict, extra_info: str = ""):
        """Print epoch summary"""
        print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}{extra_info}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Train Fusion - Qwen: {train_metrics['fusion_weights']['qwen']:.3f}, "
              f"DINOv2: {train_metrics['fusion_weights']['dinov2']:.3f}")
        print(f"  Val Fusion - Qwen: {val_metrics['fusion_weights']['qwen']:.3f}, "
              f"DINOv2: {val_metrics['fusion_weights']['dinov2']:.3f}")
    
    def _log_to_wandb(self, train_metrics: Dict, val_metrics: Dict, epoch: int, extra_metrics: Dict = None):
        """Log metrics to wandb"""
        log_dict = {
            'val/loss': val_metrics['loss'],
            'val/fusion_weight_qwen': val_metrics['fusion_weights']['qwen'],
            'val/fusion_weight_dinov2': val_metrics['fusion_weights']['dinov2'],
            'epoch': epoch
        }
        
        if extra_metrics:
            log_dict.update(extra_metrics)
        
        wandb.log(log_dict)
    
    def _save_checkpoint_if_needed(self, epoch: int, val_loss: float):
        """Save checkpoint if needed"""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
        
        if epoch % self.config['save_steps'] == 0 or is_best:
            # Include freeze config in checkpoint
            additional_info = {
                'freeze_config': self.config.get('freeze_config', {}),
                'training_mode': self.config.get('training_mode', 'full')
            }
            
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                loss=val_loss,
                save_dir=self.config['output_dir'],
                is_best=is_best,
                additional_info=additional_info
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--training_mode', type=str, 
                       choices=['full', 'encoders_only', 'llm_only', 'fusion_only', 'qwen_only', 'dinov2_only'],
                       help='Training mode preset (overrides config)')
    parser.add_argument('--freeze_qwen_vision', action='store_true', help='Freeze Qwen vision encoder')
    parser.add_argument('--freeze_dinov2', action='store_true', help='Freeze DINOv2')
    parser.add_argument('--freeze_llm', action='store_true', help='Freeze language model')
    parser.add_argument('--freeze_fusion', action='store_true', help='Freeze fusion layers')
    parser.add_argument('--freeze_projections', action='store_true', help='Freeze projection layers')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override freeze config with command line arguments
    if args.training_mode:
        print(f"Using training mode preset: {args.training_mode}")
        # This will be handled by the trainer
        config['training_mode'] = args.training_mode
    else:
        # Use individual freeze flags
        freeze_config = config.get('freeze_config', {})
        if args.freeze_qwen_vision:
            freeze_config['freeze_qwen_vision'] = True
        if args.freeze_dinov2:
            freeze_config['freeze_dinov2'] = True
        if args.freeze_llm:
            freeze_config['freeze_llm'] = True
        if args.freeze_fusion:
            freeze_config['freeze_fusion_layers'] = True
        if args.freeze_projections:
            freeze_config['freeze_projections'] = True
        
        config['freeze_config'] = freeze_config
    
    # Setup logging
    setup_logging(config['output_dir'])
    
    # Create model
    model = DualVisionEncoder(
        qwen_model_name=config['qwen_model_name'],
        dinov2_model_name=config['dinov2_model_name'],
        common_dim=config['common_dim'],
        dropout=config['dropout']
    )
    
    # Create datasets
    train_dataset = DualVisionDataset(
        data_path=config['train_data_path'],
        split='train'
    )
    val_dataset = DualVisionDataset(
        data_path=config['val_data_path'],
        split='val'
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=val_dataset.collate_fn
    )
    
    # Create trainer
    trainer = DualVisionTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # Apply training mode if specified
    if args.training_mode:
        trainer.switch_training_mode(args.training_mode)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, trainer.optimizer, trainer.scheduler)
        
        # Restore freeze config from checkpoint if available
        if 'freeze_config' in checkpoint:
            print("Restoring freeze configuration from checkpoint...")
            trainer.change_freeze_settings(checkpoint['freeze_config'])
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()