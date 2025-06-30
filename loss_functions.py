"""
loss_functions.py - Comprehensive loss functions for Dual Vision Encoder training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoTokenizer


class DualVisionLoss(nn.Module):
    """
    Comprehensive loss function for Dual Vision Encoder training
    
    Components:
    1. Language Modeling Loss - Primary loss for text generation
    2. Fusion Diversity Loss - Encourages balanced encoder usage
    3. Contrastive Loss - Aligns vision and text representations
    4. Consistency Loss - Similar images should have similar fusion weights
    5. Regularization terms
    """
    
    def __init__(
        self,
        tokenizer,
        language_loss_weight: float = 1.0,
        diversity_loss_weight: float = 0.1,
        contrastive_loss_weight: float = 0.05,
        consistency_loss_weight: float = 0.01,
        temperature: float = 0.07,
        ignore_index: int = -100
    ):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.language_loss_weight = language_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.temperature = temperature
        self.ignore_index = ignore_index
        
        # Loss functions
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(
        self, 
        model_outputs: Dict[str, Any], 
        batch: Dict[str, Any],
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute comprehensive loss
        
        Args:
            model_outputs: Outputs from DualVisionEncoder.forward()
            batch: Batch data containing ground truth
            return_components: Whether to return individual loss components
        """
        device = next(iter(model_outputs.values())).device if model_outputs else torch.device('cpu')
        
        loss_components = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # 1. Language Modeling Loss (Primary)
        if 'logits' in model_outputs and 'target_ids' in batch:
            lang_loss = self._compute_language_loss(model_outputs, batch)
            loss_components['language_loss'] = lang_loss
            total_loss = total_loss + self.language_loss_weight * lang_loss
        
        # Vision-specific losses (only when images are present)
        if model_outputs.get('has_images', False):
            
            # 2. Fusion Diversity Loss
            if 'alpha' in model_outputs:
                diversity_loss = self._compute_diversity_loss(model_outputs['alpha'])
                loss_components['diversity_loss'] = diversity_loss
                total_loss = total_loss + self.diversity_loss_weight * diversity_loss
            
            # 3. Contrastive Loss (align vision and text)
            if 'fused_tokens' in model_outputs and 'text_embeddings' in batch:
                contrastive_loss = self._compute_contrastive_loss(model_outputs, batch)
                loss_components['contrastive_loss'] = contrastive_loss
                total_loss = total_loss + self.contrastive_loss_weight * contrastive_loss
            
            # 4. Consistency Loss
            if self.consistency_loss_weight > 0:
                consistency_loss = self._compute_consistency_loss(model_outputs)
                loss_components['consistency_loss'] = consistency_loss
                total_loss = total_loss + self.consistency_loss_weight * consistency_loss
        
        if return_components:
            return total_loss, loss_components
        return total_loss
    
    def _compute_language_loss(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Compute language modeling loss using cross-entropy
        """
        logits = model_outputs['logits']  # [batch_size, seq_len, vocab_size]
        target_ids = batch['target_ids']  # [batch_size, seq_len]
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # Flatten for cross-entropy computation
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        loss = self.cross_entropy(flat_logits, flat_labels)
        return loss
    
    def _compute_diversity_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute fusion diversity loss to encourage balanced encoder usage
        
        Uses entropy to encourage α ≈ 0.5 (balanced fusion)
        """
        epsilon = 1e-8
        
        # Entropy of fusion weights (higher entropy = more balanced)
        entropy = -(alpha * torch.log(alpha + epsilon) + 
                   (1 - alpha) * torch.log(1 - alpha + epsilon))
        
        # We want to maximize entropy, so minimize negative entropy
        diversity_loss = -torch.mean(entropy)
        return diversity_loss
    
    def _compute_contrastive_loss(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        """
        Compute contrastive loss to align vision and text representations
        """
        # Get vision features
        vision_features = model_outputs['fused_tokens']  # [batch_size, seq_len, dim]
        vision_pooled = vision_features.mean(dim=1)  # [batch_size, dim]
        
        # Get text features
        text_features = batch['text_embeddings']  # [batch_size, dim]
        
        # Normalize features
        vision_norm = F.normalize(vision_pooled, p=2, dim=1)
        text_norm = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(vision_norm, text_norm.T) / self.temperature
        
        # Create positive pairs (diagonal elements)
        batch_size = similarity.shape[0]
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Contrastive loss (InfoNCE)
        loss_v2t = self.cross_entropy(similarity, labels)
        loss_t2v = self.cross_entropy(similarity.T, labels)
        
        contrastive_loss = (loss_v2t + loss_t2v) / 2
        return contrastive_loss
    
    def _compute_consistency_loss(self, model_outputs: Dict) -> torch.Tensor:
        """
        Compute consistency loss: similar images should have similar fusion weights
        """
        if 'intermediates' not in model_outputs:
            return torch.tensor(0.0)
        
        alpha = model_outputs['alpha']  # [batch_size, 1]
        
        # Simple consistency: minimize variance in fusion weights
        consistency_loss = torch.var(alpha)
        return consistency_loss


class TaskSpecificLoss(nn.Module):
    """
    Task-specific loss functions for different applications
    """
    
    def __init__(self, task_type: str, **kwargs):
        super().__init__()
        self.task_type = task_type
        
        if task_type == 'vqa':
            self.loss_fn = self._vqa_loss
        elif task_type == 'captioning':
            self.loss_fn = self._captioning_loss
        elif task_type == 'medical_vqa':
            self.loss_fn = self._medical_vqa_loss
        elif task_type == 'classification':
            self.loss_fn = self._classification_loss
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        return self.loss_fn(model_outputs, batch)
    
    def _vqa_loss(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        """VQA-specific loss with answer accuracy"""
        # Standard language modeling loss
        base_loss = F.cross_entropy(
            model_outputs['logits'].view(-1, model_outputs['logits'].size(-1)),
            batch['target_ids'].view(-1),
            ignore_index=-100
        )
        
        # Could add answer-specific weighting here
        return base_loss
    
    def _captioning_loss(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        """Captioning-specific loss focusing on descriptive quality"""
        # Standard language modeling loss
        base_loss = F.cross_entropy(
            model_outputs['logits'].view(-1, model_outputs['logits'].size(-1)),
            batch['target_ids'].view(-1),
            ignore_index=-100
        )
        
        # Could add diversity penalty for captioning
        return base_loss
    
    def _medical_vqa_loss(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        """Medical VQA loss with clinical accuracy weighting"""
        # Standard language modeling loss
        base_loss = F.cross_entropy(
            model_outputs['logits'].view(-1, model_outputs['logits'].size(-1)),
            batch['target_ids'].view(-1),
            ignore_index=-100
        )
        
        # Could add medical-specific penalties
        return base_loss
    
    def _classification_loss(self, model_outputs: Dict, batch: Dict) -> torch.Tensor:
        """Classification loss for image classification tasks"""
        logits = model_outputs['classification_logits']
        labels = batch['labels']
        
        return F.cross_entropy(logits, labels)


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        diversity_scheduler: str = 'cosine',
        warmup_steps: int = 1000
    ):
        super().__init__()
        self.base_loss = base_loss
        self.diversity_scheduler = diversity_scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0
    
    def forward(self, model_outputs: Dict, batch: Dict, step: Optional[int] = None) -> torch.Tensor:
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
        
        # Adjust diversity loss weight based on training progress
        if self.diversity_scheduler == 'cosine':
            # Start high, decrease over time
            progress = min(self.step_count / self.warmup_steps, 1.0)
            diversity_weight = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        elif self.diversity_scheduler == 'linear':
            # Linearly decrease
            progress = min(self.step_count / self.warmup_steps, 1.0)
            diversity_weight = 1.0 - progress
        else:
            diversity_weight = 1.0
        
        # Update base loss diversity weight
        if hasattr(self.base_loss, 'diversity_loss_weight'):
            original_weight = self.base_loss.diversity_loss_weight
            self.base_loss.diversity_loss_weight = original_weight * diversity_weight
        
        loss = self.base_loss(model_outputs, batch)
        
        # Restore original weight
        if hasattr(self.base_loss, 'diversity_loss_weight'):
            self.base_loss.diversity_loss_weight = original_weight
        
        return loss


def prepare_batch_for_loss(batch: Dict, tokenizer, device: torch.device) -> Dict[str, Any]:
    """
    Prepare batch data for loss computation
    
    Args:
        batch: Raw batch from dataloader
        tokenizer: Tokenizer for text processing
        device: Target device
    
    Returns:
        Prepared batch with target_ids and other necessary fields
    """
    prepared_batch = {}
    
    # Extract target text from conversations
    target_texts = []
    for messages in batch['messages']:
        # Find assistant responses
        for message in messages:
            if message['role'] == 'assistant':
                if isinstance(message['content'], str):
                    target_texts.append(message['content'])
                else:
                    # Handle complex content
                    text_parts = [part['text'] for part in message['content'] if part['type'] == 'text']
                    target_texts.append(' '.join(text_parts))
                break
        else:
            target_texts.append("")  # Fallback
    
    # Tokenize target texts
    target_encoding = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    prepared_batch['target_ids'] = target_encoding['input_ids'].to(device)
    prepared_batch['target_attention_mask'] = target_encoding['attention_mask'].to(device)
    
    # Add other fields as needed
    prepared_batch.update(batch)
    
    return prepared_batch


# Example usage functions
def create_loss_function(
    tokenizer,
    task_type: str = 'general',
    loss_config: Optional[Dict] = None
) -> nn.Module:
    """
    Factory function to create appropriate loss function
    """
    if loss_config is None:
        loss_config = {
            'language_loss_weight': 1.0,
            'diversity_loss_weight': 0.1,
            'contrastive_loss_weight': 0.05,
            'consistency_loss_weight': 0.01
        }
    
    if task_type == 'general':
        return DualVisionLoss(tokenizer, **loss_config)
    else:
        # Task-specific loss
        base_loss = DualVisionLoss(tokenizer, **loss_config)
        task_loss = TaskSpecificLoss(task_type)
        
        # Combine losses (simplified)
        return base_loss


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Create loss function
    loss_fn = create_loss_function(tokenizer, task_type='general')
    
    print("Loss function created successfully!")
    print(f"Loss components: Language, Diversity, Contrastive, Consistency")