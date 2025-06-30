"""
example_loss_usage.py - Example demonstrating how to use the loss functions
"""

import torch
from transformers import AutoTokenizer
from loss_functions import DualVisionLoss, create_loss_function, prepare_batch_for_loss


def example_basic_loss():
    """Example of basic loss function usage"""
    print("=== BASIC LOSS FUNCTION EXAMPLE ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Create loss function
    loss_fn = DualVisionLoss(
        tokenizer=tokenizer,
        language_loss_weight=1.0,
        diversity_loss_weight=0.1,
        contrastive_loss_weight=0.05,
        consistency_loss_weight=0.01
    )
    
    # Mock model outputs
    batch_size, seq_len, vocab_size = 2, 20, 32000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_outputs = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'has_images': True,
        'alpha': torch.sigmoid(torch.randn(batch_size, 1)),  # Fusion weights
        'fused_tokens': torch.randn(batch_size, 10, 512),
        'intermediates': {
            'alpha': torch.sigmoid(torch.randn(batch_size, 1))
        }
    }
    
    # Mock batch data
    batch = {
        'target_ids': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'text_embeddings': torch.randn(batch_size, 512)
    }
    
    # Move to device
    for key, value in model_outputs.items():
        if isinstance(value, torch.Tensor):
            model_outputs[key] = value.to(device)
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    loss_fn = loss_fn.to(device)
    
    # Compute loss
    total_loss, loss_components = loss_fn(model_outputs, batch, return_components=True)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print("Loss Components:")
    for component, value in loss_components.items():
        print(f"  {component}: {value.item():.4f}")
    
    return total_loss, loss_components


def example_task_specific_loss():
    """Example of task-specific loss functions"""
    print("\n=== TASK-SPECIFIC LOSS FUNCTIONS ===")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Different task types
    tasks = ['general', 'vqa', 'captioning', 'medical_vqa']
    
    for task in tasks:
        print(f"\n{task.upper()} Loss Function:")
        
        try:
            loss_fn = create_loss_function(
                tokenizer=tokenizer,
                task_type=task,
                loss_config={
                    'language_loss_weight': 1.0,
                    'diversity_loss_weight': 0.1 if task != 'medical_vqa' else 0.15,
                    'contrastive_loss_weight': 0.05,
                    'consistency_loss_weight': 0.01
                }
            )
            print(f"  ✅ Successfully created {task} loss function")
            
        except Exception as e:
            print(f"  ❌ Error creating {task} loss: {e}")


def example_loss_configuration():
    """Example of different loss configurations"""
    print("\n=== LOSS CONFIGURATION EXAMPLES ===")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    configs = {
        'balanced': {
            'language_loss_weight': 1.0,
            'diversity_loss_weight': 0.1,
            'contrastive_loss_weight': 0.05,
            'consistency_loss_weight': 0.01
        },
        'diversity_focused': {
            'language_loss_weight': 1.0,
            'diversity_loss_weight': 0.3,  # Higher diversity weight
            'contrastive_loss_weight': 0.02,
            'consistency_loss_weight': 0.05
        },
        'language_focused': {
            'language_loss_weight': 1.0,
            'diversity_loss_weight': 0.05,  # Lower diversity weight
            'contrastive_loss_weight': 0.01,
            'consistency_loss_weight': 0.0   # No consistency
        },
        'contrastive_focused': {
            'language_loss_weight': 1.0,
            'diversity_loss_weight': 0.1,
            'contrastive_loss_weight': 0.2,  # Higher contrastive weight
            'consistency_loss_weight': 0.01
        }
    }
    
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()} Configuration:")
        for component, weight in config.items():
            print(f"  {component}: {weight}")
        
        # Create loss function with this config
        loss_fn = create_loss_function(
            tokenizer=tokenizer,
            task_type='general',
            loss_config=config
        )
        print(f"  ✅ Loss function created successfully")


def example_loss_analysis():
    """Example of analyzing loss behavior"""
    print("\n=== LOSS BEHAVIOR ANALYSIS ===")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    loss_fn = DualVisionLoss(tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)
    
    # Analyze how diversity loss changes with different alpha values
    print("\nDiversity Loss vs Fusion Weight (α):")
    print("α=0.1 (DINOv2 dominant):", end=" ")
    alpha = torch.tensor([[0.1]], device=device)
    diversity_loss = loss_fn._compute_diversity_loss(alpha)
    print(f"Diversity Loss = {diversity_loss.item():.4f}")
    
    print("α=0.5 (Balanced):", end=" ")
    alpha = torch.tensor([[0.5]], device=device)
    diversity_loss = loss_fn._compute_diversity_loss(alpha)
    print(f"Diversity Loss = {diversity_loss.item():.4f}")
    
    print("α=0.9 (Qwen dominant):", end=" ")
    alpha = torch.tensor([[0.9]], device=device)
    diversity_loss = loss_fn._compute_diversity_loss(alpha)
    print(f"Diversity Loss = {diversity_loss.item():.4f}")
    
    # The balanced case (α=0.5) should have the lowest diversity loss


def example_batch_preparation():
    """Example of preparing batch data for loss computation"""
    print("\n=== BATCH PREPARATION EXAMPLE ===")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Mock batch data (as would come from DataLoader)
    batch = {
        'messages': [
            [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': 'path/to/image1.jpg'},
                        {'type': 'text', 'text': 'What do you see?'}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': 'I see a beautiful landscape with mountains.'
                }
            ],
            [
                {
                    'role': 'user', 
                    'content': [
                        {'type': 'image', 'image': 'path/to/image2.jpg'},
                        {'type': 'text', 'text': 'Describe this image.'}
                    ]
                },
                {
                    'role': 'assistant',
                    'content': 'This image shows a cat sitting on a table.'
                }
            ]
        ],
        'images': ['mock_image1', 'mock_image2'],
        'image_paths': ['path/to/image1.jpg', 'path/to/image2.jpg']
    }
    
    # Prepare batch for loss computation
    prepared_batch = prepare_batch_for_loss(batch, tokenizer, device)
    
    print("Original batch keys:", list(batch.keys()))
    print("Prepared batch keys:", list(prepared_batch.keys()))
    print("Target IDs shape:", prepared_batch['target_ids'].shape)
    print("Target texts extracted:")
    for i, messages in enumerate(batch['messages']):
        for message in messages:
            if message['role'] == 'assistant':
                print(f"  Sample {i+1}: '{message['content']}'")


def main():
    """Run all examples"""
    print("🔥 DUAL VISION ENCODER - LOSS FUNCTION EXAMPLES")
    print("=" * 60)
    
    try:
        example_basic_loss()
        example_task_specific_loss()
        example_loss_configuration()
        example_loss_analysis()
        example_batch_preparation()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()