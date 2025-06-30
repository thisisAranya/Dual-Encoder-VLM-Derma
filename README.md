# Dual Vision Encoder

A multimodal architecture that combines Qwen 2.5-VL and DINOv2 vision encoders with learnable fusion for enhanced visual understanding.

## Architecture Overview

```
Input Image
     │
     ├─────────────────────┬
     │                     │
┌────▼────┐           ┌────▼────┐
│Qwen-VL  │           │DINOv2   │
│Encoder  │           │Encoder  │
└────┬────┘           └────┬────┘
     │                     │
┌────▼────┐           ┌────▼────┐
│Linear   │           │Linear   │
│Proj     │           │Proj     │
└────┬────┘           └────┬────┘
     │                     │
     └─────────┬───────────┘
               │
        ┌──────▼──────┐
        │Global Pool  │
        │& Gating     │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │Weighted     │
        │Fusion       │
        └──────┬──────┘
               │
        ┌──────▼──────┐
        │Language     │
        │Model (LLM)  │
        └─────────────┘
```

## Key Features

- **Dual Vision Encoding**: Combines Qwen 2.5-VL's multimodal pre-trained encoder with DINOv2's self-supervised features
- **Learnable Fusion**: Gating network learns optimal weighting between encoders for different image types
- **Flexible Architecture**: Easy to modify fusion strategies and add new encoders
- **Production Ready**: Includes training, inference, and analysis tools

## Installation

```bash
# Clone repository
git clone <repository-url>
cd dual-vision-encoder

# Install dependencies
pip install -r requirements.txt

# Install flash attention (optional, for better performance)
pip install flash-attn
```

## Quick Start

### 1. Basic Inference

```python
from model import DualVisionEncoder

# Create model
model = DualVisionEncoder()

# Prepare message (same format as Qwen)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "What do you see?"}
    ]
}]

# Generate response
response = model.chat(messages)
print(response)
```

### 2. Training Scenarios

The framework provides predefined training scenarios for common use cases:

```bash
# Create all scenario configurations
python training_examples.py --create_configs

# Quick training examples
python train.py --config configs/config_vision_finetuning.json     # Improve vision
python train.py --config configs/config_language_adaptation.json   # Adapt language
python train.py --config configs/config_fusion_learning.json       # Learn fusion
python train.py --config configs/config_progressive_training.json  # Curriculum learning
```

### 3. Analyze Encoder Fusion

```python
# Analyze which encoder the model prefers
weights = model.analyze_fusion_weights(messages)
print(f"Qwen-VL weight: {weights['qwen_weight']:.3f}")
print(f"DINOv2 weight: {weights['dinov2_weight']:.3f}")
print(f"Dominant: {weights['dominant_encoder']}")
```

### 4. Command Line Inference

```bash
python inference.py \
    --image path/to/image.jpg \
    --question "Describe this image" \
    --compare_encoders
```

## Training

### 1. Prepare Data

Create a JSON file with your training data:

```json
[
  {
    "image_path": "images/image1.jpg",
    "conversations": [
      {"role": "user", "content": "What's in this image?"},
      {"role": "assistant", "content": "This image shows..."}
    ]
  }
]
```

### 2. Configure Training

Copy and modify the configuration template:

```bash
cp config.json my_config.json
# Edit my_config.json with your settings
```

### 3. Training with Flexible Freezing

The framework supports flexible component freezing for different training scenarios:

#### **Training Mode Presets:**

```bash
# Train everything (default)
python train.py --config my_config.json --training_mode full

# Train only vision encoders + fusion, freeze LLM
python train.py --config my_config.json --training_mode encoders_only

# Train only LLM, freeze vision encoders
python train.py --config my_config.json --training_mode llm_only

# Train only fusion layers, freeze everything else
python train.py --config my_config.json --training_mode fusion_only

# Train only Qwen vision encoder + fusion
python train.py --config my_config.json --training_mode qwen_only

# Train only DINOv2 + fusion
python train.py --config my_config.json --training_mode dinov2_only
```

#### **Custom Freezing:**

```bash
# Freeze specific components
python train.py --config my_config.json \
    --freeze_llm \
    --freeze_dinov2

# Individual component control
python train.py --config my_config.json \
    --freeze_qwen_vision \
    --freeze_fusion \
    --freeze_projections
```

#### **Configuration-based Freezing:**

```json
{
  "freeze_config": {
    "freeze_qwen_vision": false,
    "freeze_dinov2": true,
    "freeze_llm": false,
    "freeze_fusion_layers": false,
    "freeze_projections": false
  }
}
```

#### **Curriculum Training:**

Train in progressive stages with different freeze settings:

```json
{
  "curriculum_training": {
    "enabled": true,
    "stages": [
      {
        "mode": "fusion_only",
        "epochs": 3,
        "description": "First train only fusion layers"
      },
      {
        "mode": "encoders_only", 
        "epochs": 4,
        "description": "Then train encoders with fusion"
      },
      {
        "mode": "full",
        "epochs": 3,
        "description": "Finally train everything"
      }
    ]
  }
}
```

### 4. Resume Training

```bash
python train.py --config my_config.json --resume outputs/checkpoint_latest.pth
```

The freeze configuration is automatically restored from checkpoints.

## Dataset Formats

The framework supports multiple dataset formats:

### VQA Format
```json
{
  "image_path": "path/to/image.jpg",
  "conversations": [
    {"role": "user", "content": "What color is the car?"},
    {"role": "assistant", "content": "The car is red."}
  ]
}
```

### Captioning Format
```json
{
  "image_path": "path/to/image.jpg", 
  "conversations": [
    {"role": "assistant", "content": "A beautiful sunset over the mountains."}
  ]
}
```

### Medical VQA Format
```json
{
  "image_path": "path/to/medical_image.jpg",
  "conversations": [
    {"role": "user", "content": "What abnormalities do you see?"},
    {"role": "assistant", "content": "The image shows..."}
  ]
}
```

## Model Components

### DualVisionEncoder
- Main model class combining both encoders
- Handles message processing and generation
- Provides analysis tools for fusion weights

### Training Components
- `DualVisionTrainer`: Training loop management
- `DualVisionDataset`: Data loading and preprocessing
- Various specialized dataset classes (VQA, Medical, etc.)

### Utilities
- Checkpoint management
- Visualization tools
- Performance analysis
- Configuration validation

## Configuration Options

### Model Configuration
- `qwen_model_name`: Qwen model variant to use
- `dinov2_model_name`: DINOv2 model variant  
- `common_dim`: Dimension for fusion layer
- `dropout`: Dropout rate
- `use_flash_attention`: Enable flash attention

### Freeze Configuration
- `freeze_qwen_vision`: Freeze Qwen vision encoder
- `freeze_dinov2`: Freeze DINOv2 model (default: true)
- `freeze_llm`: Freeze language model components
- `freeze_fusion_layers`: Freeze gating network and vision-to-LM projection
- `freeze_projections`: Freeze linear projection layers

### Training Configuration
- `learning_rate`: Initial learning rate
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `diversity_loss_weight`: Weight for fusion diversity loss

### Curriculum Training Configuration
- `enabled`: Enable curriculum training
- `stages`: List of training stages with different freeze settings
  - `mode`: Training mode for this stage
  - `epochs`: Number of epochs for this stage
  - `description`: Human-readable description

### Data Configuration
- `train_data_path`: Path to training data
- `val_data_path`: Path to validation data
- `image_root`: Root directory for images

## Performance Analysis

### Fusion Weight Analysis
```python
from utils import analyze_model_outputs

# Analyze on test data
results = analyze_model_outputs(model, test_messages, save_dir="analysis")

# View statistics
print(f"Average Qwen weight: {results['statistics']['avg_qwen_weight']:.3f}")
print(f"Average DINOv2 weight: {results['statistics']['avg_dinov2_weight']:.3f}")
```

### Benchmarking
```bash
python inference.py \
    --image test_image.jpg \
    --question "Describe this image" \
    --benchmark
```

## Advanced Usage

### Training Scenarios

#### **Scenario 1: Fine-tune Vision Understanding**
When you want to improve visual understanding while keeping language capabilities frozen:

```bash
# Train only vision encoders and fusion
python train.py --config config.json --training_mode encoders_only
```

#### **Scenario 2: Adapt to New Language Domain** 
When you want to adapt the language model to a new domain while keeping vision features frozen:

```bash
# Train only language model
python train.py --config config.json --training_mode llm_only
```

#### **Scenario 3: Learn Optimal Fusion Strategy**
When you want to learn how to optimally combine the two encoders:

```bash
# Train only fusion components
python train.py --config config.json --training_mode fusion_only
```

#### **Scenario 4: Progressive Training**
Use curriculum training to train components progressively:

```json
{
  "curriculum_training": {
    "enabled": true,
    "stages": [
      {"mode": "fusion_only", "epochs": 5},
      {"mode": "encoders_only", "epochs": 10}, 
      {"mode": "full", "epochs": 5}
    ]
  }
}
```

### Dynamic Freeze Control

```python
from train import DualVisionTrainer

# Create trainer
trainer = DualVisionTrainer(model, train_loader, val_loader, config)

# Switch training modes dynamically
trainer.switch_training_mode('encoders_only')

# Or set custom freeze settings
trainer.change_freeze_settings({
    'freeze_qwen_vision': True,
    'freeze_llm': False,
    'freeze_fusion_layers': False
})
```

### Parameter Monitoring

The training script provides detailed parameter statistics:

```
============================================================
PARAMETER FREEZE STATUS
============================================================
Qwen Vision Encoder :  1,234,567 total,  1,234,567 trainable 🔓 TRAINABLE
DINOv2              :    768,000 total,          0 trainable ❄️  FROZEN
Language Model      : 50,000,000 total,          0 trainable ❄️  FROZEN
Gating Network     :      2,048 total,      2,048 trainable 🔓 TRAINABLE
============================================================
```

### Custom Dataset
```python
from dataset import DualVisionDataset

class CustomDataset(DualVisionDataset):
    def format_conversations(self, conversations, image_path):
        # Custom conversation formatting
        return formatted_messages

# Use in training
dataset = CustomDataset("data.json")
```

### Custom Fusion Strategy
```python
# Modify the gating network in model.py
self.gating_network = nn.Sequential(
    # Your custom fusion architecture
)
```

### Multi-Dataset Training
```python
from dataset import MultiDataset, create_dataset

# Combine multiple datasets
config = {
    "type": "multi",
    "datasets": [
        {"type": "vqa", "args": {"data_path": "vqa_data.json"}},
        {"type": "captioning", "args": {"data_path": "caption_data.json"}}
    ],
    "weights": [0.7, 0.3]  # Sampling weights
}

dataset = create_dataset(config)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient checkpointing
   - Enable flash attention

2. **Model Loading Errors**
   - Check Hugging Face model names
   - Ensure sufficient disk space
   - Verify internet connection for downloads

3. **Dataset Loading Issues**
   - Verify image paths are correct
   - Check JSON format
   - Ensure images are readable

### Performance Tips

1. **Faster Training**
   - Use flash attention
   - Increase batch size if memory allows
   - Use multiple GPUs with data parallel

2. **Better Results**
   - Tune diversity loss weight
   - Experiment with different fusion dimensions
   - Fine-tune on domain-specific data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this work, please cite:

```bibtex
@misc{dual-vision-encoder,
  title={Dual Vision Encoder: Combining Qwen 2.5-VL and DINOv2 for Enhanced Multimodal Understanding},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## Acknowledgments

- Qwen team for Qwen 2.5-VL
- Meta AI for DINOv2
- Hugging Face for Transformers library
