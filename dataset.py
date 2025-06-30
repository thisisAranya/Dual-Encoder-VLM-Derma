"""
dataset.py - Dataset classes for Dual Vision Encoder
"""

import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from typing import List, Dict, Any, Optional
import random


class DualVisionDataset(Dataset):
    """
    Dataset class for dual vision encoder training
    Supports various multimodal datasets in a unified format
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_samples: Optional[int] = None,
        image_root: Optional[str] = None
    ):
        """
        Args:
            data_path: Path to the dataset JSON file
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load (for debugging)
            image_root: Root directory for images (if not absolute paths)
        """
        self.data_path = data_path
        self.split = split
        self.image_root = image_root or ""
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Filter by split if specified
        if 'split' in self.data[0]:
            self.data = [item for item in self.data if item.get('split', 'train') == split]
        
        # Limit samples if specified
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset
        
        Expected data format:
        {
            "image_path": "path/to/image.jpg",
            "conversations": [
                {"role": "user", "content": "What's in this image?"},
                {"role": "assistant", "content": "This image shows..."}
            ]
        }
        """
        item = self.data[idx]
        
        # Get image path
        image_path = item['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Format conversations into Qwen message format
        conversations = item.get('conversations', [])
        messages = self.format_conversations(conversations, image_path)
        
        return {
            'messages': messages,
            'image': image,
            'image_path': image_path,
            'metadata': item.get('metadata', {})
        }
    
    def format_conversations(self, conversations: List[Dict], image_path: str) -> List[Dict]:
        """
        Format conversations into Qwen message format
        """
        messages = []
        
        for i, conv in enumerate(conversations):
            role = conv['role']
            content = conv['content']
            
            if role == 'user' and i == 0:
                # First user message - include image
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": content}
                    ]
                })
            else:
                # Other messages - text only
                messages.append({
                    "role": role,
                    "content": content
                })
        
        return messages
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for DataLoader
        """
        return {
            'messages': [item['messages'] for item in batch],
            'images': [item['image'] for item in batch],
            'image_paths': [item['image_path'] for item in batch],
            'metadata': [item['metadata'] for item in batch]
        }


class VQADataset(DualVisionDataset):
    """
    Specialized dataset for Visual Question Answering tasks
    """
    
    def format_conversations(self, conversations: List[Dict], image_path: str) -> List[Dict]:
        """Format VQA data into conversation format"""
        # VQA typically has question and answer
        question = conversations[0]['content'] if conversations else "What do you see in this image?"
        answer = conversations[1]['content'] if len(conversations) > 1 else "I can see various objects in this image."
        
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant", 
                "content": answer
            }
        ]


class CaptioningDataset(DualVisionDataset):
    """
    Specialized dataset for image captioning tasks
    """
    
    def format_conversations(self, conversations: List[Dict], image_path: str) -> List[Dict]:
        """Format captioning data into conversation format"""
        caption = conversations[0]['content'] if conversations else "Describe this image."
        
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Describe this image in detail."}
                ]
            },
            {
                "role": "assistant",
                "content": caption
            }
        ]


class MedicalVQADataset(DualVisionDataset):
    """
    Specialized dataset for medical VQA tasks
    """
    
    def format_conversations(self, conversations: List[Dict], image_path: str) -> List[Dict]:
        """Format medical VQA data with appropriate prompting"""
        question = conversations[0]['content'] if conversations else "What do you see in this medical image?"
        answer = conversations[1]['content'] if len(conversations) > 1 else "This appears to be a medical image."
        
        # Add medical context to the question
        medical_prompt = f"As a medical AI assistant, please analyze this medical image and answer the following question: {question}"
        
        return [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": medical_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]


class MultiDataset(Dataset):
    """
    Combines multiple datasets for joint training
    """
    
    def __init__(self, datasets: List[Dataset], weights: Optional[List[float]] = None):
        """
        Args:
            datasets: List of dataset objects
            weights: Sampling weights for each dataset
        """
        self.datasets = datasets
        self.weights = weights or [1.0] * len(datasets)
        
        # Calculate cumulative lengths
        self.lengths = [len(d) for d in datasets]
        self.cumulative_lengths = []
        cumsum = 0
        for length in self.lengths:
            cumsum += length
            self.cumulative_lengths.append(cumsum)
        
        self.total_length = sum(self.lengths)
        
        print(f"MultiDataset created with {len(datasets)} datasets, total length: {self.total_length}")
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item using weighted sampling"""
        if idx >= self.total_length:
            raise IndexError("Index out of range")
        
        # Find which dataset this index belongs to
        dataset_idx = 0
        while idx >= self.cumulative_lengths[dataset_idx]:
            dataset_idx += 1
        
        # Calculate local index within the dataset
        local_idx = idx
        if dataset_idx > 0:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]
        
        # Get item from appropriate dataset
        item = self.datasets[dataset_idx][local_idx]
        item['dataset_idx'] = dataset_idx
        
        return item
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function that handles multiple datasets"""
        # Use the collate function from the first dataset
        return self.datasets[0].collate_fn(batch)


def create_dataset(dataset_config: Dict) -> Dataset:
    """
    Factory function to create datasets based on configuration
    """
    dataset_type = dataset_config.get('type', 'dual_vision')
    
    if dataset_type == 'vqa':
        return VQADataset(**dataset_config['args'])
    elif dataset_type == 'captioning':
        return CaptioningDataset(**dataset_config['args'])
    elif dataset_type == 'medical_vqa':
        return MedicalVQADataset(**dataset_config['args'])
    elif dataset_type == 'multi':
        # Create multiple datasets
        datasets = []
        for sub_config in dataset_config['datasets']:
            datasets.append(create_dataset(sub_config))
        return MultiDataset(datasets, dataset_config.get('weights'))
    else:
        return DualVisionDataset(**dataset_config['args'])


# Example data format creators
def create_vqa_format(image_path: str, question: str, answer: str) -> Dict:
    """Create VQA format data entry"""
    return {
        "image_path": image_path,
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }


def create_captioning_format(image_path: str, caption: str) -> Dict:
    """Create captioning format data entry"""
    return {
        "image_path": image_path,
        "conversations": [
            {"role": "assistant", "content": caption}
        ]
    }


def create_dialogue_format(image_path: str, dialogue: List[Dict]) -> Dict:
    """Create multi-turn dialogue format"""
    return {
        "image_path": image_path,
        "conversations": dialogue
    }


# Utility functions for dataset creation
def convert_coco_to_format(coco_json_path: str, image_root: str, output_path: str):
    """Convert COCO dataset to our format"""
    import json
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image_id to filename mapping
    id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Convert annotations
    converted_data = []
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        filename = id_to_filename[image_id]
        caption = ann['caption']
        
        converted_data.append(create_captioning_format(
            image_path=os.path.join(image_root, filename),
            caption=caption
        ))
    
    # Save converted data
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} COCO samples to {output_path}")


def convert_vqav2_to_format(vqa_json_path: str, image_root: str, output_path: str):
    """Convert VQA v2 dataset to our format"""
    import json
    
    with open(vqa_json_path, 'r') as f:
        vqa_data = json.load(f)
    
    converted_data = []
    for item in vqa_data:
        image_path = os.path.join(image_root, f"COCO_val2014_{item['image_id']:012d}.jpg")
        question = item['question']
        
        # Get most common answer
        answers = [ans['answer'] for ans in item['answers']]
        most_common_answer = max(set(answers), key=answers.count)
        
        converted_data.append(create_vqa_format(
            image_path=image_path,
            question=question,
            answer=most_common_answer
        ))
    
    # Save converted data
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Converted {len(converted_data)} VQA v2 samples to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Testing dataset loading...")
    
    # Create a dummy dataset file for testing
    dummy_data = [
        {
            "image_path": "path/to/image1.jpg",
            "conversations": [
                {"role": "user", "content": "What's in this image?"},
                {"role": "assistant", "content": "This image shows a cat."}
            ]
        },
        {
            "image_path": "path/to/image2.jpg", 
            "conversations": [
                {"role": "user", "content": "Describe the scene."},
                {"role": "assistant", "content": "The scene shows a beautiful landscape."}
            ]
        }
    ]
    
    with open('/tmp/dummy_dataset.json', 'w') as f:
        json.dump(dummy_data, f)
    
    # Test dataset loading
    dataset = DualVisionDataset('/tmp/dummy_dataset.json')
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting an item
    try:
        item = dataset[0]
        print(f"Sample item keys: {item.keys()}")
        print(f"Messages format: {item['messages']}")
    except Exception as e:
        print(f"Error loading item: {e}")

        