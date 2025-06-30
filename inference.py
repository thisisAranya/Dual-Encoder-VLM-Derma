"""
inference.py - Inference script for Dual Vision Encoder
"""

import torch
import argparse
import json
import os
from PIL import Image
from typing import List, Dict, Any, Optional
import time

from model import DualVisionEncoder
from utils import load_checkpoint, print_device_info, analyze_model_outputs


class DualVisionInference:
    """Inference wrapper for Dual Vision Encoder"""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to model config (optional)
            device: Device to run inference on
        """
        self.device = self._setup_device(device)
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Use default config
            config = self._get_default_config()
        
        # Initialize model
        print("Loading Dual Vision Encoder...")
        self.model = DualVisionEncoder(
            qwen_model_name=config.get('qwen_model_name', 'Qwen/Qwen2.5-VL-3B-Instruct'),
            dinov2_model_name=config.get('dinov2_model_name', 'facebook/dinov2-base'),
            common_dim=config.get('common_dim', 1024),
            dropout=config.get('dropout', 0.1),
            use_flash_attention=config.get('use_flash_attention', True)
        )
        
        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading checkpoint from {model_path}")
            load_checkpoint(model_path, self.model)
        
        self.model.eval()
        print("Model loaded successfully!")
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        return device
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'qwen_model_name': 'Qwen/Qwen2.5-VL-3B-Instruct',
            'dinov2_model_name': 'facebook/dinov2-base',
            'common_dim': 1024,
            'dropout': 0.1,
            'use_flash_attention': True
        }
    
    def chat(
        self,
        image_path: str,
        question: str,
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Chat with the model using an image and question
        
        Args:
            image_path: Path to the image file
            question: Question about the image
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary containing response and analysis
        """
        # Prepare messages in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        start_time = time.time()
        
        # Get response
        with torch.no_grad():
            result = self.model.forward(
                messages=messages,
                max_new_tokens=max_new_tokens,
                return_intermediates=True,
                **generation_kwargs
            )
        
        inference_time = time.time() - start_time
        
        # Analyze fusion weights
        fusion_analysis = self.model.analyze_fusion_weights(messages)
        
        return {
            'response': result['generated_text'][0],
            'fusion_analysis': fusion_analysis,
            'inference_time': inference_time,
            'has_images': result['has_images'],
            'intermediates': result.get('intermediates', {})
        }
    
    def batch_inference(
        self,
        image_paths: List[str],
        questions: List[str],
        max_new_tokens: int = 128,
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on multiple image-question pairs
        
        Args:
            image_paths: List of image file paths
            questions: List of questions
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of results for each input
        """
        if len(image_paths) != len(questions):
            raise ValueError("Number of images and questions must match")
        
        results = []
        for image_path, question in zip(image_paths, questions):
            try:
                result = self.chat(
                    image_path=image_path,
                    question=question,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': image_path,
                    'question': question
                })
        
        return results
    
    def analyze_image(
        self,
        image_path: str,
        analysis_type: str = "detailed",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform detailed analysis of an image
        
        Args:
            image_path: Path to image
            analysis_type: Type of analysis ('detailed', 'medical', 'captioning')
            
        Returns:
            Analysis results
        """
        # Define different analysis prompts
        prompts = {
            'detailed': "Please provide a detailed analysis of this image, describing all visible elements, objects, people, settings, colors, and any notable features.",
            'medical': "As a medical AI assistant, please analyze this medical image. Describe any visible anatomical structures, potential abnormalities, and relevant clinical observations.",
            'captioning': "Please provide a concise but comprehensive caption for this image.",
            'objects': "List and describe all the objects you can identify in this image.",
            'scene': "Describe the overall scene, setting, and context of this image.",
            'technical': "Provide a technical analysis of this image including composition, lighting, perspective, and visual elements."
        }
        
        prompt = prompts.get(analysis_type, prompts['detailed'])
        
        result = self.chat(
            image_path=image_path,
            question=prompt,
            max_new_tokens=256,
            **kwargs
        )
        
        result['analysis_type'] = analysis_type
        return result
    
    def compare_encoders(
        self,
        image_path: str,
        question: str = "Describe this image in detail."
    ) -> Dict[str, Any]:
        """
        Compare how much the model relies on each encoder
        
        Args:
            image_path: Path to image
            question: Question to ask
            
        Returns:
            Comparison analysis
        """
        result = self.chat(image_path, question)
        
        if 'fusion_analysis' in result and 'error' not in result['fusion_analysis']:
            fusion = result['fusion_analysis']
            
            comparison = {
                'dominant_encoder': fusion['dominant_encoder'],
                'confidence': fusion['confidence'],
                'qwen_weight': fusion['qwen_weight'],
                'dinov2_weight': fusion['dinov2_weight'],
                'interpretation': self._interpret_fusion_weights(fusion)
            }
            
            result['encoder_comparison'] = comparison
        
        return result
    
    def _interpret_fusion_weights(self, fusion_analysis: Dict[str, float]) -> str:
        """Interpret what the fusion weights mean"""
        qwen_weight = fusion_analysis['qwen_weight']
        confidence = fusion_analysis['confidence']
        
        if confidence < 0.2:
            return "The model is uncertain and relies roughly equally on both encoders."
        elif qwen_weight > 0.7:
            return "The model heavily relies on Qwen-VL encoder, suggesting the image benefits from multimodal pre-training."
        elif qwen_weight < 0.3:
            return "The model heavily relies on DINOv2 encoder, suggesting the image benefits from self-supervised visual understanding."
        else:
            return f"The model moderately prefers the {'Qwen-VL' if qwen_weight > 0.5 else 'DINOv2'} encoder."
    
    def benchmark_performance(
        self,
        test_images: List[str],
        test_questions: List[str],
        num_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark model performance
        
        Args:
            test_images: List of test image paths
            test_questions: List of test questions
            num_runs: Number of runs for averaging
            
        Returns:
            Performance metrics
        """
        print(f"Running benchmark with {len(test_images)} images, {num_runs} runs each...")
        
        all_times = []
        all_fusion_weights = {'qwen': [], 'dinov2': []}
        
        for run in range(num_runs):
            print(f"Run {run + 1}/{num_runs}")
            
            for image_path, question in zip(test_images, test_questions):
                try:
                    result = self.chat(image_path, question, max_new_tokens=50)
                    all_times.append(result['inference_time'])
                    
                    if 'fusion_analysis' in result and 'error' not in result['fusion_analysis']:
                        fusion = result['fusion_analysis']
                        all_fusion_weights['qwen'].append(fusion['qwen_weight'])
                        all_fusion_weights['dinov2'].append(fusion['dinov2_weight'])
                        
                except Exception as e:
                    print(f"Error in benchmark: {e}")
                    continue
        
        # Calculate statistics
        import numpy as np
        
        metrics = {
            'avg_inference_time': np.mean(all_times),
            'std_inference_time': np.std(all_times),
            'min_inference_time': np.min(all_times),
            'max_inference_time': np.max(all_times),
            'total_samples': len(all_times)
        }
        
        if all_fusion_weights['qwen']:
            metrics.update({
                'avg_qwen_weight': np.mean(all_fusion_weights['qwen']),
                'std_qwen_weight': np.std(all_fusion_weights['qwen']),
                'avg_dinov2_weight': np.mean(all_fusion_weights['dinov2']),
                'std_dinov2_weight': np.std(all_fusion_weights['dinov2'])
            })
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Dual Vision Encoder Inference")
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, help='Path to model config')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--question', type=str, default="Describe this image in detail.", help='Question to ask')
    parser.add_argument('--analysis_type', type=str, default='detailed', 
                       choices=['detailed', 'medical', 'captioning', 'objects', 'scene', 'technical'],
                       help='Type of analysis to perform')
    parser.add_argument('--max_tokens', type=int, default=128, help='Maximum tokens to generate')
    parser.add_argument('--output', type=str, help='Output file to save results')
    parser.add_argument('--compare_encoders', action='store_true', help='Compare encoder usage')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Print device info
    print_device_info()
    
    # Initialize inference engine
    inference_engine = DualVisionInference(
        model_path=args.model_path,
        config_path=args.config_path
    )
    
    if args.benchmark:
        # Simple benchmark with the provided image
        test_images = [args.image] * 5  # Use same image 5 times
        test_questions = [args.question] * 5
        
        metrics = inference_engine.benchmark_performance(test_images, test_questions)
        
        print("\nBenchmark Results:")
        print(f"Average inference time: {metrics['avg_inference_time']:.3f}s ± {metrics['std_inference_time']:.3f}s")
        print(f"Min/Max inference time: {metrics['min_inference_time']:.3f}s / {metrics['max_inference_time']:.3f}s")
        
        if 'avg_qwen_weight' in metrics:
            print(f"Average Qwen-VL weight: {metrics['avg_qwen_weight']:.3f} ± {metrics['std_qwen_weight']:.3f}")
            print(f"Average DINOv2 weight: {metrics['avg_dinov2_weight']:.3f} ± {metrics['std_dinov2_weight']:.3f}")
    
    elif args.compare_encoders:
        # Compare encoder usage
        result = inference_engine.compare_encoders(args.image, args.question)
        
        print(f"\nQuestion: {args.question}")
        print(f"Response: {result['response']}")
        
        if 'encoder_comparison' in result:
            comp = result['encoder_comparison']
            print(f"\nEncoder Analysis:")
            print(f"Dominant encoder: {comp['dominant_encoder']}")
            print(f"Confidence: {comp['confidence']:.3f}")
            print(f"Qwen-VL weight: {comp['qwen_weight']:.3f}")
            print(f"DINOv2 weight: {comp['dinov2_weight']:.3f}")
            print(f"Interpretation: {comp['interpretation']}")
        
        print(f"Inference time: {result['inference_time']:.3f}s")
    
    else:
        # Regular analysis
        if args.analysis_type != 'detailed':
            result = inference_engine.analyze_image(args.image, args.analysis_type, max_new_tokens=args.max_tokens)
        else:
            result = inference_engine.chat(args.image, args.question, max_new_tokens=args.max_tokens)
        
        print(f"\nImage: {args.image}")
        print(f"Question: {args.question}")
        print(f"Response: {result['response']}")
        
        if 'fusion_analysis' in result and 'error' not in result['fusion_analysis']:
            fusion = result['fusion_analysis']
            print(f"\nFusion Analysis:")
            print(f"Qwen-VL weight: {fusion['qwen_weight']:.3f}")
            print(f"DINOv2 weight: {fusion['dinov2_weight']:.3f}")
            print(f"Dominant encoder: {fusion['dominant_encoder']}")
            print(f"Confidence: {fusion['confidence']:.3f}")
        
        print(f"Inference time: {result['inference_time']:.3f}s")
    
    # Save results if output path provided
    if args.output:
        with open(args.output, 'w') as f:
            if args.benchmark:
                json.dump(metrics, f, indent=2)
            else:
                json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()