# """
# model.py - Main Dual Vision Encoder Architecture
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import (
#     Qwen2_5_VLForConditionalGeneration, 
#     AutoProcessor,
#     AutoModel,
#     AutoImageProcessor
# )
# from qwen_vl_utils import process_vision_info
# from typing import Optional, Tuple, Dict, Any, List


# class DualVisionEncoder(nn.Module):
#     """
#     Dual Vision Encoder following the exact architecture diagram:
#     - Two encoders (Qwen-VL + DINOv2) 
#     - Linear projections to common dimension
#     - Global pooling and gating network
#     - Weighted fusion with learned alpha
#     - Integration with LLM
#     """
    
#     def __init__(
#         self,
#         qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
#         dinov2_model_name: str = "facebook/dinov2-base",
#         common_dim: int = 1024,
#         dropout: float = 0.1,
#         use_flash_attention: bool = True
#     ):
#         super().__init__()
        
#         # Load Qwen 2.5-VL model
#         if use_flash_attention:
#             self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#                 qwen_model_name,
#                 torch_dtype=torch.bfloat16,
#                 attn_implementation="flash_attention_2",
#                 device_map="auto"
#             )
#         else:
#             self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#                 qwen_model_name,
#                 torch_dtype="auto",
#                 device_map="auto"
#             )
        
#         # Load processor
#         self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        
#         # Extract Qwen's vision encoder (Encoder 1)
#         self.qwen_vision_encoder = self.qwen_model.visual
        
#         # Load DINOv2 model (Encoder 2)
#         self.dinov2_model = AutoModel.from_pretrained(dinov2_model_name)
#         self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        
#         # Get dimensions
#         self.qwen_vision_dim = self.qwen_model.config.vision_config.hidden_size  # D1
#         self.dinov2_dim = self.dinov2_model.config.hidden_size  # D2
#         self.common_dim = common_dim  # D
#         self.lm_hidden_size = self.qwen_model.config.hidden_size
        
#         # Linear projections to common dimension D
#         self.qwen_projection = nn.Linear(self.qwen_vision_dim, self.common_dim)
#         self.dinov2_projection = nn.Linear(self.dinov2_dim, self.common_dim)
        
#         # Gating Network (MLP + Sigmoid)
#         # Input: concatenated pooled vectors [2*D] -> Output: scalar α
#         self.gating_network = nn.Sequential(
#             nn.Linear(self.common_dim * 2, self.common_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.common_dim, self.common_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.common_dim // 2, 1),
#             nn.Sigmoid()  # Output α ∈ [0,1]
#         )
        
#         # Projection from common_dim to LLM hidden size for integration
#         self.vision_to_lm = nn.Linear(self.common_dim, self.lm_hidden_size)
        
#         # Keep references to LLM components
#         self.language_model = self.qwen_model.model
#         self.lm_head = self.qwen_model.lm_head
        
#         # Freeze DINOv2 by default
#         for param in self.dinov2_model.parameters():
#             param.requires_grad = False
    
#     def encode_image_qwen(self, pixel_values: torch.Tensor) -> torch.Tensor:
#         """
#         Encoder 1: Qwen-VL Vision Encoder
#         Input: pixel_values
#         Output: tokens1 [B, N1, D1]
#         """
#         vision_outputs = self.qwen_vision_encoder(pixel_values)
#         return vision_outputs  # [B, N1, D1]
    
#     def encode_image_dinov2(self, images: List) -> torch.Tensor:
#         """
#         Encoder 2: DINOv2 Vision Encoder  
#         Input: raw images
#         Output: tokens2 [B, N2, D2]
#         """
#         # Process images for DINOv2
#         dinov2_inputs = self.dinov2_processor(images=images, return_tensors="pt")
#         pixel_values = dinov2_inputs['pixel_values'].to(self.dinov2_model.device)
        
#         with torch.no_grad():
#             outputs = self.dinov2_model(pixel_values)
#             return outputs.last_hidden_state  # [B, N2, D2]
    
#     def project_to_common_dim(self, tokens1: torch.Tensor, tokens2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Linear projections to common dimension D
#         tokens1 [B, N1, D1] -> tokens1_proj [B, N1, D]
#         tokens2 [B, N2, D2] -> tokens2_proj [B, N2, D]
#         """
#         tokens1_proj = self.qwen_projection(tokens1)  # [B, N1, D]
#         tokens2_proj = self.dinov2_projection(tokens2)  # [B, N2, D]
#         return tokens1_proj, tokens2_proj
    
#     def global_pooling(self, tokens: torch.Tensor) -> torch.Tensor:
#         """
#         Global pooling (mean) over sequence dimension
#         Input: [B, N, D] -> Output: [B, D]
#         """
#         return tokens.mean(dim=1)  # [B, D]
    
#     def compute_gating_weight(self, pooled1: torch.Tensor, pooled2: torch.Tensor) -> torch.Tensor:
#         """
#         Gating Network: Compute fusion weight α
#         Input: pooled vectors [B, D] each
#         Output: α [B, 1] ∈ [0,1]
#         """
#         # Concatenate pooled vectors
#         concat_pooled = torch.cat([pooled1, pooled2], dim=-1)  # [B, 2*D]
        
#         # Pass through gating network
#         alpha = self.gating_network(concat_pooled)  # [B, 1]
#         return alpha
    
#     def weighted_fusion(self, tokens1_proj: torch.Tensor, tokens2_proj: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
#         """
#         Weighted fusion: fused_tokens = α*tokens1_proj + (1-α)*tokens2_proj
        
#         Args:
#             tokens1_proj: [B, N1, D]
#             tokens2_proj: [B, N2, D]  
#             alpha: [B, 1]
        
#         Returns:
#             fused_tokens: [B, N, D] where N = max(N1, N2)
#         """
#         B, N1, D = tokens1_proj.shape
#         N2 = tokens2_proj.shape[1]
        
#         # Handle different sequence lengths by padding or truncating to same length
#         N = max(N1, N2)
        
#         if N1 < N:
#             # Pad tokens1_proj
#             padding = torch.zeros(B, N - N1, D, device=tokens1_proj.device, dtype=tokens1_proj.dtype)
#             tokens1_proj = torch.cat([tokens1_proj, padding], dim=1)
#         elif N1 > N:
#             tokens1_proj = tokens1_proj[:, :N, :]
            
#         if N2 < N:
#             # Pad tokens2_proj
#             padding = torch.zeros(B, N - N2, D, device=tokens2_proj.device, dtype=tokens2_proj.dtype)
#             tokens2_proj = torch.cat([tokens2_proj, padding], dim=1)
#         elif N2 > N:
#             tokens2_proj = tokens2_proj[:, :N, :]
        
#         # Expand alpha for broadcasting: [B, 1] -> [B, 1, 1]
#         alpha = alpha.unsqueeze(-1)  # [B, 1, 1]
        
#         # Weighted fusion
#         fused_tokens = alpha * tokens1_proj + (1 - alpha) * tokens2_proj  # [B, N, D]
        
#         return fused_tokens
    
#     def integrate_with_text(self, fused_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
#         """
#         Concatenate fused vision tokens with text tokens
        
#         Args:
#             fused_tokens: [B, N, D] 
#             text_tokens: [B, M, D]
            
#         Returns:
#             combined_tokens: [B, N+M, D]
#         """
#         # Project vision tokens to LLM hidden size
#         fused_tokens_lm = self.vision_to_lm(fused_tokens)  # [B, N, lm_hidden_size]
        
#         # Concatenate along sequence dimension
#         combined_tokens = torch.cat([fused_tokens_lm, text_tokens], dim=1)  # [B, N+M, lm_hidden_size]
        
#         return combined_tokens
    
#     def forward(
#         self,
#         messages: List[Dict],
#         max_new_tokens: int = 128,
#         return_intermediates: bool = False,
#         compute_loss: bool = False,
#         target_ids: Optional[torch.Tensor] = None,
#         **generation_kwargs
#     ) -> Dict[str, Any]:
#         """
#         Full forward pass following the architecture diagram
        
#         Args:
#             messages: Chat messages in Qwen format
#             max_new_tokens: Maximum tokens to generate
#             return_intermediates: Whether to return intermediate features
#             compute_loss: Whether to compute logits for loss calculation
#             target_ids: Target token IDs for training
#         """
#         # Step 1: Process inputs using Qwen's pipeline
#         text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         image_inputs, video_inputs = process_vision_info(messages)
        
#         qwen_inputs = self.processor(
#             text=[text],
#             images=image_inputs,
#             videos=video_inputs,
#             padding=True,
#             return_tensors="pt",
#         )
#         qwen_inputs = {k: v.to(self.qwen_model.device) for k, v in qwen_inputs.items()}
        
#         # Check if we have images
#         has_images = 'pixel_values' in qwen_inputs and qwen_inputs['pixel_values'] is not None
        
#         if has_images and image_inputs:
#             # Step 2: Encode images with both encoders
#             # Encoder 1: Qwen-VL
#             tokens1 = self.encode_image_qwen(qwen_inputs['pixel_values'])  # [B, N1, D1]
            
#             # Encoder 2: DINOv2  
#             tokens2 = self.encode_image_dinov2(image_inputs)  # [B, N2, D2]
            
#             # Step 3: Linear projections to common dimension
#             tokens1_proj, tokens2_proj = self.project_to_common_dim(tokens1, tokens2)  # [B, N1, D], [B, N2, D]
            
#             # Step 4: Global pooling
#             pooled1 = self.global_pooling(tokens1_proj)  # [B, D]
#             pooled2 = self.global_pooling(tokens2_proj)  # [B, D]
            
#             # Step 5: Gating network - compute α
#             alpha = self.compute_gating_weight(pooled1, pooled2)  # [B, 1]
            
#             # Step 6: Weighted fusion
#             fused_tokens = self.weighted_fusion(tokens1_proj, tokens2_proj, alpha)  # [B, N, D]
            
#             # Store enhanced features for analysis
#             intermediates = {
#                 'tokens1': tokens1,
#                 'tokens2': tokens2,
#                 'tokens1_proj': tokens1_proj,
#                 'tokens2_proj': tokens2_proj,
#                 'pooled1': pooled1,
#                 'pooled2': pooled2,
#                 'alpha': alpha,
#                 'fused_tokens': fused_tokens
#             }
#         else:
#             # No images - standard text generation
#             fused_tokens = None
#             alpha = None
#             intermediates = {}
        
#         # Step 7: Generate or compute logits
#         if compute_loss and target_ids is not None:
#             # Training mode: compute logits for loss calculation
#             # Get text embeddings from input
#             input_ids = qwen_inputs['input_ids']
#             text_embeds = self.language_model.embed_tokens(input_ids)
            
#             # If we have vision features, integrate them
#             if has_images:
#                 # Project vision features to LM space
#                 vision_features_lm = self.vision_to_lm(fused_features)  # [batch_size, lm_hidden_size]
                
#                 # Expand vision features to match sequence length
#                 seq_len = text_embeds.shape[1]
#                 vision_expanded = vision_features_lm.unsqueeze(1).expand(-1, seq_len, -1)
                
#                 # Add vision features to text embeddings
#                 combined_embeds = text_embeds + vision_expanded
#             else:
#                 combined_embeds = text_embeds
            
#             # Forward through language model
#             lm_outputs = self.language_model(
#                 inputs_embeds=combined_embeds,
#                 attention_mask=qwen_inputs.get('attention_mask'),
#                 **generation_kwargs
#             )
            
#             # Get logits
#             logits = self.lm_head(lm_outputs.last_hidden_state)
            
#             result = {
#                 'logits': logits,
#                 'has_images': has_images
#             }
            
#         else:
#             # Inference mode: standard generation
#             with torch.no_grad():
#                 generated_ids = self.qwen_model.generate(
#                     **qwen_inputs,
#                     max_new_tokens=max_new_tokens,
#                     **generation_kwargs
#                 )
            
#             # Process output
#             generated_ids_trimmed = [
#                 out_ids[len(in_ids):] for in_ids, out_ids in zip(qwen_inputs.input_ids, generated_ids)
#             ]
            
#             output_text = self.processor.batch_decode(
#                 generated_ids_trimmed,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=False
#             )
            
#             result = {
#                 'generated_ids': generated_ids,
#                 'generated_text': output_text,
#                 'has_images': has_images
#             }
        
#         if has_images:
#             result.update({
#                 'alpha': alpha,
#                 'fusion_weight_qwen': alpha.mean().item(),
#                 'fusion_weight_dinov2': (1 - alpha).mean().item(),
#                 'fused_tokens': fused_tokens
#             })
        
#         if return_intermediates:
#             result['intermediates'] = intermediates
            
#         return result
    
#     def chat(
#         self,
#         messages: List[Dict],
#         max_new_tokens: int = 128,
#         **kwargs
#     ) -> str:
#         """Simple chat interface"""
#         result = self.forward(messages, max_new_tokens=max_new_tokens, **kwargs)
#         return result['generated_text'][0]
    
#     def analyze_fusion_weights(self, messages: List[Dict]) -> Dict[str, float]:
#         """Analyze how the model weights the two encoders"""
#         result = self.forward(messages, max_new_tokens=1, return_intermediates=True)
        
#         if not result['has_images']:
#             return {'error': 'No images provided'}
        
#         alpha = result['alpha'].mean().item()
        
#         return {
#             'qwen_weight': alpha,
#             'dinov2_weight': 1 - alpha,
#             'dominant_encoder': 'Qwen-VL' if alpha > 0.5 else 'DINOv2',
#             'confidence': abs(alpha - 0.5) * 2  # How confident the gating is (0=uncertain, 1=very confident)
#         }



"""
model.py - Main Dual Vision Encoder Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    BitsAndBytesConfig, # Import BitsAndBytesConfig for 4-bit loading
)
# Assuming qwen_vl_utils.py exists in the same directory or is importable
from qwen_vl_utils import process_vision_info 
from typing import Optional, Tuple, Dict, Any, List


class DualVisionEncoder(nn.Module):
    """
    Dual Vision Encoder following the exact architecture diagram:
    - Two encoders (Qwen-VL + DINOv2) 
    - Linear projections to common dimension
    - Global pooling and gating network
    - Weighted fusion with learned alpha
    - Integration with LLM
    """
    
    def __init__(
        self,
        qwen_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        dinov2_model_name: str = "facebook/dinov2-base",
        common_dim: int = 1024,
        dropout: float = 0.1,
        use_4bit_quantization: bool = True # New parameter for 4-bit quantization
    ):
        super().__init__()
        
        self.use_4bit_quantization = use_4bit_quantization

        # --- Quantization Configuration ---
        quantization_config = None
        if self.use_4bit_quantization:
            # Requires bitsandbytes library
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.bfloat16, 
                bnb_4bit_use_double_quant=True,
            )
            print("Enabled 4-bit quantization with BitsAndBytesConfig.")

        # Load Qwen 2.5-VL model
        # Removed use_flash_attention parameter and its conditional logic
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_name,
            torch_dtype=None, # BitsAndBytes handles the dtype
            device_map="auto",
            quantization_config=quantization_config 
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        
        # Extract Qwen's vision encoder (Encoder 1)
        self.qwen_vision_encoder = self.qwen_model.visual
        
        # Load DINOv2 model (Encoder 2)
        try:
            self.dinov2_model = AutoModel.from_pretrained(
                dinov2_model_name,
                torch_dtype=torch.bfloat16, # Load DINOv2 in bfloat16 for memory efficiency
                device_map="auto",
            )
            print(f"DINOv2 model loaded with dtype: {self.dinov2_model.dtype}")
        except Exception as e:
            print(f"Warning: Could not load DINOv2 with bfloat16 directly, falling back to auto dtype. Error: {e}")
            self.dinov2_model = AutoModel.from_pretrained(
                dinov2_model_name,
                torch_dtype="auto", # Fallback to auto dtype
                device_map="auto",
            )
        self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        
        # Get dimensions
        self.qwen_vision_dim = self.qwen_model.config.vision_config.hidden_size  # D1
        self.dinov2_dim = self.dinov2_model.config.hidden_size  # D2
        self.common_dim = common_dim  # D
        self.lm_hidden_size = self.qwen_model.config.hidden_size
        
        # Linear projections to common dimension D
        self.qwen_projection = nn.Linear(self.qwen_vision_dim, self.common_dim)
        self.dinov2_projection = nn.Linear(self.dinov2_dim, self.common_dim)
        
        # Gating Network (MLP + Sigmoid)
        self.gating_network = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.common_dim, self.common_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.common_dim // 2, 1),
            nn.Sigmoid()  # Output α ∈ [0,1]
        )
        
        # Projection from common_dim to LLM hidden size for integration
        self.vision_to_lm = nn.Linear(self.common_dim, self.lm_hidden_size)
        
        # Keep references to LLM components
        self.language_model = self.qwen_model.model 
        self.lm_head = self.qwen_model.lm_head
        
        # Freeze DINOv2 by default
        for param in self.dinov2_model.parameters():
            param.requires_grad = False
        
        # If 4-bit quantization is used, ensure these custom layers 
        if self.use_4bit_quantization:
            compute_dtype = quantization_config.bnb_4bit_compute_dtype if quantization_config else torch.bfloat16
            
            self.qwen_projection.to(self.qwen_model.device).to(compute_dtype)
            self.dinov2_projection.to(self.qwen_model.device).to(compute_dtype)
            self.gating_network.to(self.qwen_model.device).to(compute_dtype)
            self.vision_to_lm.to(self.qwen_model.device).to(compute_dtype)
            print(f"Custom projection and gating layers cast to {compute_dtype}")

    def encode_image_qwen(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encoder 1: Qwen-VL Vision Encoder
        Input: pixel_values
        Output: tokens1 [B, N1, D1]
        """
        vision_outputs = self.qwen_vision_encoder(pixel_values)
        return vision_outputs # [B, N1, D1]
    
    def encode_image_dinov2(self, images: List) -> torch.Tensor:
        """
        Encoder 2: DINOv2 Vision Encoder  
        Input: raw images
        Output: tokens2 [B, N2, D2]
        """
        # Process images for DINOv2
        dinov2_inputs = self.dinov2_processor(images=images, return_tensors="pt")
        pixel_values = dinov2_inputs['pixel_values'].to(self.dinov2_model.device).to(self.dinov2_model.dtype)
        
        with torch.no_grad(): # DINOv2 is frozen
            outputs = self.dinov2_model(pixel_values)
            return outputs.last_hidden_state # [B, N2, D2]
            
    def project_to_common_dim(self, tokens1: torch.Tensor, tokens2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear projections to common dimension D
        tokens1 [B, N1, D1] -> tokens1_proj [B, N1, D]
        tokens2 [B, N2, D2] -> tokens2_proj [B, N2, D]
        """
        tokens1_proj = self.qwen_projection(tokens1)  # [B, N1, D]
        tokens2_proj = self.dinov2_projection(tokens2)  # [B, N2, D]
        return tokens1_proj, tokens2_proj
    
    def global_pooling(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Global pooling (mean) over sequence dimension
        Input: [B, N, D] -> Output: [B, D]
        """
        return tokens.mean(dim=1) # [B, D]
    
    def compute_gating_weight(self, pooled1: torch.Tensor, pooled2: torch.Tensor) -> torch.Tensor:
        """
        Gating Network: Compute fusion weight α
        Input: pooled vectors [B, D] each
        Output: α [B, 1] ∈ [0,1]
        """
        # Concatenate pooled vectors
        concat_pooled = torch.cat([pooled1, pooled2], dim=-1) # [B, 2*D]
        
        # Pass through gating network
        alpha = self.gating_network(concat_pooled) # [B, 1]
        return alpha
    
    def weighted_fusion(self, tokens1_proj: torch.Tensor, tokens2_proj: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Weighted fusion: fused_tokens = α*tokens1_proj + (1-α)*tokens2_proj
        
        Args:
            tokens1_proj: [B, N1, D]
            tokens2_proj: [B, N2, D]  
            alpha: [B, 1]
        
        Returns:
            fused_tokens: [B, N, D] where N = max(N1, N2)
        """
        B, N1, D = tokens1_proj.shape
        N2 = tokens2_proj.shape[1]
        
        # Handle different sequence lengths by padding or truncating to same length
        N = max(N1, N2)
        
        if N1 < N:
            # Pad tokens1_proj
            padding = torch.zeros(B, N - N1, D, device=tokens1_proj.device, dtype=tokens1_proj.dtype)
            tokens1_proj = torch.cat([tokens1_proj, padding], dim=1)
        elif N1 > N:
            tokens1_proj = tokens1_proj[:, :N, :]
            
        if N2 < N:
            # Pad tokens2_proj
            padding = torch.zeros(B, N - N2, D, device=tokens2_proj.device, dtype=tokens2_proj.dtype)
            tokens2_proj = torch.cat([tokens2_proj, padding], dim=1)
        elif N2 > N:
            tokens2_proj = tokens2_proj[:, :N, :]
        
        # Expand alpha for broadcasting: [B, 1] -> [B, 1, 1]
        alpha = alpha.unsqueeze(-1) # [B, 1, 1]
        
        # Weighted fusion
        fused_tokens = alpha * tokens1_proj + (1 - alpha) * tokens2_proj  # [B, N, D]
        
        return fused_tokens
    
    def integrate_with_text(self, fused_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Concatenate fused vision tokens with text tokens
        
        Args:
            fused_tokens: [B, N, D] 
            text_tokens: [B, M, D]
            
        Returns:
            combined_tokens: [B, N+M, D]
        """
        # Project vision tokens to LLM hidden size
        fused_tokens_lm = self.vision_to_lm(fused_tokens) # [B, N, lm_hidden_size]
        
        # Concatenate along sequence dimension
        combined_tokens = torch.cat([fused_tokens_lm, text_tokens], dim=1) # [B, N+M, lm_hidden_size]
        
        return combined_tokens
    
    def forward(
        self,
        messages: List[Dict],
        max_new_tokens: int = 128,
        return_intermediates: bool = False,
        compute_loss: bool = False,
        target_ids: Optional[torch.Tensor] = None, # Labels for loss calculation
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Full forward pass following the architecture diagram
        
        Args:
            messages: Chat messages in Qwen format
            max_new_tokens: Maximum tokens to generate
            return_intermediates: Whether to return intermediate features
            compute_loss: Whether to compute logits for loss calculation.
                          If True and target_ids are provided, will calculate loss.
            target_ids: Target token IDs for training (labels for language model).
        """
        # Step 1: Process inputs using Qwen's pipeline
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        qwen_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs, 
            padding=True,
            return_tensors="pt",
        )
        qwen_inputs = {k: v.to(self.qwen_model.device) for k, v in qwen_inputs.items()}
        
        has_images = 'pixel_values' in qwen_inputs and qwen_inputs['pixel_values'] is not None
        
        fused_tokens = None
        alpha = None
        intermediates = {}

        if has_images and image_inputs:
            # Step 2: Encode images with both encoders
            tokens1 = self.encode_image_qwen(qwen_inputs['pixel_values']) # [B, N1, D1]
            tokens2 = self.encode_image_dinov2(image_inputs) # [B, N2, D2]
            
            # Step 3: Linear projections to common dimension D
            tokens1_proj, tokens2_proj = self.project_to_common_dim(tokens1, tokens2) # [B, N1, D], [B, N2, D]
            
            # Step 4: Global pooling
            pooled1 = self.global_pooling(tokens1_proj) # [B, D]
            pooled2 = self.global_pooling(tokens2_proj) # [B, D]
            
            # Step 5: Gating network - compute fusion weight α
            alpha = self.compute_gating_weight(pooled1, pooled2) # [B, 1]
            
            # Step 6: Weighted fusion
            fused_tokens = self.weighted_fusion(tokens1_proj, tokens2_proj, alpha) # [B, N, D]
            
            intermediates = {
                'tokens1': tokens1,
                'tokens2': tokens2,
                'tokens1_proj': tokens1_proj,
                'tokens2_proj': tokens2_proj,
                'pooled1': pooled1,
                'pooled2': pooled2,
                'alpha': alpha,
                'fused_tokens': fused_tokens
            }
        
        # Step 7: Generate response or compute loss using the LLM
        if compute_loss:
            input_ids = qwen_inputs['input_ids']
            text_embeds = self.language_model.embed_tokens(input_ids) # [B, M, lm_hidden_size]
            
            combined_embeds = text_embeds
            if has_images and fused_tokens is not None: 
                fused_global_feature = self.global_pooling(fused_tokens) # [B, D]
                vision_features_lm = self.vision_to_lm(fused_global_feature) # [B, lm_hidden_size]
                
                seq_len = text_embeds.shape[1]
                vision_expanded = vision_features_lm.unsqueeze(1).expand(-1, seq_len, -1)
                vision_expanded = vision_expanded.to(text_embeds.device).to(text_embeds.dtype)
                
                combined_embeds = text_embeds + vision_expanded

            lm_outputs = self.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=qwen_inputs.get('attention_mask'),
                labels=target_ids, 
                **generation_kwargs
            )
            
            logits = lm_outputs.logits
            loss = lm_outputs.loss 

            result = {
                'logits': logits,
                'loss': loss,
                'has_images': has_images
            }
            
        else:
            with torch.no_grad():
                generated_ids = self.qwen_model.generate(
                    **qwen_inputs, 
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(qwen_inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            result = {
                'generated_ids': generated_ids,
                'generated_text': output_text,
                'has_images': has_images
            }
        
        if has_images:
            result.update({
                'alpha': alpha,
                'fusion_weight_qwen': alpha.mean().item() if alpha is not None else None,
                'fusion_weight_dinov2': (1 - alpha).mean().item() if alpha is not None else None,
                'fused_tokens': fused_tokens
            })
        
        if return_intermediates:
            result['intermediates'] = intermediates
            
        return result
    
    def chat(
        self,
        messages: List[Dict],
        max_new_tokens: int = 128,
        **kwargs
    ) -> str:
        """
        Simple chat interface to generate a response for a given chat history.
        Uses the model in inference mode.
        """
        result = self.forward(messages, max_new_tokens=max_new_tokens, **kwargs)
        return result['generated_text'][0]
    
    def analyze_fusion_weights(self, messages: List[Dict]) -> Dict[str, float]:
        """
        Analyzes how the gating network weights the two encoders for a given input.
        Returns the average alpha value and dominant encoder.
        """
        result = self.forward(messages, max_new_tokens=1, return_intermediates=True, compute_loss=False)
        
        if not result['has_images']:
            return {'error': 'No images provided in the messages for fusion analysis.'}
        
        alpha = result.get('alpha')
        if alpha is None:
             return {'error': 'Alpha could not be computed (e.g., no images processed through custom fusion path).'}
        
        alpha_val = alpha.mean().item()
        
        return {
            'qwen_weight': alpha_val,
            'dinov2_weight': 1 - alpha_val,
            'dominant_encoder': 'Qwen-VL' if alpha_val > 0.5 else 'DINOv2',
            'confidence': abs(alpha_val - 0.5) * 2 
        }
