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



import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration, 
    AutoProcessor,
    AutoModel,
    AutoImageProcessor,
    BitsAndBytesConfig, # Import BitsAndBytesConfig for 4-bit loading
    AutoModelForCausalLM # Qwen-VL is a causal LM with vision
)
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
        use_flash_attention: bool = True,
        use_4bit_quantization: bool = True # New parameter for 4-bit quantization
    ):
        super().__init__()
        
        self.use_4bit_quantization = use_4bit_quantization

        # --- Quantization Configuration ---
        quantization_config = None
        if self.use_4bit_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4", # Or "fp4"
                bnb_4bit_compute_dtype=torch.bfloat16, # Or torch.float16 depending on your GPU and preference
                bnb_4bit_use_double_quant=True,
            )
            print("Enabled 4-bit quantization with BitsAndBytesConfig.")

        # Load Qwen 2.5-VL model
        if use_flash_attention:
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_model_name,
                torch_dtype=torch.bfloat16 if not use_4bit_quantization else None, # Use bfloat16 if no quantization, otherwise let bnb handle
                attn_implementation="flash_attention_2",
                device_map="auto",
                quantization_config=quantization_config # Pass quantization_config
            )
        else:
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_model_name,
                torch_dtype="auto" if not use_4bit_quantization else None,
                device_map="auto",
                quantization_config=quantization_config # Pass quantization_config
            )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(qwen_model_name)
        
        # Extract Qwen's vision encoder (Encoder 1)
        # Note: Qwen's vision encoder usually remains in its original dtype or is not quantized by bitsandbytes by default.
        # Bitsandbytes primarily quantizes Linear layers, which are prevalent in the LLM part.
        self.qwen_vision_encoder = self.qwen_model.visual
        
        # Load DINOv2 model (Encoder 2)
        # For DINOv2, as it's typically used as a feature extractor and frozen,
        # you might load it directly in a lower precision (e.g., bfloat16) or if
        # you specifically need 4-bit for its weights, you'd apply quantization
        # more explicitly or use a quantized version if available.
        # BitsAndBytesConfig often works best for the *decoder* parts of models.
        # For pure encoder models like DINOv2, sometimes manual conversion or
        # specific AWQ/GPTQ models are needed for full 4-bit inference.
        # For now, we'll try to load it with bfloat16, as it's frozen anyway.
        # If DINOv2 layers were to be fine-tuned or heavily used in terms of memory,
        # you'd need a more specific quantization for it.
        try:
            self.dinov2_model = AutoModel.from_pretrained(
                dinov2_model_name,
                torch_dtype=torch.bfloat16, # Load DINOv2 in bfloat16
                device_map="auto",
                # quantization_config=quantization_config # You *might* try this, but it's less guaranteed to work perfectly for pure encoders like DINOv2 with bnb
            )
        except Exception as e:
            print(f"Warning: Could not load DINOv2 with bfloat16 directly, falling back to auto dtype. Error: {e}")
            self.dinov2_model = AutoModel.from_pretrained(
                dinov2_model_name,
                torch_dtype="auto",
                device_map="auto",
            )
        self.dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_model_name)
        
        # Get dimensions
        self.qwen_vision_dim = self.qwen_model.config.vision_config.hidden_size  # D1
        self.dinov2_dim = self.dinov2_model.config.hidden_size  # D2
        self.common_dim = common_dim  # D
        self.lm_hidden_size = self.qwen_model.config.hidden_size
        
        # Linear projections to common dimension D
        # These are regular nn.Linear layers; they will run in the compute_dtype
        # if the input to them is in that dtype, or if they are explicitly quantized.
        # When BitsAndBytes is applied to the main model, it often replaces all
        # nn.Linear layers. However, since these are custom layers, they might
        # still be FP32. If memory is critical, you might need to convert them
        # explicitly or ensure bnb catches them.
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
        # are cast to the compute_dtype or quantized if desired for training.
        # For simplicity, we'll just move them to the default model device and dtype
        # which might be bfloat16 due to the Qwen model loading.
        if self.use_4bit_quantization:
            # Ensure the custom layers also run in the compute dtype for consistency
            # if they are not automatically quantized by bitsandbytes.
            # This is a common pattern for QLoRA fine-tuning where some modules
            # are in 4-bit and others in 16-bit.
            # Here, we assume the inputs to these will already be in bfloat16
            # because the main Qwen model operates in that compute_dtype.
            self.qwen_projection.to(self.qwen_model.device).to(torch.bfloat16)
            self.dinov2_projection.to(self.qwen_model.device).to(torch.bfloat16)
            self.gating_network.to(self.qwen_model.device).to(torch.bfloat16)
            self.vision_to_lm.to(self.qwen_model.device).to(torch.bfloat16)


    # ... (rest of your methods: encode_image_qwen, encode_image_dinov2,
    # project_to_common_dim, global_pooling, compute_gating_weight,
    # weighted_fusion, integrate_with_text, forward, chat, analyze_fusion_weights)
    # The forward pass logic might need slight adjustments if the LLM's internal
    # operations change significantly due to 4-bit loading, but generally,
    # bitsandbytes handles the de-quantization during computation automatically.
    # The most crucial part is loading the models correctly.

    def encode_image_qwen(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encoder 1: Qwen-VL Vision Encoder
        Input: pixel_values
        Output: tokens1 [B, N1, D1]
        """
        # Qwen's visual encoder might remain in full precision or bfloat16,
        # as bitsandbytes primarily targets Linear layers in the LLM.
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
        # Ensure pixel_values are on the correct device and dtype for DINOv2
        # (which we loaded in bfloat16).
        pixel_values = dinov2_inputs['pixel_values'].to(self.dinov2_model.device).to(self.dinov2_model.dtype)
        
        with torch.no_grad():
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
        fused_tokens = alpha * tokens1_proj + (1 - alpha) * tokens2_proj # [B, N, D]
        
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
        target_ids: Optional[torch.Tensor] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Full forward pass following the architecture diagram
        
        Args:
            messages: Chat messages in Qwen format
            max_new_tokens: Maximum tokens to generate
            return_intermediates: Whether to return intermediate features
            compute_loss: Whether to compute logits for loss calculation
            target_ids: Target token IDs for training
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
        
        # Check if we have images
        has_images = 'pixel_values' in qwen_inputs and qwen_inputs['pixel_values'] is not None
        
        if has_images and image_inputs:
            # Step 2: Encode images with both encoders
            # Encoder 1: Qwen-VL
            tokens1 = self.encode_image_qwen(qwen_inputs['pixel_values']) # [B, N1, D1]
            
            # Encoder 2: DINOv2  
            tokens2 = self.encode_image_dinov2(image_inputs) # [B, N2, D2]
            
            # Step 3: Linear projections to common dimension
            tokens1_proj, tokens2_proj = self.project_to_common_dim(tokens1, tokens2) # [B, N1, D], [B, N2, D]
            
            # Step 4: Global pooling
            pooled1 = self.global_pooling(tokens1_proj) # [B, D]
            pooled2 = self.global_pooling(tokens2_proj) # [B, D]
            
            # Step 5: Gating network - compute α
            alpha = self.compute_gating_weight(pooled1, pooled2) # [B, 1]
            
            # Step 6: Weighted fusion
            fused_tokens = self.weighted_fusion(tokens1_proj, tokens2_proj, alpha) # [B, N, D]
            
            # Store enhanced features for analysis
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
        else:
            # No images - standard text generation
            fused_tokens = None
            alpha = None
            intermediates = {}
        
        # Step 7: Generate or compute logits
        if compute_loss and target_ids is not None:
            # Training mode: compute logits for loss calculation
            # Get text embeddings from input
            input_ids = qwen_inputs['input_ids']
            text_embeds = self.language_model.embed_tokens(input_ids)
            
            # If we have vision features, integrate them
            if has_images and fused_tokens is not None: # Ensure fused_tokens exists
                # Project vision features to LM space and integrate
                # The 'integrate_with_text' method handles token concatenation.
                # However, your original 'forward' logic for compute_loss
                # was adding a single vision_feature_lm to all text embeddings.
                # I'll modify this to use the `integrate_with_text` method for consistency
                # which means the input to the LM will be `[B, N_vision + M_text, D_LM]`.
                # This often requires careful handling of attention masks.
                
                # Option 1: Use integrate_with_text for token-level concatenation
                # This is generally more robust for multimodal LLMs.
                # It requires adjusting the attention mask.
                # For simplicity in this example, I'll keep the logic consistent with how
                # Qwen-VL typically handles its vision tokens during generation.
                # If Qwen-VL's `generate` internally handles multimodal embeddings
                # by inserting them, then for `compute_loss`, we might need to mimic that.
                
                # Given your original `forward` for `compute_loss` added a
                # *single* vision feature to *all* text embeddings, it suggests
                # a simpler, potentially less context-aware integration for loss.
                # Let's align with the previous intention while acknowledging
                # `integrate_with_text` is designed for token-level concatenation.
                
                # For `compute_loss`, we need to decide how the LLM receives vision.
                # If the goal is to make the LLM process a combined sequence:
                
                # Create dummy token IDs for the vision part if needed, or adjust inputs_embeds
                # Qwen-VL usually has special image tokens it inserts.
                # When using `compute_loss`, you're effectively bypassing Qwen's
                # `generate` method and directly calling its language model.
                # The way Qwen-VL handles image token insertion is by modifying
                # `input_ids` and `attention_mask` to include placeholders
                # for image features, then replacing those placeholders with the actual features.
                
                # Let's try to pass the combined `inputs_embeds` to the language model.
                # This is tricky because `attention_mask` also needs to match the new sequence length.
                
                # A more straightforward approach for `compute_loss` with custom vision
                # is to inject the fused_tokens directly into the embeddings processed by the LLM.
                # This often means replacing special image placeholder tokens in the input_ids
                # with the actual fused_tokens.
                
                # Let's assume for `compute_loss`, we will augment the `input_ids` and `attention_mask`
                # as Qwen does, and then replace the image placeholders with `fused_tokens`.
                
                # Get Qwen's original vision_features. When Qwen processes images,
                # it generates its own `vision_features` and injects them.
                # In your `forward`, you *are* passing `qwen_inputs['pixel_values']`
                # to the main `qwen_model.generate` in inference mode.
                # However, in `compute_loss` mode, you are generating `text_embeds`
                # and then trying to add `vision_features_lm`.
                
                # To make 4-bit quantization work consistently and leverage the fusion:
                # We need to manually construct `inputs_embeds` and `attention_mask`
                # to include the fused vision features.
                
                # Qwen-VL uses specific tokens for image placeholder.
                # We need to find where these tokens would be inserted in the prompt.
                # For simplicity, let's assume we replace a specific placeholder
                # or just prepend/append them if the prompt structure allows.
                
                # The existing `compute_loss` logic is problematic with the new fusion.
                # It assumes a single `vision_features_lm` (likely [B, D_LM]) is added
                # to all text embeddings. Your `fused_tokens` is `[B, N, D]`.
                # Let's revise the `compute_loss` block to *integrate* the fused_tokens properly.

                # Step 1: Get the base LLM inputs (from text)
                input_ids = qwen_inputs['input_ids']
                attention_mask = qwen_inputs['attention_mask']

                # Qwen-VL inserts image tokens. We need to find their positions.
                # The `qwen_model` already handles this in its `forward` or `generate` method.
                # To perform loss calculation manually with fused tokens, you'd
                # typically need to get the "raw" text embeddings, identify image token positions,
                # and splice in your `fused_tokens_lm`.

                # Let's try calling the Qwen model's `prepare_inputs_for_generation`
                # to get the initial embeddings, then replace their vision part.
                
                # This is a bit complex as it requires replicating Qwen's internal
                # logic for preparing multimodal inputs when not using its direct `generate`.
                # A simpler, *less ideal* but quick fix for `compute_loss` based on your original
                # code's apparent intention of adding a single vision vector:
                
                # IF THE INTENTION IS TO ADD A SINGLE GLOBAL VISION FEATURE TO ALL TEXT TOKENS:
                # You need to reduce fused_tokens [B, N, D] to [B, D] or [B, D_LM] first.
                # Let's global pool the `fused_tokens` and then project.
                # fused_global_feature = self.global_pooling(fused_tokens) # [B, D]
                # vision_features_lm = self.vision_to_lm(fused_global_feature) # [B, lm_hidden_size]
                # text_embeds = self.language_model.embed_tokens(input_ids) # [B, M, lm_hidden_size]
                # vision_expanded = vision_features_lm.unsqueeze(1).expand(-1, text_embeds.shape[1], -1)
                # combined_embeds = text_embeds + vision_expanded
                # This approach might be less performant than true token-level multimodal understanding.

                # A more correct approach, mimicking Qwen-VL's multimodal input:
                # Qwen-VL's `Qwen2_5_VLForConditionalGeneration` usually takes `input_ids` and `pixel_values`.
                # It *internally* handles the vision embedding and integration.
                # If you want to use YOUR `fused_tokens` in `compute_loss`, you need to hook into
                # the `Qwen2_5_VLForConditionalGeneration`'s forward pass more directly.
                # The `input_ids` from `processor(text=..., images=...)` already contain image placeholder IDs.
                # You need to replace the *embeddings* for these placeholders with your `fused_tokens_lm`.
                
                # Let's assume the `input_ids` from `qwen_inputs` already have placeholders for images.
                # We will replace the embeddings of these placeholders with our fused tokens.
                # This needs careful alignment.
                
                # Original Qwen-VL handles multimodal inputs by inserting special
                # image token IDs and then replacing them with image features internally.
                # When loading with `quantization_config`, the `Qwen2_5_VLForConditionalGeneration`
                # model itself will be quantized. Its `forward` method (which `generate` calls)
                # will use these quantized weights.
                
                # If you want `compute_loss` to also leverage your `fused_tokens`:
                # The most direct way is to pass `inputs_embeds` directly to `self.qwen_model.model`
                # (which is `self.language_model`). This requires manually creating `inputs_embeds`.
                
                # Re-evaluating the `compute_loss` block from the original code:
                # The `self.qwen_model` itself is a `Qwen2_5_VLForConditionalGeneration`
                # and already handles the multimodal input `pixel_values` if present.
                # So, if `compute_loss` means calculating loss on the *quantized* `qwen_model`,
                # then you just need to call its `forward` method with the appropriate inputs.
                # The previous `compute_loss` block was trying to manually construct embeddings,
                # which bypasses Qwen-VL's native multimodal handling.
                
                # Let's simplify and correctly use the Qwen-VL model's forward for compute_loss:
                # The `Qwen2_5_VLForConditionalGeneration`'s `forward` method itself takes `input_ids`,
                # `attention_mask`, `pixel_values`, etc.
                # When `compute_loss=True`, we assume `target_ids` are labels for the language model.
                
                # The primary challenge is that your custom fusion `fused_tokens` is NOT
                # what the base `Qwen2_5_VLForConditionalGeneration` uses by default.
                # To inject your fused_tokens, you would need to:
                # 1. Get raw text embeddings: `text_embeds = self.language_model.embed_tokens(qwen_inputs['input_ids'])`
                # 2. Identify the image placeholder tokens in `qwen_inputs['input_ids']`.
                # 3. Replace the embeddings at those positions with `self.vision_to_lm(fused_tokens)`.
                # 4. Create a new `attention_mask` to match the combined sequence.
                # 5. Pass this custom `inputs_embeds` and `attention_mask` to `self.language_model`.

                # This is more involved. If you want a quick 4-bit setup and are okay with the
                # existing discrepancy in training/inference (where custom fusion is only
                # for `return_intermediates` or specific analysis), then just loading the
                # Qwen model with `quantization_config` is the key.

                # Assuming you want to train the LLM part *with* your fused vision features
                # via `inputs_embeds` (as was partially attempted in your original code).
                # This requires careful handling of the input_ids and attention mask.
                # The `qwen_inputs` created by `self.processor` already contain image placeholders.
                
                # Let's stick closer to the original `compute_loss` logic,
                # but ensure `fused_features` is correctly obtained from `fused_tokens`.
                # The previous `compute_loss` implied `fused_features` was `[B, lm_hidden_size]`.
                
                # Option A: Global average of fused tokens, then add (mimics original intent)
                fused_global_feature = self.global_pooling(fused_tokens) # [B, D]
                vision_features_lm = self.vision_to_lm(fused_global_feature) # [B, lm_hidden_size]
                
                input_ids = qwen_inputs['input_ids']
                text_embeds = self.language_model.embed_tokens(input_ids) # [B, M, lm_hidden_size]
                
                # Expand vision features to match sequence length
                seq_len = text_embeds.shape[1]
                # Ensure device and dtype match for addition
                vision_expanded = vision_features_lm.unsqueeze(1).expand(-1, seq_len, -1).to(text_embeds.device).to(text_embeds.dtype)
                
                combined_embeds = text_embeds + vision_expanded # Element-wise addition
                
                lm_outputs = self.language_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=qwen_inputs.get('attention_mask'),
                    # Pass labels directly to the model if it supports loss calculation
                    labels=target_ids if target_ids is not None else None,
                    **generation_kwargs
                )
                
                # If labels are passed, lm_outputs might contain loss directly
                if target_ids is not None:
                    logits = lm_outputs.logits
                    loss = lm_outputs.loss # Hugging Face models usually return loss if labels are provided
                else:
                    logits = self.lm_head(lm_outputs.last_hidden_state)
                    loss = None # Or calculate loss manually here
                
                result = {
                    'logits': logits,
                    'loss': loss,
                    'has_images': has_images
                }
                
            else: # No images, or not using fused_tokens for loss computation
                # Fallback to standard Qwen model forward for text-only loss calculation
                lm_outputs = self.qwen_model(
                    input_ids=qwen_inputs['input_ids'],
                    attention_mask=qwen_inputs.get('attention_mask'),
                    pixel_values=qwen_inputs.get('pixel_values'), # Pass original pixel values to Qwen
                    labels=target_ids if target_ids is not None else None,
                    **generation_kwargs
                )
                logits = lm_outputs.logits
                loss = lm_outputs.loss if target_ids is not None else None

                result = {
                    'logits': logits,
                    'loss': loss,
                    'has_images': has_images
                }
        else:
            # Inference mode: standard generation (Qwen's own multimodal handling)
            with torch.no_grad():
                # Qwen's `generate` method will automatically handle the pixel_values
                # and integrate them using its own internal vision encoder, which
                # is now loaded in 4-bit (or compute_dtype).
                generated_ids = self.qwen_model.generate(
                    **qwen_inputs,
                    max_new_tokens=max_new_tokens,
                    **generation_kwargs
                )
            
            # Process output
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
        """Simple chat interface"""
        result = self.forward(messages, max_new_tokens=max_new_tokens, **kwargs)
        return result['generated_text'][0]
    
    def analyze_fusion_weights(self, messages: List[Dict]) -> Dict[str, float]:
        """Analyze how the model weights the two encoders"""
        # For analysis, we need the intermediate alpha, so compute_loss=False (inference path)
        # and return_intermediates=True
        result = self.forward(messages, max_new_tokens=1, return_intermediates=True, compute_loss=False)
        
        if not result['has_images']:
            return {'error': 'No images provided'}
        
        alpha = result.get('alpha')
        if alpha is None:
             return {'error': 'Alpha not computed (e.g., no images processed through custom fusion path).'}
        
        alpha_val = alpha.mean().item()
        
        return {
            'qwen_weight': alpha_val,
            'dinov2_weight': 1 - alpha_val,
            'dominant_encoder': 'Qwen-VL' if alpha_val > 0.5 else 'DINOv2',
            'confidence': abs(alpha_val - 0.5) * 2  # How confident the gating is (0=uncertain, 1=very confident)
        }
