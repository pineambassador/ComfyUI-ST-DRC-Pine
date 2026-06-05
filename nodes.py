import torch
import torch.nn.functional as F

class STDRCContextInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_video": ("LATENT",),
                "reference_face": ("LATENT",),
                "scaling_correction": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("extended_latent", "reference_frames")
    FUNCTION = "inject_context"
    CATEGORY = "ST-DRC/Preparation"

    def inject_context(self, latent_video, reference_face, scaling_correction):
        video_tensor = latent_video["samples"].clone() # Shape: [B, C, F_vid, H, W]
        ref_tensor = reference_face["samples"].clone() # Shape: [B, C, F_ref, H, W]
        
        B, C, F_vid, H, W = video_tensor.shape
        
        # 1. Scaling correction for LTX latent space distribution 
        if scaling_correction:
            if torch.std(ref_tensor).item() < 0.5: 
                ref_tensor = ref_tensor / 0.18215

        # 2. Synchronize spatial coordinates via bilinear interpolation if mismatched
        if ref_tensor.shape[-2:] != (H, W):
            ref_flat = ref_tensor.squeeze(2) if ref_tensor.dim() == 5 else ref_tensor
            ref_resized = F.interpolate(ref_flat, size=(H, W), mode="bilinear")
            ref_tensor = ref_resized.unsqueeze(2) if ref_tensor.dim() == 5 else ref_resized.unsqueeze(1).transpose(1,2)

        # 3. Concatenate reference frames directly onto the temporal (F) axis
        extended_samples = torch.cat([ref_tensor, video_tensor], dim=2) 
        reference_frames = ref_tensor.shape[2]

        new_latent_dict = {
            "samples": extended_samples,
            "type": "video"
        }
        
        # Build matching zero-denoise noise mask for the prefix frames if a video mask exists
        if "noise_mask" in latent_video:
            orig_mask = latent_video["noise_mask"].clone()
            ref_mask = torch.zeros((B, 1, reference_frames, H, W), device=video_tensor.device, dtype=video_tensor.dtype)
            new_latent_dict["noise_mask"] = torch.cat([ref_mask, orig_mask], dim=2)

        return (new_latent_dict, reference_frames)

import torch

class TASSRoPEPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "reference_frames": ("INT", {"default": 1, "min": 1, "max": 64}),
                "spatial_shift": ("INT", {"default": 1000, "min": 0, "max": 5000, "step": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "ST-DRC/Patches"

    def patch_model(self, model, reference_frames, spatial_shift):
        # Clone the ComfyUI model wrapper structure to avoid modifying global memory
        patched_model = model.clone()
        
        # This is our custom attention patcher function that overrides LTX's internal layer
        def tass_rope_attention_patch(q, k, v, extra_options):
            # LTX-2.3 shapes for q and k inside attention blocks are typically:
            # [Batch, Sequence_Length, Heads, Head_Dim] or [Batch, Heads, Sequence_Length, Head_Dim]
            # We determine where the sequence length is (usually dim 1 or dim 2)
            
            # For this patch, we assume standard DiT layout where dim=1 is sequence length
            # If the architecture uses a different dimension order, we adapt via extra_options
            seq_dim = 1
            
            # Calculate the exact token offset. 
            # LTX compresses video spatially by a factor of 8 and temporally by a factor of 8 (or 4).
            # A single reference frame usually translates to: (H // 8) * (W // 8) tokens.
            # We can calculate this dynamically based on the absolute ratio:
            total_tokens = q.shape[seq_dim]
            
            # We determine how many tokens belong to our reference prefix frame(s)
            # Since the reference was concatenated at the front, they occupy the first N tokens
            # For simplicity, we assume reference tokens = (total_tokens / total_frames) * reference_frames
            # A more precise way is passing the exact token count, but let's look at frame ratio:
            pass
            
            # --- THE CORE TASS-ROPE OVERRIDE MATH ---
            # 1. Slice the tensor into [Reference Tokens] and [Video Tokens]
            # 2. Leave Video RoPE intact.
            # 3. For Reference RoPE: 
            #    - Shift its spatial frequency grid parameters by adding `spatial_shift`
            #    - Match its temporal index frequency to the current active video segment
            
            # For now, we return q and k unmodified until we hook the exact block function name
            return q, k, v

        # We attach this patch hook natively to ComfyUI's model patcher system
        # In LTX-2.3, the target attention hook is typically registered under 'patched_attention'
        patched_model.set_model_attn_1_patch(tass_rope_attention_patch)
        
        return (patched_model,)


# --- STRICT ST-DRC REGISTRATION MAPPINGS ---

NODE_CLASS_MAPPINGS = {
    "STDRCContextInjector": STDRCContextInjector,
    "TASSRoPEPatcher": TASSRoPEPatcher, # REGISTER NEW PATCHER
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STDRCContextInjector": "ST-DRC Context Latent Injector (Pine)",
    "TASSRoPEPatcher": "ST-DRC TASS-RoPE Patcher (Pine)", # REGISTER DISPLAY NAME
}
