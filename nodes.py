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


# --- STRICT ST-DRC REGISTRATION MAPPINGS ---

NODE_CLASS_MAPPINGS = {
    "STDRCContextInjector": STDRCContextInjector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STDRCContextInjector": "ST-DRC Context Latent Injector (Pine)",
}
