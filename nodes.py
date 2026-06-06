import torch
import torch.nn.functional as F
import random
import comfy.samplers

class STDRCContextInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_video": ("LATENT",),
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent_video", "ref_latent")
    FUNCTION = "inject_context"
    CATEGORY = "ST-DRC/Preparation"

    def inject_context(self, latent_video, reference_image):
        video_tensor = latent_video["samples"].clone()
        B, C, F, H, W = video_tensor.shape
        
        ref_img = reference_image.permute(0, 3, 1, 2)
        ref_img = F.interpolate(ref_img, size=(H * 8, W * 8), mode="bilinear", align_corners=False)
        ref_latent = F.interpolate(ref_img, size=(H, W), mode="bilinear", align_corners=False)
        
        if ref_latent.shape[1] != C:
            ref_latent = ref_latent.repeat(1, C // ref_latent.shape[1] + 1, 1, 1)[:, :C, :, :]
            
        ref_tensor = ref_latent.unsqueeze(2) # [B, C, 1, H, W]
        return (latent_video, {"samples": ref_tensor})


class TASSRoPEPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT",),
                "spatial_shift": ("INT", {"default": 1000, "min": 0, "max": 5000, "step": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "ST-DRC/Patches"

    def patch_model(self, model, ref_latent, spatial_shift):
        patched_model = model.clone()
        ref_tensor = ref_latent["samples"]

        def injection_patch(q, k, v, extra_options):
            # Flatten reference to token format [B, Tokens, D]
            ref_tokens = ref_tensor.view(ref_tensor.shape[0], ref_tensor.shape[1], -1).permute(0, 2, 1)
            
            # Apply spatial shift to reference features
            shift_val = torch.tensor(spatial_shift, dtype=q.dtype, device=q.device)
            ref_k = k * torch.cos(shift_val)
            
            # Inject into Key/Value streams
            k = torch.cat([k, ref_k], dim=1)
            v = torch.cat([v, ref_tokens], dim=1)
            return q, k, v

        patched_model.set_model_attn1_patch(injection_patch)
        return (patched_model,)


class STDRCThreeStreamGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive_text": ("CONDITIONING",),
                "positive_reference": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "reference_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "setup_guider"
    CATEGORY = "ST-DRC/Sampling"

    def setup_guider(self, model, positive_text, positive_reference, negative, cfg, reference_scale):
        # ... (Keep your existing ThreeStreamGuider internal logic here)
        # Note: Your forward patching logic remains valid as long as 'x' is not 
        # modified by concatenation.
        pass

NODE_CLASS_MAPPINGS = {
    "STDRCContextInjector": STDRCContextInjector,
    "TASSRoPEPatcher": TASSRoPEPatcher,
    "STDRCThreeStreamGuider": STDRCThreeStreamGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STDRCContextInjector": "ST-DRC Context Injector (Pine)",
    "TASSRoPEPatcher": "ST-DRC TASS-RoPE Patcher (Pine)",
    "STDRCThreeStreamGuider": "ST-DRC CFG Guider (Pine)",
}
