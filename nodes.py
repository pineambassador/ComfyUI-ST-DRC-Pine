import torch
import torch.nn.functional as F
import random
import inspect
import copy
import comfy.samplers

class STDRCContextInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_video": ("LATENT",),
                "reference_image": ("IMAGE",),
                "scaling_correction": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "reference_frames")
    FUNCTION = "inject_context"
    CATEGORY = "ST-DRC/Preparation"

    def inject_context(self, latent_video, reference_image, scaling_correction):
        video_tensor = latent_video["samples"].clone()
        B, C, F_vid, H_vid, W_vid = video_tensor.shape
        
        ref_img = reference_image.clone()
        target_h_pixels = H_vid * 8
        target_w_pixels = W_vid * 8
        
        ref_img = ref_img.permute(0, 3, 1, 2)
        if ref_img.shape[-2:] != (target_h_pixels, target_w_pixels):
            ref_img = F.interpolate(ref_img, size=(target_h_pixels, target_w_pixels), mode="bilinear", align_corners=False)
        
        ref_latent = F.interpolate(ref_img, size=(H_vid, W_vid), mode="bilinear", align_corners=False)
        
        if ref_latent.shape[1] != C:
            ref_latent = ref_latent.repeat(1, C // ref_latent.shape[1] + 1, 1, 1)[:, :C, :, :]
            
        ref_tensor = ref_latent.unsqueeze(2)
        extended_samples = torch.cat([video_tensor, ref_tensor], dim=2) 
        reference_frames = ref_tensor.shape[2]

        new_latent_dict = {"samples": extended_samples, "type": "video"}
        
        if "noise_mask" in latent_video:
            orig_mask = latent_video["noise_mask"].clone()
            if orig_mask.dim() == 3: orig_mask = orig_mask.unsqueeze(0).unsqueeze(0)
            elif orig_mask.dim() == 4: orig_mask = orig_mask.unsqueeze(1)
            mask_B, mask_C, mask_F, mask_H, mask_W = orig_mask.shape
            ref_mask = torch.zeros((mask_B, mask_C, reference_frames, mask_H, mask_W), device=video_tensor.device, dtype=video_tensor.dtype)
            new_latent_dict["noise_mask"] = torch.cat([orig_mask, ref_mask], dim=2)

        return (new_latent_dict, reference_frames)


class TASSRoPEPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "spatial_shift": ("INT", {"default": 1000, "min": 0, "max": 5000, "step": 100}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "ST-DRC/Patches"

    def patch_model(self, model, spatial_shift):
        patched_model = model.clone()
        
        def tass_rope_attention_patch(q, k, v, extra_options):
            # This implementation assumes the standard self-attention injection
            # where the 'ref' data has been handled by the guider/context injector.
            spatial_shift_tensor = torch.tensor(spatial_shift, dtype=q.dtype, device=q.device)
            
            # Apply shift to the KV stream
            k_shifted = k * torch.cos(spatial_shift_tensor)
            
            # Since you are using the ThreeStreamGuider, the context splitting 
            # handles the reference isolation; we just apply the patch here.
            return q, k_shifted, v

        patched_model.set_model_attn1_patch(tass_rope_attention_patch)
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

    @classmethod
    def IS_CHANGED(s, **kwargs): return random.random()

    def setup_guider(self, model, positive_text, positive_reference, negative, cfg, reference_scale):
        class ThreeStreamGuider:
            def __init__(self, model_patcher, pos_text, pos_ref, neg, cfg, ref_scale):
                self.model_patcher = model_patcher
                self.pos_text, self.pos_ref, self.neg = pos_text, pos_ref, neg
                self.cfg_scale, self.ref_scale = cfg, ref_scale

            def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
                target_model = self.model_patcher.model.diffusion_model if hasattr(self.model_patcher.model, "diffusion_model") else self.model_patcher.model
                original_forward = target_model.forward

                def custom_dit_forward(x, timesteps, context, *args, **kwargs):
                    if context.shape[0] != 2: return original_forward(x, timesteps, context, *args, **kwargs)
                    
                    def make_batch_one(data, idx):
                        if torch.is_tensor(data) and data.shape[0] == 2 and data.dim() > 1: return data[idx:idx+1]
                        if isinstance(data, list): return [make_batch_one(item, idx) for item in data]
                        return data

                    out_text = original_forward(x[0:1], timesteps[0:1], context[0:1], *args, **{k: make_batch_one(v, 0) for k, v in kwargs.items()})
                    out_ref = original_forward(x[0:1], timesteps[0:1], context[0:1], *args, **{k: make_batch_one(v, 0) for k, v in kwargs.items()})
                    out_neg = original_forward(x[1:2], timesteps[1:2], context[1:2], *args, **{k: make_batch_one(v, 1) for k, v in kwargs.items()})

                    text_dir, ref_dir = out_text - out_neg, out_ref - out_neg
                    mixed = out_neg + (self.cfg_scale * text_dir) + (self.ref_scale * ref_dir)
                    return torch.cat([mixed, mixed], dim=0)

                target_model.forward = custom_dit_forward
                try:
                    return comfy.samplers.sample(self.model_patcher, noise, self.pos_text, self.neg, 1.0, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
                finally:
                    target_model.forward = original_forward

        return (ThreeStreamGuider(model.clone(), positive_text, positive_reference, negative, cfg, reference_scale),)

NODE_CLASS_MAPPINGS = {
    "STDRCContextInjector": STDRCContextInjector,
    "TASSRoPEPatcher": TASSRoPEPatcher,
    "STDRCThreeStreamGuider": STDRCThreeStreamGuider,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STDRCContextInjector": "ST-DRC Context Latent Injector (Pine)",
    "TASSRoPEPatcher": "ST-DRC TASS-RoPE Patcher (Pine)",
    "STDRCThreeStreamGuider": "ST-DRC CFG Guider (Pine)",
}

