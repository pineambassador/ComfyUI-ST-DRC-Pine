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
    def IS_CHANGED(s, **kwargs):
        return random.random()

    def setup_guider(self, model, positive_text, positive_reference, negative, cfg, reference_scale):
        model_patched = model.clone()

        class ThreeStreamGuider(object):
            def __init__(self, model_patcher):
                self.model_patcher = model_patcher
                self.inner_model = model_patcher.model
                self.pos_text = positive_text
                self.pos_ref = positive_reference
                self.neg = negative
                self.cfg_scale = cfg
                self.ref_scale = reference_scale

            def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
                compute_device = getattr(self.model_patcher, "load_device", torch.device("cuda"))
                target_model = getattr(self.inner_model, "diffusion_model", self.inner_model)
                original_forward = target_model.forward

                def to_tensor(val):
                    # If the model returns a list/tuple (common with custom patchers), 
                    # extract the first element, which is the primary latent output.
                    if isinstance(val, (list, tuple)):
                        return val[0]
                    return val

                def custom_dit_forward(x, timesteps, context, *args, **kwargs):
                    if context.shape[0] < 2:
                        return original_forward(x, timesteps, context, *args, **kwargs)

                    cond_text = context[0:1]
                    cond_uncond = context[1:2]
                    cond_ref = cond_text.clone()

                    out_text = to_tensor(original_forward(x, timesteps, cond_text, *args, **kwargs))
                    out_ref = to_tensor(original_forward(x, timesteps, cond_ref, *args, **kwargs))
                    out_neg = to_tensor(original_forward(x, timesteps, cond_uncond, *args, **kwargs))

                    text_dir = out_text - out_neg
                    ref_dir = out_ref - out_neg
                    
                    return out_neg + (self.cfg_scale * text_dir) + (self.ref_scale * ref_dir)

                target_model.forward = custom_dit_forward
                try:
                    return comfy.samplers.sample(
                        model=self.model_patcher,
                        noise=noise,
                        positive=self.pos_text,
                        negative=self.neg,
                        cfg=1.0, 
                        latent_image=latent_image,
                        sampler=sampler,
                        sigmas=sigmas,
                        denoise_mask=denoise_mask,
                        callback=callback,
                        disable_pbar=disable_pbar,
                        seed=seed,
                        device=compute_device
                    )
                finally:
                    target_model.forward = original_forward

            def predict_noise(self, x, timestep, model_options={}, seed=None):
                return x

            def clone(self):
                return ThreeStreamGuider(self.model_patcher)

        return (ThreeStreamGuider(model_patched),)

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

