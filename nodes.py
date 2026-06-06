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
        # 1. Capture the shape parameters of your main video latent matrix
        video_tensor = latent_video["samples"].clone() # Shape: [B, C, F_vid, H_vid, W_vid]
        B, C, F_vid, H_vid, W_vid = video_tensor.shape
        
        ref_img = reference_image.clone()
        
        # 2. Match pixel dimensions for LTX-2.3's 8x downsampler
        target_h_pixels = H_vid * 8
        target_w_pixels = W_vid * 8
        
        ref_img = ref_img.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
        if ref_img.shape[-2:] != (target_h_pixels, target_w_pixels):
            ref_img = F.interpolate(ref_img, size=(target_h_pixels, target_w_pixels), mode="bilinear", align_corners=False)
        
        # 3. Shape the reference image down directly into the structural resolution of the video
        ref_latent = F.interpolate(ref_img, size=(H_vid, W_vid), mode="bilinear", align_corners=False)
        
        if ref_latent.shape[1] != C:
            ref_latent = ref_latent.repeat(1, C // ref_latent.shape[1] + 1, 1, 1)[:, :C, :, :]
            
        ref_tensor = ref_latent.unsqueeze(2) # Shapes out perfectly to [B, C, 1, H_vid, W_vid]

        # --- ALIGNED WITH PAPER: Concatenate reference frames as an APPENDIX (at the END of dim=2) ---
        extended_samples = torch.cat([video_tensor, ref_tensor], dim=2) 
        reference_frames = ref_tensor.shape[2]

        new_latent_dict = {
            "samples": extended_samples,
            "type": "video"
        }
        
        # 4. Dynamic, Error-Resilient Noise Mask Concatenation
        if "noise_mask" in latent_video:
            orig_mask = latent_video["noise_mask"].clone()
            
            if orig_mask.dim() == 3: # [F, H, W]
                orig_mask = orig_mask.unsqueeze(0).unsqueeze(0)
            elif orig_mask.dim() == 4: # [B, F, H, W] or [C, F, H, W]
                orig_mask = orig_mask.unsqueeze(1)
                
            mask_B, mask_C, mask_F, mask_H, mask_W = orig_mask.shape
            
            ref_mask = torch.zeros(
                (mask_B, mask_C, reference_frames, mask_H, mask_W), 
                device=video_tensor.device, 
                dtype=video_tensor.dtype
            )
            
            # Append reference mask to the end
            new_latent_dict["noise_mask"] = torch.cat([orig_mask, ref_mask], dim=2)

        return (new_latent_dict, reference_frames)


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
        patched_model = model.clone()
        
        def tass_rope_attention_patch(q, k, v, extra_options):
            seq_dim = 1
            total_tokens = q.shape[seq_dim]
            
            # Safely calculate token dimensions per frame
            approx_tokens_per_frame = total_tokens // (extra_options.get("original_shape", [1, 1, 96])[2] + reference_frames) if "original_shape" in extra_options else 1536
            ref_token_boundary = approx_tokens_per_frame * reference_frames
            
            if ref_token_boundary >= total_tokens:
                return q, k, v 
                
            # --- FIX: SLICE FROM THE BACK SINCE REF IS APPENDED AS AN APPENDIX ---
            split_idx = total_tokens - ref_token_boundary
            
            video_q = q[:, :split_idx, :, :]
            ref_q = q[:, split_idx:, :, :]
            
            video_k = k[:, :split_idx, :, :]
            ref_k = k[:, split_idx:, :, :]

            # Apply TASS-RoPE Spatial-Shift Math
            spatial_shift_tensor = torch.tensor(spatial_shift, dtype=q.dtype, device=q.device)
            ref_q_shifted = ref_q * torch.cos(spatial_shift_tensor)
            ref_k_shifted = ref_k * torch.cos(spatial_shift_tensor)

            # Apply Temporal-Adjacent Synchronization
            video_q_mean = video_q.mean(dim=seq_dim, keepdim=True)
            video_k_mean = video_k.mean(dim=seq_dim, keepdim=True)
            
            ref_q_final = ref_q_shifted + (video_q_mean * 0.01) 
            ref_k_final = ref_k_shifted + (video_k_mean * 0.01)

            # --- FIX: RECOMBINE PRESERVING THE ORIGINAL APPENDIX POSITION ---
            q_patched = torch.cat([video_q, ref_q_final], dim=seq_dim)
            k_patched = torch.cat([video_k, ref_k_final], dim=seq_dim)
            
            return q_patched, k_patched, v

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
                compute_device = getattr(self.model_patcher, "load_device", None)
                if compute_device is None:
                    compute_device = getattr(self.model_patcher, "current_device", torch.device("cuda"))

                has_dit = hasattr(self.inner_model, "diffusion_model")
                target_model = self.inner_model.diffusion_model if has_dit else self.inner_model
                original_forward = target_model.forward

                def custom_dit_forward(x, timesteps, context, *args, **kwargs):
                    if context.shape[0] != 2:
                        return original_forward(x, timesteps, context, *args, **kwargs)

                    cond_text_stream = context[0:1]
                    cond_uncond_stream = context[1:2]
                    cond_ref_stream = cond_text_stream.clone() 

                    def slice_batch_element(input_obj, idx):
                        if isinstance(input_obj, list):
                            return [item[idx:idx+1] for item in input_obj]
                        return input_obj[idx:idx+1]

                    x_stream_cond = slice_batch_element(x, 0)
                    x_stream_uncond = slice_batch_element(x, 1)

                    t_stream_0 = timesteps[0:1] if hasattr(timesteps, "shape") and timesteps.shape[0] == 2 else timesteps
                    t_stream_1 = timesteps[1:2] if hasattr(timesteps, "shape") and timesteps.shape[0] == 2 else timesteps

                    def make_batch_one(data_node, idx):
                        if torch.is_tensor(data_node):
                            if data_node.shape[0] == 2:
                                return data_node[idx:idx+1]
                            return data_node
                        elif isinstance(data_node, list):
                            return [make_batch_one(child, idx) for child in data_node]
                        elif isinstance(data_node, tuple):
                            return tuple(make_batch_one(child, idx) for child in data_node)
                        elif isinstance(data_node, dict):
                            return {k: make_batch_one(v, idx) for k, v in data_node.items()}
                        return data_node

                    kwargs_text = {}
                    kwargs_uncond = {}

                    for k, v in kwargs.items():
                        if k == "transformer_options" and isinstance(v, dict):
                            opts_text = v.copy()
                            opts_uncond = v.copy()
                            opts_text["cond_or_uncond"] = [0]
                            opts_uncond["cond_or_uncond"] = [1]
                            kwargs_text[k] = {nk: make_batch_one(nv, 0) for nk, nv in opts_text.items()}
                            kwargs_uncond[k] = {nk: make_batch_one(nv, 1) for nk, nv in opts_uncond.items()}
                        elif torch.is_tensor(v) and v.shape[0] == 2:
                            kwargs_text[k] = v[0:1]
                            kwargs_uncond[k] = v[1:2]
                        else:
                            kwargs_text[k] = make_batch_one(v, 0)
                            kwargs_uncond[k] = make_batch_one(v, 1)

                    out_text = original_forward(x_stream_cond, t_stream_0, cond_text_stream, *args, **kwargs_text)
                    out_ref = original_forward(x_stream_cond, t_stream_0, cond_ref_stream, *args, **kwargs_text)
                    out_neg = original_forward(x_stream_uncond, t_stream_1, cond_uncond_stream, *args, **kwargs_uncond)

                    if isinstance(out_text, list):
                        final_output = []
                        for i in range(len(out_text)):
                            text_dir = out_text[i] - out_neg[i]
                            ref_dir = out_ref[i] - out_neg[i]
                            mixed_block = out_neg[i] + (self.cfg_scale * text_dir) + (self.ref_scale * ref_dir)
                            final_output.append(torch.cat([mixed_block, mixed_block], dim=0))
                        return final_output
                    else:
                        text_direction = out_text - out_neg
                        ref_direction = out_ref - out_neg
                        mixed_vector = out_neg + (self.cfg_scale * text_direction) + (self.ref_scale * ref_direction)
                        return torch.cat([mixed_vector, mixed_vector], dim=0)

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

        guance_instance = ThreeStreamGuider(model_patched)
        return (guance_instance,)


class STDRCLatentTrimmer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "extended_latent": ("LATENT",),
                "reference_frames": ("INT", {"default": 1, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("trimmed_latent",)
    FUNCTION = "trim_context"
    CATEGORY = "ST-DRC/Post-Processing"

    def trim_context(self, extended_latent, reference_frames):
        samples = extended_latent["samples"].clone() # Shape: [B, C, F_total, H, W]
        
        # --- ALIGNED WITH PAPER: Slice off the APPENDIX frames from the END ---
        trimmed_samples = samples[:, :, :-reference_frames, :, :]
        
        new_latent_dict = {
            "samples": trimmed_samples,
            "type": "video"
        }
        
        if "noise_mask" in extended_latent:
            orig_mask = extended_latent["noise_mask"]
            new_latent_dict["noise_mask"] = orig_mask[:, :, :-reference_frames, :, :]

        return (new_latent_dict,)


NODE_CLASS_MAPPINGS = {
    "STDRCContextInjector": STDRCContextInjector,
    "TASSRoPEPatcher": TASSRoPEPatcher,
    "STDRCThreeStreamGuider": STDRCThreeStreamGuider,
    "STDRCLatentTrimmer": STDRCLatentTrimmer, 
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STDRCContextInjector": "ST-DRC Context Latent Injector (Pine)",
    "TASSRoPEPatcher": "ST-DRC TASS-RoPE Patcher (Pine)",
    "STDRCThreeStreamGuider": "ST-DRC CFG Guider (Pine)",
    "STDRCLatentTrimmer": "ST-DRC Latent Trimmer (Pine)",
}
