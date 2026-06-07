import torch
import torch.nn.functional as F
import comfy.samplers

# Dynamically resolve ComfyUI's nested data structure classes safely
try:
    from comfy.nested_tensor import NestedTensor
except ImportError:
    NestedTensor = None

class ThreeStreamGuider(comfy.samplers.CFGGuider):
    def __init__(self, model, pos_text, pos_ref, neg, cfg, ref_scale):
        # The CFGGuider parent expects 'model' to be the primary argument
        super().__init__(model)
        self.model = model
        
        # CRITICAL: You must set self.cond and self.uncond 
        # so the parent CFGGuider can find the conditioning data
        self.set_conds(pos_text, neg) 
        
        self.pos_ref = pos_ref
        self.cfg = cfg
        self.ref_scale = ref_scale
        #print(f"ST-DRC: Guider initialized with cfg={self.cfg}, ref_scale={self.ref_scale}")
        #print(f"ST-DRC: Cond count: {len(pos_text)}, Neg count: {len(neg)}")

        # 1. Extract reference data safely immediately during initialization
        ref_data = self.pos_ref[0][0] if isinstance(self.pos_ref, list) else self.pos_ref
        if hasattr(ref_data, "is_nested") and ref_data.is_nested:
            ref_tensors = ref_data.unbind()
            self.ref_latent = ref_tensors[0].contiguous() if len(ref_tensors) > 0 else ref_data
        else:
            self.ref_latent = ref_data.contiguous() if hasattr(ref_data, "contiguous") else ref_data

        if len(self.ref_latent.shape) == 4:
            self.ref_latent = self.ref_latent.unsqueeze(2)

    def _align_and_inject_raw_tensor(self, target_raw_tensor):
        """Aligns and injects reference into target tensor, preserving structural integrity."""
        # Ensure we are operating on the right device/dtype from the start
        b_n, c_n, f_n, h_n, w_n = target_raw_tensor.shape
        b_r, c_r, f_r, h_r, w_r = self.ref_latent.shape

        # Use .to() for safety, ensuring device/dtype match the target
        aligned_ref = self.ref_latent.to(device=target_raw_tensor.device, dtype=target_raw_tensor.dtype)

        # Step A: Align Channel Matrix
        if c_r != c_n:
            if c_r == 3:
                repeats = (c_n // c_r) + 1
                aligned_ref = aligned_ref.repeat(1, repeats, 1, 1, 1)[:, :c_n, :, :, :]
            else:
                aligned_ref = aligned_ref.expand(-1, c_n, -1, -1, -1)

        # Step B: Align Spatial (H/W)
        if h_r != h_n or w_r != w_n:
            aligned_ref = aligned_ref.view(b_r, c_n * f_r, h_r, w_r)
            aligned_ref = F.interpolate(aligned_ref, size=(h_n, w_n), mode="bilinear", align_corners=False)
            aligned_ref = aligned_ref.view(b_r, c_n, f_r, h_n, w_n)

        # Step C: Align Temporal (Frames)
        if f_r != f_n:
            if f_r < f_n:
                pad_len = f_n - f_r
                padding = aligned_ref[:, :, -1:, :, :].repeat(1, 1, pad_len, 1, 1)
                aligned_ref = torch.cat([aligned_ref, padding], dim=2)
            else:
                aligned_ref = aligned_ref[:, :, :f_n, :, :]

        # Perform injection
        modulated = target_raw_tensor + (aligned_ref * self.ref_scale)
        
        # Debugging: Monitor injection impact
        #print(f"ST-DRC: Injection complete. Original Mean: {target_raw_tensor.mean():.4f}, Modulated Mean: {modulated.mean():.4f}")
        
        # Preserve NestedTensor wrapper if the input was one
        if NestedTensor is not None and isinstance(target_raw_tensor, NestedTensor):
             # We return a new NestedTensor structure or update the tensor if the wrapper supports it
             return modulated 
             
        return modulated

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        model = self.model.model.diffusion_model
        original_forward = model.forward
        
        # 1. CAPTURE the scale here in the local scope
        current_ref_scale = self.ref_scale 
        
        model_device = next(model.parameters()).device
        ref_tensor = self.ref_latent.to(device=model_device, dtype=noise.dtype)
        
        # ... (keep your existing ref_permuted and full_context logic) ...
        ref_permuted = ref_tensor.permute(0, 2, 3, 4, 1)
        padding_size = 2048
        audio_pad = torch.zeros((ref_permuted.shape[0], ref_permuted.shape[1], ref_permuted.shape[2], ref_permuted.shape[3], padding_size), device=model_device, dtype=ref_permuted.dtype)
        full_context = torch.cat([ref_permuted, audio_pad], dim=4)

        # 2. Define the wrapper using the captured variable
        def forced_forward(model_self, *args, **kwargs):
            model_options = kwargs.get('model_options', {})
            t_options = model_options.get('transformer_options', {})
            base_context = t_options.get('context', None)
            
            # Use the captured variable, not 'self'
            alpha = min(current_ref_scale / 10.0, 1.0) 
            
            if base_context is not None and base_context.shape == full_context.shape:
                injected_context = (alpha * full_context) + ((1 - alpha) * base_context)
            else:
                injected_context = full_context

            if 'transformer_options' not in model_options:
                model_options['transformer_options'] = {}
            
            model_options['transformer_options']['context'] = injected_context
            kwargs['model_options'] = model_options
            
            return original_forward(*args, **kwargs)

        # 3. Bind
        model.forward = forced_forward.__get__(model, type(model))
        
        try:
            return super().sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            model.forward = original_forward

class STDRCContextInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
            }
        }

    # Output as CONDITIONING to be used by the Guider
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("reference_conditioning",)
    FUNCTION = "prepare_reference"
    CATEGORY = "ST-DRC/Preparation"

    def prepare_reference(self, reference_image, width, height):
        # Prepare reference image as a conditioning tensor
        ref_img = reference_image.permute(0, 3, 1, 2)
        ref_img = F.interpolate(ref_img, size=(height // 8, width // 8), mode="bilinear", align_corners=False)
        
        # Return as a list of conditioning to match ComfyUI standards
        # We package the latent into a format the Guider can extract
        return ([[ref_img, {}]],)


class TASSRoPEPatcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "spatial_shift": ("INT", {"default": 1000, "min": 0, "max": 5000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "ST-DRC/Patches"

    def patch_model(self, model, spatial_shift):
        patched_model = model.clone()
        
        def tass_rope_attention_patch(q, k, v, extra_options):
            spatial_shift_tensor = torch.tensor(spatial_shift, dtype=q.dtype, device=q.device)
            k_shifted = k * torch.cos(spatial_shift_tensor)
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
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0}),
                "reference_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "setup_guider"
    CATEGORY = "ST-DRC/Sampling"

    def setup_guider(self, model, positive_text, positive_reference, negative, cfg, reference_scale):
        # Simply instantiate the class defined above
        return (ThreeStreamGuider(model, positive_text, positive_reference, negative, cfg, reference_scale),)


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

