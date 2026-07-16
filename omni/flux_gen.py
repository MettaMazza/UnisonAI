"""
FluxGenerator — Local image generation via libstable-diffusion.dylib (ctypes FFI).

Follows the same pattern as ErnodDecent's image_gen.ep:
- Uses the Flux GGUF model + separate CLIP-L, T5XXL, and VAE encoders
- All weights live on the external SSD (/Volumes/One Touch/models library/)
- The dylib at ~/.ernosdecent/lib/libstable-diffusion.dylib does the heavy lifting
- CFG forced to 1 for Flux (guidance-distilled model)

Generated images are returned as raw PNG bytes for encoding via ImageEncoder
and banking into the SynapticGraph.
"""
import os
import ctypes
import time
import tempfile
from omni.logging_config import get_logger

flux_logger = get_logger("OmniFlux", "flux.log")

# ── Model paths (matching ErnodDecent's config/image.json) ──
_SSD_BASE = "/Volumes/One Touch/models library"
_FLUX_CONFIG = {
    "diffusion_model": f"{_SSD_BASE}/Creative_Models/flux1-dev-Q4_K_S.gguf",
    "clip_l": f"{_SSD_BASE}/huggingface/hub/models--lzyvegetable--FLUX.1-schnell/snapshots/cb2d0f958483a378c26558405a06224a6889b9f4/text_encoder/model.safetensors",
    "t5xxl": f"{_SSD_BASE}/Creative_Models/flux_encoders/t5xxl-Q8_0.gguf",
    "vae": f"{_SSD_BASE}/huggingface/hub/models--lzyvegetable--FLUX.1-schnell/snapshots/cb2d0f958483a378c26558405a06224a6889b9f4/vae/diffusion_pytorch_model.safetensors",
}

_DYLIB_PATH = os.path.expanduser("~/.ernosdecent/lib/libstable-diffusion.dylib")

# Output directory for generated images
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_images")


class FluxGenerator:
    """
    Local Flux image generation via the stable-diffusion.cpp dylib.
    
    Same FFI pattern as ErnodDecent: load the dylib, call sd_ep_generate
    with Flux mode (empty model_path, separate diffusion+encoders).
    """
    def __init__(self, width=512, height=512, steps=20, seed=-1):
        self.width = width
        self.height = height
        self.steps = steps
        self.seed = seed  # -1 = random
        self.dylib = None
        self.available = False
        
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        self._load_dylib()
    
    def _load_dylib(self):
        """Attempt to load the libstable-diffusion dylib."""
        if not os.path.exists(_DYLIB_PATH):
            flux_logger.warning(
                f"libstable-diffusion.dylib not found at {_DYLIB_PATH}. "
                f"Image generation disabled."
            )
            return
        
        try:
            self.dylib = ctypes.CDLL(_DYLIB_PATH)
            self.available = True
            flux_logger.info(f"Loaded libstable-diffusion from {_DYLIB_PATH}")
        except Exception as e:
            flux_logger.error(f"Failed to load dylib: {e}")
    
    def _preflight(self):
        """Check all required model files exist."""
        missing = []
        for name, path in _FLUX_CONFIG.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            flux_logger.error(f"Missing model files: {missing}")
            return False, missing
        return True, []
    
    def generate(self, prompt, width=None, height=None, steps=None, seed=None):
        """
        Generate an image from a text prompt.
        
        Returns (success: bool, png_bytes or error_message: str).
        """
        if not self.available:
            return False, "Image generation not available (dylib not loaded)"
        
        ok, missing = self._preflight()
        if not ok:
            return False, f"Missing model files: {missing}"
        
        w = width or self.width
        h = height or self.height
        s = steps or self.steps
        sd = seed if seed is not None else self.seed
        
        # Generate unique output filename
        timestamp = int(time.time() * 1000)
        out_name = f"flux_{timestamp}.png"
        out_path = os.path.join(_OUTPUT_DIR, out_name)
        
        flux_logger.info(
            f"Generating: prompt=\"{prompt[:80]}\" "
            f"size={w}x{h} steps={s} seed={sd}"
        )
        
        t0 = time.perf_counter()
        
        try:
            # Call the dylib — matching ErnodDecent's FFI signature:
            # sd_ep_generate(model_path, diffusion_model, clip_l, t5xxl, vae,
            #                prompt, negative, width, height, steps, cfg, seed, out_path)
            # For Flux: model_path="" (empty), cfg=1 (guidance-distilled)
            
            func = self.dylib.sd_ep_generate
            func.restype = ctypes.c_int
            func.argtypes = [
                ctypes.c_char_p,  # model_path
                ctypes.c_char_p,  # diffusion_model
                ctypes.c_char_p,  # clip_l
                ctypes.c_char_p,  # t5xxl
                ctypes.c_char_p,  # vae
                ctypes.c_char_p,  # prompt
                ctypes.c_char_p,  # negative
                ctypes.c_int,     # width
                ctypes.c_int,     # height
                ctypes.c_int,     # steps
                ctypes.c_int,     # cfg
                ctypes.c_int,     # seed
                ctypes.c_char_p,  # out_png_path
            ]
            
            rc = func(
                b"",  # model_path empty for Flux
                _FLUX_CONFIG["diffusion_model"].encode(),
                _FLUX_CONFIG["clip_l"].encode(),
                _FLUX_CONFIG["t5xxl"].encode(),
                _FLUX_CONFIG["vae"].encode(),
                prompt.encode("utf-8"),
                b"",  # no negative prompt
                w, h, s,
                1,    # cfg=1 for Flux (guidance-distilled)
                sd,
                out_path.encode(),
            )
            
            elapsed_ms = (time.perf_counter() - t0) * 1000
            
            if rc == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    png_bytes = f.read()
                
                flux_logger.info(
                    f"Generation SUCCESS: {out_path} | "
                    f"{len(png_bytes)} bytes | {elapsed_ms:.0f}ms"
                )
                return True, png_bytes
            else:
                reason = self._rc_reason(rc)
                flux_logger.error(f"Generation FAILED: rc={rc} ({reason}) after {elapsed_ms:.0f}ms")
                return False, f"Flux generation failed (rc={rc}): {reason}"
                
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            flux_logger.error(f"Generation EXCEPTION after {elapsed_ms:.0f}ms: {e}", exc_info=True)
            return False, f"Exception: {e}"
    
    def _rc_reason(self, rc):
        """Map shim return codes to human-readable reasons (matching ErnodDecent)."""
        reasons = {
            2: "bad arguments (empty model/prompt/output path)",
            3: "model context failed to LOAD — check paths, is external drive mounted?",
            4: "pipeline ran but produced no image",
            5: "PNG write failed (output path/disk issue)",
            99: "image runtime not built (dylib missing at build time)",
        }
        return reasons.get(rc, f"unknown shim code {rc}")
    
    def list_generated(self):
        """List all generated images."""
        if not os.path.isdir(_OUTPUT_DIR):
            return []
        return sorted([
            f for f in os.listdir(_OUTPUT_DIR) if f.endswith(".png")
        ])
