import numpy as np, sys
sys.path.insert(0, ".")
from rung2f_rowblock import probe_rowblocks
import gguf
from gguf.quants import dequantize
LIB = "/Volumes/One Touch/models library/GGUF_Models"
for label, path in (("R1-DISTILL-Qwen-32B (reasoning)", f"{LIB}/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"),
                    ("qwen2.5-coder-32b (sibling)", f"{LIB}/qwen2.5-coder-32b-instruct-q4_k_m.gguf")):
    r = gguf.GGUFReader(path)
    cands = [t for t in r.tensors if ("ffn_gate" in t.name or "ffn_up" in t.name)
             and t.name.endswith(".weight") and "norm" not in t.name and "inp" not in t.name]
    for t in (cands[0], cands[len(cands)//2], cands[-1]):
        w = np.asarray(dequantize(t.data, t.tensor_type)).ravel()
        shape = tuple(int(x) for x in t.shape)
        w = w[:shape[-2]*shape[-1]].reshape(shape[-2], shape[-1])
        probe_rowblocks(f"{label} {t.name}", w)
print("RUNG 2h COMPLETE", flush=True)
