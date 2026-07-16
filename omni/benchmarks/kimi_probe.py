import numpy as np, sys
sys.path.insert(0, ".")
from rung2f_rowblock import probe_rowblocks
import gguf
from gguf.quants import dequantize
p = "/Volumes/One Touch/models library/GGUF_Models/Kimi-K2.6/UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL-00002-of-00014.gguf"
r = gguf.GGUFReader(p)
cands = [t for t in r.tensors if ("ffn_gate" in t.name or "ffn_up" in t.name)
         and t.name.endswith(".weight") and "norm" not in t.name and "inp" not in t.name]
for t in (cands[0], cands[len(cands)//2], cands[-1]):
    w = np.asarray(dequantize(t.data, t.tensor_type)).ravel()
    shape = tuple(int(x) for x in t.shape)
    if len(shape) == 3:
        per = shape[1] * shape[2]
        w = w[:per].reshape(shape[1], shape[2])
    else:
        w = w.reshape(shape[-2], shape[-1])
    probe_rowblocks(f"Kimi-K2.6-1T {t.name}", w)
print("KIMI COMPLETE", flush=True)
