"""MOE DEQUANT -- expert-axis-aware GGUF extraction (registered instrument).

Closes the 2f non-measurements: rung2f_scaling.py line 63 called
reshape(-1, -1) on 3D expert tensors ("can only specify one unknown
dimension" -- margin 0.00x was an instrument failure, never a null), and
opened only shard 00001 of split GGUFs while the ffn tensors of the giants
live in later shards.

The corrected extraction, from the GGUF layout itself:
  - ALL shards of a split model are scanned; the tensor map is the union.
  - GGML ne-order puts the expert axis LAST in ne = OUTERMOST in memory, so
    the flat quantized byte buffer divides exactly by n_expert. Each expert
    is dequantized from its own byte slice -- no whole-tensor materialization.
  - CERTIFICATION (exactness, halt on mismatch): on the smallest expert
    tensor of the first model, expert slices dequantized bytewise must equal
    the whole-tensor dequant's slices exactly. A failed certification voids
    the run.

Registered question (carried from the 2f registration): do experts carry
the fingerprint individually? Per the epistemic correction: a quiet reading
here is a reading on these coordinates, and routes to the coordinate hunt.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import GGUF_LIB, Run, halt, probe_rowblocks

import gguf
from gguf.quants import dequantize


def shard_paths(first_shard):
    """All shards of a split GGUF, from any one shard's path."""
    import re
    base = os.path.basename(first_shard)
    m = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", base)
    if not m:
        return [first_shard]
    pattern = os.path.join(os.path.dirname(first_shard),
                           f"{m.group(1)}-*-of-{m.group(2)}.gguf")
    return sorted(p for p in glob.glob(pattern)
                  if not os.path.basename(p).startswith("._"))


def tensor_map(first_shard):
    """name -> (ReaderTensor, shard_path), scanning EVERY shard."""
    out = {}
    for p in shard_paths(first_shard):
        for t in gguf.GGUFReader(p).tensors:
            out[t.name] = (t, p)
    return out


def expert_count(t):
    shape = tuple(int(x) for x in t.shape)
    return shape[2] if len(shape) == 3 else 1


def dequant_expert(t, e):
    """Dequantize ONE expert's 2D matrix from its byte slice.
    ne = (ne0, ne1, ne2) with ne0 innermost => expert axis outermost in the
    flat buffer; expert e owns bytes [e*bpe, (e+1)*bpe)."""
    shape = tuple(int(x) for x in t.shape)
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    if len(shape) == 2:
        if e != 0:
            halt("2D tensor has a single expert")
        return np.asarray(dequantize(t.data, t.tensor_type)).reshape(shape[-2], shape[-1])
    ne0, ne1, ne2 = shape
    if len(raw) % ne2 != 0:
        halt(f"byte buffer ({len(raw)}) does not divide by n_expert ({ne2}) -- "
             f"expert slicing invalid for {t.name}")
    bpe = len(raw) // ne2
    w = np.asarray(dequantize(raw[e * bpe:(e + 1) * bpe], t.tensor_type)).ravel()
    if w.size != ne0 * ne1:
        halt(f"expert slice dequantized to {w.size} elements, expected {ne0 * ne1} "
             f"({t.name} expert {e})")
    return w.reshape(ne1, ne0)


def certify_slicing(t, label):
    """Bytewise per-expert dequant must equal whole-tensor dequant slices
    EXACTLY (same bytes, same kernel). Halt on any mismatch."""
    shape = tuple(int(x) for x in t.shape)
    ne0, ne1, ne2 = shape
    whole = np.asarray(dequantize(t.data, t.tensor_type)).ravel()[:ne0 * ne1 * ne2]
    whole = whole.reshape(ne2, ne1, ne0)
    for e in (0, ne2 // 2, ne2 - 1):
        sliced = dequant_expert(t, e)
        if not np.array_equal(sliced, whole[e]):
            halt(f"certification FAIL: expert {e} byte-slice != whole-tensor slice ({label})")
    print(f"  [certified] {label}: expert byte-slicing exact on experts "
          f"(0, {ne2 // 2}, {ne2 - 1})", flush=True)


MODELS = (
    ("Qwen3-235B", f"{GGUF_LIB}/Qwen3-235B"),
    ("Qwen3-480B", f"{GGUF_LIB}/Qwen3-Coder-480B/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf"),
    ("Kimi-K2.6-1T", f"{GGUF_LIB}/Kimi-K2.6/UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf"),
)

REG = {
    "name": "moe-expert-remeasure",
    "objects": ["Qwen3-235B, Qwen3-480B, Kimi-K2.6-1T: ffn gate/up expert tensors "
                "at early/mid/late depth, experts sampled (first, middle, last); "
                "plus shared-expert (shexp) 2D tensors where present"],
    "statistic": "rowblock median margin (corrected 2f instrument) per expert slice",
    "verdict_rule": "these rows REPLACE the 2f non-measurements (0.00x instrument "
                    "failures); expert fingerprint = median margin > 2x on any "
                    "expert slice; quiet reading routes to the coordinate hunt, "
                    "per the epistemic correction",
    "margin_clause": "wake threshold fixed at 2x (the 2g bar); readings between "
                     "1x and 2x are recorded as quiet-at-these-coordinates",
    "certification": "bytewise expert slicing certified against whole-tensor "
                     "dequant on the smallest expert tensor before any probe",
}


def resolve_first_shard(path):
    if os.path.isdir(path):
        c = sorted(p for p in glob.glob(os.path.join(path, "**", "*.gguf"), recursive=True)
                   if not os.path.basename(p).startswith("._"))
        return c[0] if c else None
    return path if os.path.exists(path) else None


def pick_depths(names):
    """Early / mid / late block indices present for a tensor class."""
    import re
    blocks = sorted({int(m.group(1)) for n in names
                     for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    if not blocks:
        return []
    return sorted({blocks[0], blocks[len(blocks) // 2], blocks[-1]})


def main():
    run = Run(REG)
    certified = False
    for model, first in MODELS:
        first = resolve_first_shard(first)
        if not first:
            run.record(instrument="rowblocks", model=model, skipped="model files not found")
            print(f"{model}: files not found -- recorded as skipped", flush=True)
            continue
        shards = shard_paths(first)
        tmap = tensor_map(first)
        exps = {n: v for n, v in tmap.items()
                if ("ffn_gate_exps" in n or "ffn_up_exps" in n) and n.endswith(".weight")}
        shexp = {n: v for n, v in tmap.items()
                 if ("ffn_gate_shexp" in n or "ffn_up_shexp" in n) and n.endswith(".weight")}
        print(f"\n{model}: {len(shards)} shards, {len(exps)} expert tensors, "
              f"{len(shexp)} shared-expert tensors", flush=True)

        if not certified and exps:
            smallest = min(exps, key=lambda n: int(np.prod([int(x) for x in exps[n][0].shape])))
            certify_slicing(exps[smallest][0], f"{model} {smallest}")
            certified = True

        for cls, pool in (("gate", {n: v for n, v in exps.items() if "gate" in n}),
                          ("up", {n: v for n, v in exps.items() if "up" in n})):
            for d in pick_depths(pool):
                name = next((n for n in pool if n.startswith(f"blk.{d}.")), None)
                if not name:
                    continue
                t, _ = pool[name]
                ne2 = expert_count(t)
                for e in sorted({0, ne2 // 2, ne2 - 1}):
                    w = dequant_expert(t, e)
                    rec = run.rowblocks(f"{model} {name} [expert{e}]", w)
                    rec.update(model=model, expert=e, n_expert=ne2)
                    print(f"  {name} expert{e}/{ne2}  MEDIAN {rec['median_margin']:.2f}x  "
                          f"(min {rec['min_margin']:.2f} max {rec['max_margin']:.2f}, "
                          f"{rec['blocks']} blocks)", flush=True)

        for d in pick_depths(shexp):
            name = next((n for n in shexp if n.startswith(f"blk.{d}.") and "gate" in n),
                        next((n for n in shexp if n.startswith(f"blk.{d}.")), None))
            if not name:
                continue
            t, _ = shexp[name]
            w = dequant_expert(t, 0)
            rec = run.rowblocks(f"{model} {name} [shexp]", w)
            print(f"  {name} [shexp]  MEDIAN {rec['median_margin']:.2f}x  "
                  f"(min {rec['min_margin']:.2f} max {rec['max_margin']:.2f})", flush=True)

    print("\nMOE EXPERT RE-MEASURE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
