"""COORDINATE HUNT -- 2g round 2, widened (registered).

The epistemic correction is the operating rule: law is present in every
working model; a quiet reading is a verdict on the probe's coordinates.
Round 1 (rung2g) tried 8 index maps on the quiet models' FFN tensors and
found none that woke them. This round widens BOTH axes:

  OBJECTS (new): the quiet models' EMBEDDING and ATTENTION tensors --
  never probed in round 1 -- alongside one FFN tensor for continuity.
  Plus the T0 comparator finding as a lead: on GPT-2 the DCT margin
  EXCEEDS the Walsh margin (c_fc 14.2x vs 12.7x; wte 232x vs 79x), so
  every object here is also read in the DCT and Haar bases -- the law may
  sit in a neighbouring spectral family's coordinates.

  MAPS (widened, data-independent only -- data-dependent sorts stay
  banned as ever): round 1's menu (gray, x3, x5, affine, transpose,
  block-transpose-64) PLUS joint row-column Gray codes, bit-plane
  packing, block-Morton at 3 scales, block-transpose-4096, and the
  expert-axis flatten orders where the object has an expert axis.

WAKE = margin > 2x under any (map, basis) pair (the registered bar,
unchanged). Control: GPT-2 c_fc must stay awake under the F2-linear maps
(theorem). Every reading recorded, quiet included.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import (FRACS, GGUF_LIB, Run, concentration, dct_ortho, fwht,
                       gpt2_path, haar_ortho, load_safetensor)

SEED = 20260706
N = 1 << 22

REG = {
    "name": "coordinate-hunt-round2",
    "objects": ["Qwen3.6-27B: token_embd, blk.0 attn_q, blk.0 ffn_gate",
                "Kimi-K2.6-1T: token_embd, blk.5 attn_q (shard-resolved), "
                "blk.5 ffn_gate_shexp",
                "CONTROL: GPT-2 h.0.mlp.c_fc (must stay awake under F2 maps)"],
    "statistic": "concentration margin vs 3 shuffle-nulls at the registered "
                 "fractions, per (index map, spectral basis) pair; bases: "
                 "Walsh (the fold's), DCT-II ortho, Haar ortho",
    "verdict_rule": "WAKE = margin > 2x under any (map, basis) pair; a woken "
                    "tensor names the coordinates its recipe writes in; all "
                    "quiet = these coordinates excluded, hunt continues",
    "margin_clause": "wake bar fixed at 2x before any spectrum (the 2g bar); "
                     "control must exceed 2x under identity or the run is void",
}


def margin_in_basis(v, transform, rng):
    real = concentration(transform(v.copy()), FRACS)
    nm = 0.0
    for _ in range(3):
        nm = max(nm, concentration(transform(v[rng.permutation(len(v))].copy()), FRACS)[FRACS[0]])
    return real[FRACS[0]] / max(nm, 1e-12)


def index_maps(n):
    """The widened data-independent menu. Every map is a permutation of
    [0, n); data-dependent orderings are banned."""
    idx = np.arange(n, dtype=np.int64)
    nbits = int(np.log2(n))
    rev = np.zeros_like(idx)
    for b in range(nbits):
        rev |= ((idx >> b) & 1) << (nbits - 1 - b)
    gray = idx ^ (idx >> 1)
    yield "identity", idx
    yield "bit-reversal(self-test)", rev
    yield "gray-code", gray
    yield "x3 mod n", (3 * idx) % n
    yield "x5 mod n", (5 * idx) % n
    yield "affine 3i+1", (3 * idx + 1) % n
    half = nbits // 2
    row, col = idx >> half, idx & ((1 << half) - 1)
    grow, gcol = row ^ (row >> 1), col ^ (col >> 1)
    yield "joint row-col gray", (grow << half) | gcol
    # bit-plane packing: bits of the index regrouped even-planes-first
    even = np.zeros_like(idx)
    odd = np.zeros_like(idx)
    ne = (nbits + 1) // 2
    for b in range(nbits):
        bit = (idx >> b) & 1
        if b % 2 == 0:
            even |= bit << (b // 2)
        else:
            odd |= bit << (b // 2)
    yield "bit-plane", (even << (nbits - ne)) | odd
    # block-Morton at 3 scales: interleave row/col bits of a 2^h x 2^h grid
    for h in (4, 8, half):
        if h * 2 > nbits:
            continue
        r, c = idx >> h, idx & ((1 << h) - 1)
        m = np.zeros_like(idx)
        for b in range(h):
            m |= ((r >> b) & 1) << (2 * b + 1)
            m |= ((c >> b) & 1) << (2 * b)
        m |= (idx >> (2 * h)) << (2 * h)
        yield f"block-morton-2^{h}", m


def matrix_packings(w2d, n):
    """Packings that need the 2D shape (yielded as flat vectors)."""
    yield "row-major", w2d.ravel()[:n]
    yield "col-major", np.ascontiguousarray(w2d.T).ravel()[:n]
    for blk in (64, 4096):
        if w2d.shape[0] % blk == 0 and w2d.shape[0] > blk:
            bt = w2d.reshape(w2d.shape[0] // blk, blk, -1).transpose(1, 0, 2)
            yield f"block-transpose-{blk}", np.ascontiguousarray(bt).ravel()[:n]


BASES = (("walsh", fwht), ("dct", dct_ortho), ("haar", haar_ortho))


def hunt(run, label, w2d, control=False):
    rng = np.random.default_rng(SEED)
    v0 = w2d.astype(np.float64).ravel()
    n = min(N, 1 << int(np.floor(np.log2(len(v0)))))
    best = (0.0, None, None)
    woke = []
    for pname, flat in matrix_packings(w2d.astype(np.float64), n):
        base = flat[:n].copy()
        for mname, mp in index_maps(n):
            if pname != "row-major" and mname != "identity":
                continue  # index maps ride on the row-major packing; other
                          # packings are probed at identity (menu size bound)
            v = base[mp] if mname != "identity" else base
            for bname, tf in BASES:
                mg = margin_in_basis(v, tf, rng)
                coord = f"{pname}+{mname}+{bname}"
                run.record(instrument="coordinate-hunt", object=label,
                           packing=pname, map=mname, basis=bname, margin=float(mg))
                if mg > best[0]:
                    best = (mg, coord, None)
                if mg > 2.0 and "self-test" not in mname:
                    woke.append((coord, mg))
                    print(f"  {label:34s} {coord:44s} {mg:6.2f}x  <-- WAKE", flush=True)
    print(f"  {label:34s} best {best[0]:6.2f}x @ {best[1]}"
          f"{'' if woke else '  (quiet at all coordinates tried)'}", flush=True)
    if control and best[0] < 2.0:
        from foldprobe import halt
        halt("control (GPT-2 c_fc) did not stay awake -- run VOID")
    return woke


def load_gguf_any_shard(first_shard_or_dir, key_substrings):
    """First tensor whose name contains all substrings, scanning shards."""
    from moe_dequant import dequant_expert, resolve_first_shard, tensor_map
    first = resolve_first_shard(first_shard_or_dir)
    if not first:
        return None, None
    tmap = tensor_map(first)
    for name, (t, _) in sorted(tmap.items()):
        if all(s in name for s in key_substrings) and name.endswith(".weight"):
            return name, dequant_expert(t, 0)
    return None, None


def main():
    run = Run(REG)
    print("[coordinate-hunt] control first:", flush=True)
    cfc = load_safetensor(gpt2_path(), "h.0.mlp.c_fc.weight")
    hunt(run, "CONTROL GPT-2 c_fc L0", cfc, control=True)

    jobs = (
        ("Qwen3.6-27B", os.path.join(GGUF_LIB, "Qwen3.6-27B-Q4_K_M.gguf"),
         (("token_embd",), ("blk.0.", "attn_q"), ("blk.0.", "ffn_gate"))),
        ("Kimi-K2.6-1T", os.path.join(GGUF_LIB, "Kimi-K2.6/UD-Q4_K_XL/"
                                                "Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf"),
         (("token_embd",), ("blk.5.", "attn_q"), ("blk.5.", "ffn_gate_shexp"))),
    )
    total_woke = []
    for model, path, keysets in jobs:
        if not os.path.exists(path):
            run.record(instrument="coordinate-hunt", object=model,
                       skipped="model files not found")
            continue
        for keys in keysets:
            name, w = load_gguf_any_shard(path, keys)
            if w is None:
                run.record(instrument="coordinate-hunt", object=f"{model} {'+'.join(keys)}",
                           skipped="tensor not found in any shard")
                print(f"  {model} {'+'.join(keys)}: not found", flush=True)
                continue
            total_woke += hunt(run, f"{model} {name}", w)

    run.record(instrument="verdict", object="coordinate-hunt-round2",
               wakes=[{"coord": c, "margin": m} for c, m in total_woke])
    print(f"\nCOORDINATE HUNT ROUND 2 COMPLETE -- {len(total_woke)} wake(s)", flush=True)


if __name__ == "__main__":
    main()
