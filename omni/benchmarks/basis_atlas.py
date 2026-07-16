"""BASIS ATLAS -- Rung 7b (registered). Every tensor class x every recipe
x every spectral family.

TARGET LAW (registered as the prediction under test): EACH TENSOR CLASS
WRITES IN THE BASIS OF ITS OWN OPERATION -- selection/logic circuitry
(FFN expansion gates) in the dyadic family (Walsh, GEN_B), similarity
geometry (token embeddings) in the smooth family (DCT). The lead: on
GPT-2 the DCT margin exceeds Walsh on wte (232x vs 79x) and c_fc
(14.2x vs 12.7x); Kimi's embedding peaked in DCT (15.98x vs 9.67x).

DESIGN: for each model on the reference drive, one representative tensor
per class -- embedding / attention-Q / FFN expansion (gate or up) / FFN
contraction (down) -- at mid depth (expansion/attention/contraction) and
the token table (embedding); margin vs 3 seeded shuffle-nulls in each of
{Walsh, DCT-II ortho, Haar ortho}, same permutations across bases so the
comparison is paired. Expert tensors read expert 0 (the certified slice).
Slant-family arm registered as the follow-on, not silently dropped.

VERDICT RULE: per (model, class) the loudest basis is recorded; the
two-basis law is SUPPORTED at atlas level iff, across models with any
wake (>2x) in the class, embeddings are DCT-loudest in a majority AND
expansions are Walsh-loudest in a majority; any other pattern is the
finding instead. Margin clause: wake bar 2x, basis-preference calls only
on woken cells (a 1.0x-vs-1.1x preference is noise, not law).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import (FRACS, GGUF_LIB, Run, concentration, dct_ortho, fwht,
                       gpt2_path, haar_ortho, load_safetensor, slant_ortho)
from moe_dequant import dequant_expert, resolve_first_shard, tensor_map

SEED = 20260706
N = 1 << 22

REG = {
    "name": "basis-atlas-7b-round2-slant",
    "objects": ["one representative tensor per class (embedding / attn-Q / "
                "ffn-expansion / ffn-contraction) per on-drive model, expert-0 "
                "slices for MoE tensors"],
    "statistic": "paired margin vs 3 seeded shuffle-nulls in Walsh, DCT-II "
                 "(ortho), Haar (ortho); same permutations across bases",
    "verdict_rule": "two-basis law SUPPORTED iff woken embeddings are "
                    "DCT-loudest in a majority AND woken expansions are "
                    "Walsh-loudest in a majority; any other pattern recorded "
                    "as the finding",
    "margin_clause": "wake bar 2x; basis preference read only on cells with "
                     "at least one basis > 2x",
    "followon": "round 2: the slant arm live (verified: ramp 1-sparse, energy exact)",
}

BASES = (("walsh", fwht), ("dct", dct_ortho), ("haar", haar_ortho),
         ("slant", slant_ortho))


def paired_margins(v):
    """Margins in all three bases under the SAME null permutations."""
    v = v.astype(np.float64).ravel()
    n = min(N, 1 << int(np.floor(np.log2(len(v)))))
    v = v[:n].copy()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(n) for _ in range(3)]
    out = {}
    for bname, tf in BASES:
        real = concentration(tf(v.copy()), FRACS)[FRACS[0]]
        nm = max(concentration(tf(v[p].copy()), FRACS)[FRACS[0]] for p in perms)
        out[bname] = real / max(nm, 1e-12)
    return out


def gguf_class_tensors(first_shard):
    """Representative tensor names per class from a (possibly split) GGUF."""
    import re
    tmap = tensor_map(first_shard)
    names = list(tmap)
    blocks = sorted({int(m.group(1)) for n in names
                     for m in [re.match(r"blk\.(\d+)\.", n)] if m})
    mid = blocks[len(blocks) // 2] if blocks else 0

    def pick(*subs):
        cands = [n for n in names if all(s in n for s in subs) and n.endswith(".weight")]
        pref = [n for n in cands if n.startswith(f"blk.{mid}.")]
        return (pref or cands or [None])[0]

    return tmap, {
        "embedding": pick("token_embd"),
        "attention": pick("attn_q") or pick("attn_qkv"),
        "expansion": pick("ffn_gate") or pick("ffn_up"),
        "contraction": pick("ffn_down"),
    }


MODELS = (
    ("Qwen3.5-4B", f"{GGUF_LIB}/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Llama-3.1-8B", f"{GGUF_LIB}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("gemma-2-9b", f"{GGUF_LIB}/gemma-2-9b-it-Q4_K_M.gguf"),
    ("gpt-oss-20b", f"{GGUF_LIB}/gpt-oss-20b-Q4_K_M.gguf"),
    ("Qwen3.6-27B", f"{GGUF_LIB}/Qwen3.6-27B-Q4_K_M.gguf"),
    ("gemma-4-31B", f"{GGUF_LIB}/gemma-4-31B-it-Q4_K_M.gguf"),
    ("R1-Distill-32B", f"{GGUF_LIB}/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"),
    ("Llama-3.3-70B", f"{GGUF_LIB}/Llama-3.3-70B-Instruct-Q4_K_M.gguf"),
    ("DeepSeek-R1-671B", f"{GGUF_LIB}/DeepSeek-R1-Q4/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf"),
    ("Kimi-K2.6-1T", f"{GGUF_LIB}/Kimi-K2.6/UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf"),
)


def main():
    run = Run(REG)
    cells = []

    # GPT-2 first (safetensors; the anchor whose comparator lead started this)
    gp = gpt2_path()
    for cls, key in (("embedding", "wte.weight"),
                     ("attention", "h.6.attn.c_attn.weight"),
                     ("expansion", "h.6.mlp.c_fc.weight"),
                     ("contraction", "h.6.mlp.c_proj.weight")):
        m = paired_margins(load_safetensor(gp, key))
        best = max(m, key=m.get)
        cells.append(("GPT-2", cls, m, best))
        run.record(instrument="basis-atlas", model="GPT-2", tensor_class=cls,
                   object=key, **{f"{b}_margin": round(v, 3) for b, v in m.items()},
                   loudest=best, woken=bool(max(m.values()) > 2.0))
        print(f"  GPT-2            {cls:12s} walsh {m['walsh']:7.2f}  dct {m['dct']:7.2f}  "
              f"haar {m['haar']:7.2f}  slant {m['slant']:7.2f}   loudest={best}", flush=True)

    for model, path in MODELS:
        first = resolve_first_shard(path)
        if not first:
            run.record(instrument="basis-atlas", model=model, skipped="not found")
            print(f"  {model}: not found, skipped", flush=True)
            continue
        try:
            tmap, picks = gguf_class_tensors(first)
        except Exception as e:
            run.record(instrument="basis-atlas", model=model, skipped=str(e))
            print(f"  {model}: {e}", flush=True)
            continue
        for cls, name in picks.items():
            if not name:
                run.record(instrument="basis-atlas", model=model, tensor_class=cls,
                           skipped="no tensor of this class found")
                continue
            t, _ = tmap[name]
            try:
                w = dequant_expert(t, 0)
            except SystemExit:
                raise
            except Exception as e:
                run.record(instrument="basis-atlas", model=model, tensor_class=cls,
                           object=name, skipped=str(e))
                print(f"  {model} {cls}: {e}", flush=True)
                continue
            m = paired_margins(w)
            best = max(m, key=m.get)
            cells.append((model, cls, m, best))
            run.record(instrument="basis-atlas", model=model, tensor_class=cls,
                       object=name, **{f"{b}_margin": round(v, 3) for b, v in m.items()},
                       loudest=best, woken=bool(max(m.values()) > 2.0))
            print(f"  {model:16s} {cls:12s} walsh {m['walsh']:7.2f}  dct {m['dct']:7.2f}  "
                  f"haar {m['haar']:7.2f}  slant {m['slant']:7.2f}   loudest={best}", flush=True)

    # the registered verdict
    def tally(cls, basis):
        woken = [(mm, b) for mo, c, mm, b in cells if c == cls and max(mm.values()) > 2.0]
        hits = sum(1 for _, b in woken if b == basis)
        return hits, len(woken)

    e_hit, e_n = tally("embedding", "dct")
    x_hit, x_n = tally("expansion", "walsh")
    supported = e_n > 0 and x_n > 0 and e_hit * 2 > e_n and x_hit * 2 > x_n
    run.record(instrument="verdict", object="two-basis-law",
               embeddings_dct_loudest=f"{e_hit}/{e_n}",
               expansions_walsh_loudest=f"{x_hit}/{x_n}",
               supported=bool(supported))
    print(f"\nTWO-BASIS LAW: embeddings DCT-loudest {e_hit}/{e_n} woken; "
          f"expansions Walsh-loudest {x_hit}/{x_n} woken -> "
          f"{'SUPPORTED' if supported else 'NOT SUPPORTED AS STATED -- the pattern above is the finding'}",
          flush=True)
    print("BASIS ATLAS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
