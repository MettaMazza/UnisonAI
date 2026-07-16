"""CHECKPOINT TELESCOPE -- Rung 7a (registered). The deposition curve from
PRODUCTION training runs, nothing trained here.

OBJECT: EleutherAI's Pythia suite publishes full checkpoints of real
training runs, with the early checkpoints on an exactly dyadic clock
(step 1, 2, 4, ..., 512) -- the fold's own sampling of the training axis,
published by the incumbent paradigm itself. This instrument walks the
ladder of pythia-70m (extendable by MODEL/REVS), runs the locked battery
(with comparator bases) on the loud-class tensors at every checkpoint,
and writes margin(step) to the ledger.

QUESTIONS (registered): (1) the deposition curve at production scale --
when does the law arrive in real training? (2) does the twin-factory
reading (structure present early) hold off the toy? (3) does the
embedding's DCT-loudness (the 7b lead) EMERGE over training or exist
from initialization? Step 0 is the He-init analogue: it must read at
null or the run is suspect (the instrument's own negative control,
built into the ladder).

Checkpoints are downloaded one at a time to a scratch cache and deleted
after probing (IO discipline; ~200 MB each, never more than one on disk).
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run, battery

CACHE = os.path.expanduser("~/.cache/foldprobe_telescope")

# ladder configs: (hf repo, mlp layers to probe, checkpoint steps)
# 70m ran the full dyadic ladder (committed 2026-07-14); the scaling rungs
# sample around the 70m peak (~step 4000) to locate peak(scale).
LADDER = {
    "pythia-70m": ("EleutherAI/pythia-70m", (0, 2, 5),
                   [0] + [1 << k for k in range(10)] + [1000, 4000, 16000, 64000, 143000]),
    "pythia-410m": ("EleutherAI/pythia-410m", (0, 11, 23),
                    [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]),
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", (0, 11, 23),
                    [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 64000, 143000]),
    "pythia-1.4b-onset": ("EleutherAI/pythia-1.4b", (0, 11, 23),
                          [32, 64, 128, 256, 512]),
}
WHICH = sys.argv[1] if len(sys.argv) > 1 else "pythia-70m"
MODEL, LAYERS, REVS = LADDER[WHICH]

REG = {
    "name": f"checkpoint-telescope-7a-{WHICH}",
    "objects": [f"{MODEL} public checkpoints at steps {REVS}: embed_in + "
                f"mlp.dense_h_to_4h at layers {LAYERS} (the loud class)"],
    "statistic": "locked battery margin (3 shuffle-nulls) + DCT/Haar/slant "
                 "comparator margins per tensor per checkpoint",
    "verdict_rule": "the deposition curve margin(step) is the finding; step-0 "
                    "must sit at null (built-in negative control) or the run "
                    "is void; across ladder rungs: does the deposition peak "
                    "step move with model scale, and by what law?",
    "margin_clause": "wake bar 2x (the standing bar); step-0 control null = "
                     "no fraction beyond both nulls on any tensor",
}

TENSORS = (["gpt_neox.embed_in.weight"] +
           [f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight" for i in LAYERS])


def fetch_state(rev):
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=f"step{rev}", cache_dir=CACHE,
                             allow_patterns=["*.bin", "*.safetensors", "config.json"])
    sd = None
    for f in os.listdir(path):
        fp = os.path.join(path, f)
        if f.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(fp)
            break
        if f.endswith(".bin"):
            sd = torch.load(fp, map_location="cpu", weights_only=True)
            break
    if sd is None:
        raise RuntimeError(f"no weights file in snapshot step{rev}")
    return sd


def main():
    run = Run(REG)
    step0_clean = None
    for rev in REVS:
        try:
            sd = fetch_state(rev)
        except Exception as e:
            run.record(instrument="telescope", model=MODEL, step=rev,
                       skipped=f"fetch failed: {e}")
            print(f"  step {rev}: fetch failed ({e})", flush=True)
            continue
        for name in TENSORS:
            if name not in sd:
                run.record(instrument="telescope", model=MODEL, step=rev,
                           object=name, skipped="tensor absent")
                continue
            v = sd[name].float().numpy().ravel()
            rec = battery(v, n_shuffle=3, comparators=True)
            run.record(instrument="telescope", model=MODEL, step=rev,
                       object=name, margin=rec["margin"],
                       dct_margin=rec["dct_margin"], haar_margin=rec["haar_margin"],
                       beyond=sum(rec["beyond_nulls"]))
            print(f"  step {rev:6d}  {name.split('.')[-2][:12]:12s} "
                  f"walsh {rec['margin']:6.2f}x  dct {rec['dct_margin']:6.2f}x  "
                  f"haar {rec['haar_margin']:6.2f}x  beyond {sum(rec['beyond_nulls'])}/3",
                  flush=True)
            if rev == 0 and name == TENSORS[0]:
                step0_clean = sum(rec["beyond_nulls"]) == 0
        # IO discipline: one checkpoint on disk at a time
        shutil.rmtree(CACHE, ignore_errors=True)
    run.record(instrument="verdict", object="telescope-step0-control",
               step0_embed_null=step0_clean)
    print(f"\nstep-0 control (embed at null): "
          f"{'CLEAN' if step0_clean else 'NOT CLEAN -- investigate' if step0_clean is not None else 'not read'}",
          flush=True)
    print("CHECKPOINT TELESCOPE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
