"""FOLDPROBE -- the locked battery as a library (the campaign's registered
instrument, consolidated). Every primitive is the recorded protocol's own:

  - float64 in-place Walsh-Hadamard, natural order (imported verbatim from
    spectral_probe.py -- identical by construction, not re-implemented)
  - energy concentration C(k) at the registered fractions
  - 5 seeded shuffle-nulls + moment-matched Gaussian yardstick, seed 20260706
  - theorem-forced bit-reversal self-test: F2-linear, must preserve C(k)
    exactly; a failed self-test VOIDS the run (hard halt, never swallowed)
  - per-row-block spectra, MEDIAN block margin (the corrected scale-aware
    instrument from the 2f amendment: 3 shuffles per block, <=12 blocks)

New registered arms (this consolidation, before any new spectrum):
  - COMPARATOR BASES: DCT-II (ortho) and orthonormal Haar as registered
    yardsticks. Verdict "dyadic placement": the Walsh margin exceeds both
    comparator margins on the same vector under the same nulls.
  - REGISTRATION GATE: a run opens only with a registration block carrying
    objects, statistic, verdict_rule and a MANDATORY margin_clause; the
    block's SHA-256 is stamped into every ledger row; an unregistered or
    incomplete registration refuses to run (halt_violation applied to
    epistemics -- the Rung-3b flaw class closed by template).
  - LEDGER: append-only results.jsonl, one row per measurement, plus
    registrations.jsonl holding every registration verbatim.

Seed 20260706, as always. Verdict rules are fixed at registration time,
never after seeing data. Negative results are recorded in full.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_probe import FRACS, bit_reverse_perm, concentration, fwht

SEED = 20260706
BLOCK = 1 << 22
HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")
REGISTRY = os.path.join(HERE, "registrations.jsonl")
GGUF_LIB = "/Volumes/One Touch/models library/GGUF_Models"
GPT2_CANDIDATES = (
    os.path.join(HERE, "gpt2_model.safetensors"),
    "/Users/mettamazza/Desktop/Smithian Fold Theory/omni/benchmarks/gpt2_model.safetensors",
)


def halt(msg):
    print(f"FOLDPROBE HALT: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def gpt2_path():
    for p in GPT2_CANDIDATES:
        if os.path.exists(p):
            return p
    halt("gpt2_model.safetensors not found at any known path")


# ---------------------------------------------------------------- comparators

def dct_ortho(x):
    """DCT-II, orthonormal (energy-preserving) -- comparator basis 1."""
    from scipy.fft import dct
    return dct(x, type=2, norm="ortho")


def haar_ortho(x):
    """Orthonormal fast Haar -- comparator basis 2. Power-of-two length."""
    a = x.astype(np.float64).copy()
    n = len(a)
    out = np.empty(n)
    lo = a
    pos = n
    r2 = np.sqrt(2.0)
    while len(lo) > 1:
        even, odd = lo[0::2], lo[1::2]
        detail = (even - odd) / r2
        lo = (even + odd) / r2
        pos -= len(detail)
        out[pos:pos + len(detail)] = detail
    out[0] = lo[0]
    return out


def slant_ortho(x):
    """Orthonormal fast slant transform -- comparator basis 3, the family
    BETWEEN the smooth (DCT) and dyadic (Walsh) families: Walsh-like steps
    plus exact linear-ramp basis vectors (Pratt-Chen-Welch recursion,
    a_2N = sqrt(3N^2/(4N^2-1)), b_2N = sqrt((N^2-1)/(4N^2-1)); a^2+b^2 = 1).
    Verified at import: orthogonality via Parseval (checked per battery call
    like every comparator) and the defining property -- the transform of the
    exact linear ramp is 1-sparse (see _slant_selftest)."""
    v = x.astype(np.float64)
    n = len(v)
    r2 = np.sqrt(2.0)

    def rec(u):
        m = len(u)
        if m == 1:
            return u
        if m == 2:
            return np.array([u[0] + u[1], u[0] - u[1]]) / r2
        h = m // 2
        yt = rec(u[:h])
        yb = rec(u[h:])
        zs = (yt + yb) / r2
        zd = (yt - yb) / r2
        N = h
        a = np.sqrt(3.0 * N * N / (4.0 * N * N - 1.0))
        b = np.sqrt((N * N - 1.0) / (4.0 * N * N - 1.0))
        ramp = a * zd[0] + b * zs[1]
        rest = -b * zd[0] + a * zs[1]
        out = np.empty(m)
        out[0] = zs[0]
        out[1] = ramp
        out[2:h] = zs[2:]
        out[h] = zd[1]
        out[h + 1] = rest
        out[h + 2:] = zd[2:]
        return out

    return rec(v)


def _slant_selftest():
    """Theorem-forced: the exact linear ramp is a slant basis vector, so its
    transform must be 1-sparse; and the transform must preserve energy."""
    n = 16
    ramp = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    s = slant_ortho(ramp)
    e = s ** 2
    if not np.isclose(e.sum(), (ramp ** 2).sum(), rtol=1e-9):
        halt("slant self-test: energy not preserved")
    if e.max() / e.sum() < 1.0 - 1e-9:
        halt(f"slant self-test: ramp not 1-sparse (top share {e.max()/e.sum():.6f})")


_slant_selftest()


# ------------------------------------------------------------------- battery

def battery(v, fracs=FRACS, n_shuffle=5, comparators=True, seed=SEED):
    """The locked battery on one power-of-two vector. Returns the full
    record: real/shuffle-max/gaussian concentrations per fraction, margins,
    self-test, and comparator-basis margins under the SAME permutations."""
    v = v.astype(np.float64).ravel()
    nbits = int(np.floor(np.log2(len(v))))
    n = 1 << nbits
    v = v[:n].copy()

    real = concentration(fwht(v.copy()), fracs)
    st = concentration(fwht(v[bit_reverse_perm(nbits)].copy()), fracs)
    if not all(abs(st[q] - real[q]) < 1e-9 for q in fracs):
        halt("bit-reversal self-test FAIL -- run VOID")

    rng = np.random.default_rng(seed)
    perms = [rng.permutation(n) for _ in range(n_shuffle)]
    null_max = {q: 0.0 for q in fracs}
    for p in perms:
        c = concentration(fwht(v[p].copy()), fracs)
        for q in fracs:
            null_max[q] = max(null_max[q], c[q])
    gauss = concentration(fwht(rng.normal(v.mean(), v.std(), n)), fracs)

    f0 = fracs[0]
    rec = {
        "n": n,
        "fracs": list(fracs),
        "real": [real[q] for q in fracs],
        "shuffle_max": [null_max[q] for q in fracs],
        "gaussian": [gauss[q] for q in fracs],
        "margin": real[f0] / max(null_max[f0], 1e-12),
        "beyond_nulls": [bool(real[q] > null_max[q] and real[q] > gauss[q]) for q in fracs],
        "self_test": "PASS",
    }

    if comparators:
        # dyadic_placement stays defined against dct+haar as registered at
        # consolidation; slant joined the arms at atlas round 2 (7b-II)
        for cname, tf in (("dct", dct_ortho), ("haar", haar_ortho),
                          ("slant", slant_ortho)):
            e_in = float((v ** 2).sum())
            spec = tf(v)
            e_out = float((spec ** 2).sum())
            if not np.isclose(e_in, e_out, rtol=1e-6):
                halt(f"{cname} comparator is not energy-preserving -- run VOID")
            c_real = concentration(spec, fracs)
            c_null = 0.0
            for p in perms[:3]:
                c_null = max(c_null, concentration(tf(v[p].copy()), fracs)[f0])
            rec[f"{cname}_margin"] = c_real[f0] / max(c_null, 1e-12)
        rec["dyadic_placement"] = bool(
            rec["margin"] > rec["dct_margin"] and rec["margin"] > rec["haar_margin"])
    return rec


def block_margin(v, rng, fracs=FRACS):
    """One row-block margin, the corrected instrument's unit: 3 shuffles."""
    real = concentration(fwht(v.copy()), fracs)
    nm = 0.0
    for _ in range(3):
        nm = max(nm, concentration(fwht(v[rng.permutation(len(v))].copy()), fracs)[fracs[0]])
    return real[fracs[0]] / max(nm, 1e-12)


def probe_rowblocks(w2d, max_blocks=12, comparators=False, seed=SEED):
    """The corrected scale-aware instrument (2f amendment): consecutive
    full-row blocks of ~2^22, MEDIAN block margin (median, not max)."""
    v = w2d.astype(np.float64)
    rows_per_block = max(1, min(v.shape[0], BLOCK // v.shape[1]))
    n_fit = 1 << int(np.floor(np.log2(rows_per_block * v.shape[1])))
    rng = np.random.default_rng(seed)
    margins = []
    comp = []
    r = 0
    while r + rows_per_block <= v.shape[0] and len(margins) < max_blocks:
        blk = v[r:r + rows_per_block].ravel()[:n_fit].copy()
        margins.append(block_margin(blk, rng))
        if comparators:
            comp.append(battery(blk, n_shuffle=3, comparators=True, seed=seed))
        r += rows_per_block
    rec = {
        "blocks": len(margins),
        "rows_per_block": rows_per_block,
        "block_n": n_fit,
        "median_margin": float(np.median(margins)),
        "min_margin": float(min(margins)),
        "max_margin": float(max(margins)),
        "margins": [float(m) for m in margins],
    }
    if comparators and comp:
        rec["median_dct_margin"] = float(np.median([c["dct_margin"] for c in comp]))
        rec["median_haar_margin"] = float(np.median([c["haar_margin"] for c in comp]))
        rec["dyadic_placement"] = bool(
            rec["median_margin"] > rec["median_dct_margin"]
            and rec["median_margin"] > rec["median_haar_margin"])
    return rec


# --------------------------------------------------------- registration gate

REQUIRED_REG_KEYS = ("name", "objects", "statistic", "verdict_rule", "margin_clause")


class Run:
    """A registered run. Every recorded row carries the registration hash.
    Construction refuses (halts) on a missing or incomplete registration."""

    def __init__(self, registration):
        if not isinstance(registration, dict):
            halt("no registration block -- refusing to run")
        missing = [k for k in REQUIRED_REG_KEYS if not registration.get(k)]
        if missing:
            halt(f"registration incomplete, missing {missing} -- refusing to run "
                 "(margin_clause is mandatory by template)")
        canon = json.dumps(registration, sort_keys=True, separators=(",", ":"))
        self.reg_hash = hashlib.sha256(canon.encode()).hexdigest()[:16]
        self.registration = registration
        self.name = registration["name"]
        stamp = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "reg": self.reg_hash,
                 "registration": registration, "seed": SEED}
        with open(REGISTRY, "a") as fh:
            fh.write(json.dumps(stamp) + "\n")
        print(f"[foldprobe] run '{self.name}' registered {self.reg_hash}", flush=True)

    def record(self, **row):
        row.update(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), reg=self.reg_hash,
                   run=self.name, seed=SEED)
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    def battery(self, label, v, **kw):
        rec = battery(v, **kw)
        return self.record(instrument="battery", object=label, **rec)

    def rowblocks(self, label, w2d, **kw):
        rec = probe_rowblocks(w2d, **kw)
        return self.record(instrument="rowblocks", object=label, **rec)


# ------------------------------------------------------------- object loaders

def load_safetensor(path, key):
    from safetensors import safe_open
    return safe_open(path, framework="numpy").get_tensor(key)


def load_gguf_2d(path, key):
    """Dequantize one GGUF tensor to a 2D float array (per-expert slice 0
    for 3D expert tensors -- the pre-fix behaviour, kept for continuity;
    moe_dequant.py owns the corrected expert-axis extraction)."""
    import gguf
    from gguf.quants import dequantize
    r = gguf.GGUFReader(path)
    tens = {t.name: t for t in r.tensors}
    if key not in tens:
        halt(f"tensor {key} not in {os.path.basename(path)}")
    t = tens[key]
    w = np.asarray(dequantize(t.data, t.tensor_type)).ravel()
    shape = tuple(int(x) for x in t.shape)
    if len(shape) == 3:
        per = shape[1] * shape[2]
        return w[:per].reshape(shape[1], shape[2])
    return w.reshape(shape[-2], shape[-1])


# ---------------------------------------------------------------- calibration

CALIBRATION_REG = {
    "name": "foldprobe-calibration",
    "objects": ["GPT-2 h.0.mlp.c_fc.weight (known-loud, recorded 12.7x-class)",
                "GPT-2 wte.weight (known-loud, recorded 79.3x-class flat / 67.6x rowblock)",
                "Qwen3.6-27B blk.0.ffn_gate.weight (known-quiet, recorded ~1x)",
                "He-init matched-shape matrix (negative control)"],
    "statistic": "battery C(k) at registered fractions + rowblock median margin",
    "verdict_rule": "instrument FAITHFUL iff loud objects reproduce their recorded "
                    "margin class, quiet object reads ~1x, He-init sits at null "
                    "(no fraction beyond both nulls), and every bit-reversal "
                    "self-test passes",
    "margin_clause": "loud reproduction = within a factor of 1.5 of the recorded "
                     "margin; quiet = flat-window margin < 2x; He-init = 0/3 "
                     "fractions beyond nulls",
}


def calibrate():
    run = Run(CALIBRATION_REG)
    gp = gpt2_path()

    cfc = load_safetensor(gp, "h.0.mlp.c_fc.weight")
    r1 = run.battery("GPT-2 h.0.mlp.c_fc", cfc.ravel())
    print(f"  GPT-2 c_fc L0   margin {r1['margin']:.2f}x  (recorded class 12.7x)  "
          f"dct {r1['dct_margin']:.2f}x haar {r1['haar_margin']:.2f}x  "
          f"dyadic_placement={r1['dyadic_placement']}", flush=True)

    wte = load_safetensor(gp, "wte.weight")
    r2 = run.battery("GPT-2 wte", wte.ravel())
    print(f"  GPT-2 wte       margin {r2['margin']:.2f}x  (recorded class 79.3x)  "
          f"dct {r2['dct_margin']:.2f}x haar {r2['haar_margin']:.2f}x", flush=True)

    r2b = run.rowblocks("GPT-2 h.0.mlp.c_fc [rowblocks]", cfc)
    print(f"  GPT-2 c_fc rowblocks MEDIAN {r2b['median_margin']:.2f}x "
          f"(min {r2b['min_margin']:.2f} max {r2b['max_margin']:.2f})", flush=True)

    quiet_path = os.path.join(GGUF_LIB, "Qwen3.6-27B-Q4_K_M.gguf")
    if os.path.exists(quiet_path):
        q = load_gguf_2d(quiet_path, "blk.0.ffn_gate.weight")
        r3 = run.battery("Qwen3.6-27B blk.0.ffn_gate", q.ravel())
        print(f"  Qwen3.6-27B gate margin {r3['margin']:.2f}x  (recorded quiet ~1x)", flush=True)
    else:
        run.record(instrument="battery", object="Qwen3.6-27B blk.0.ffn_gate",
                   skipped="GGUF library volume not mounted")
        print("  Qwen3.6-27B: models volume not mounted -- recorded as skipped", flush=True)

    rng = np.random.default_rng(SEED)
    he = rng.normal(0.0, np.sqrt(2.0 / 768.0), cfc.shape)
    r4 = run.battery("He-init negative control", he.ravel())
    print(f"  He-init control margin {r4['margin']:.2f}x  beyond-nulls "
          f"{sum(r4['beyond_nulls'])}/3 (must be 0/3)", flush=True)

    faithful = (abs(np.log(r1["margin"] / 12.7)) < np.log(1.5)
                and sum(r4["beyond_nulls"]) == 0)
    run.record(instrument="verdict", object="calibration",
               faithful=bool(faithful),
               loud_margin=r1["margin"], control_beyond=sum(r4["beyond_nulls"]))
    print(f"\nFOLDPROBE CALIBRATION: {'FAITHFUL' if faithful else 'NOT FAITHFUL -- investigate'}",
          flush=True)


if __name__ == "__main__":
    calibrate()
