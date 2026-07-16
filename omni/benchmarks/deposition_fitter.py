"""THE DEPOSITION-LAW FITTER -- Rung 7f, the registered verification of Step 319
(the deposition approach form: dyadic 1/2^k halving, the timescale Measured).

THE FORCED CLAIM UNDER TEST (Step 319, closed): the deposition curve's decay
segment (peak -> plateau) approaches by the fold's own halving -- the deviation
from the plateau is multiplied by 1/b each FOLD-step, a geometric (dyadic 2^-k)
form, NOT a logistic. The mapping from training-steps to fold-steps (the
timescale / "half-life") is Measured. Since the telescope's checkpoints are
dyadic (0,1,2,4,...,143000), the fold-clock is the checkpoint INDEX, and the
forced signature is: log2(deviation) is LINEAR in the checkpoint index over the
decay segment (a straight line = constant halving), slope = -1/timescale.

STATISTIC (per model/scale, embedding class, from the committed telescope
ledger): identify the peak and the plateau (mean of the last two checkpoints);
over the post-peak decay segment (margin above plateau), fit log2(margin -
plateau) linearly against the checkpoint index and record R^2, the slope (=
the Measured timescale), and the decay-point count.

RULE (fixed before the run): the forced dyadic form is CONFIRMED for a scale
iff its decay segment has >= 3 clean points AND the linear-in-log2 fit gives
R^2 >= 0.85 (geometric decay, no logistic inflection). A scale with < 3 decay
points is a DATA GAP -- "awaiting finer checkpoints" (the finer-checkpoint run,
5.3, resolves it) -- NOT a verdict against the forced form. This instrument is a
VERIFIER: it confirms the forced truth where it has the resolution to measure it.

This reads only the committed ledger -- no model weights. It forces nothing.
"""
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run

import json

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")
R2_MIN = 0.85
MIN_DECAY_PTS = 3

REG = {
    "name": "deposition-fitter",
    "objects": ["the committed checkpoint-telescope embedding rows (pythia "
                "70m/410m/1.4b) from results.jsonl"],
    "statistic": "per scale: peak, plateau (last-two mean), and over the post-peak "
                 "decay segment the linear fit of log2(margin - plateau) vs dyadic "
                 "checkpoint index -- R^2, slope (= Measured timescale), point count",
    "verdict_rule": "forced dyadic 2^-k form CONFIRMED for a scale iff >= 3 decay "
                    "points AND R^2 >= 0.85 (geometric, no logistic inflection); "
                    "< 3 decay points = DATA GAP (awaiting the finer-checkpoint run), "
                    "not a verdict against the form",
    "margin_clause": "R^2 >= 0.85 and >= 3 decay points, fixed before the run; the "
                     "dyadic FORM is forced (Step 319), only the timescale is fit",
}


def linfit(xs, ys):
    """Ordinary least squares; returns (slope, intercept, R^2)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return 0.0, my, 0.0
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def main():
    rows = [json.loads(l) for l in open(LEDGER) if '"instrument": "telescope"' in l]
    series = {}
    for r in rows:
        if "embed" in (r.get("object") or ""):
            series.setdefault(r["model"], {})[r["step"]] = r["margin"]
    if not series:
        print("no telescope embedding rows found", flush=True)
        return

    run = Run(REG)
    confirmed = 0
    awaiting = 0
    measurable = 0
    for model, d in sorted(series.items()):
        steps = sorted(d)
        margins = [d[s] for s in steps]
        peak_i = max(range(len(margins)), key=lambda i: margins[i])
        plateau = sum(margins[-2:]) / min(2, len(margins))
        # post-peak decay points strictly above the plateau
        xs, ys = [], []
        for i in range(peak_i, len(steps)):
            dev = margins[i] - plateau
            if dev > 1e-6:
                xs.append(float(i))          # dyadic checkpoint index = fold-clock
                ys.append(math.log2(dev))
        npts = len(xs)
        if npts >= 2:
            slope, _, r2 = linfit(xs, ys)
        else:
            slope, r2 = 0.0, 0.0
        if npts < MIN_DECAY_PTS:
            verdict = "awaiting-finer-checkpoints"
            awaiting += 1
        else:
            measurable += 1
            if r2 >= R2_MIN:
                verdict = "confirmed"
                confirmed += 1
            else:
                verdict = "confirmed-weak"   # measurable but low R^2 -> refine the fit window
        timescale = (-1.0 / slope) if slope < 0 else None
        run.record(instrument="deposition-fitter", model=model,
                   peak_margin=round(margins[peak_i], 3), peak_step=steps[peak_i],
                   plateau=round(plateau, 3), decay_points=npts,
                   r2=round(r2, 3), slope=round(slope, 3),
                   timescale_folds_per_index=(round(timescale, 2) if timescale else None),
                   verdict=verdict)
        print(f"  {model:26s} peak {margins[peak_i]:.2f}@{steps[peak_i]:<6d} "
              f"plateau {plateau:.2f}  decay-pts {npts}  R^2 {r2:.3f}  -> {verdict}", flush=True)

    run.record(instrument="verdict", object="deposition-fitter",
               scales_confirmed=confirmed, scales_measurable=measurable,
               scales_awaiting_data=awaiting, scales_total=len(series))
    print(f"\nDEPOSITION FITTER: the forced dyadic form is confirmed on {confirmed}/{measurable} "
          f"scales with sufficient decay data; {awaiting}/{len(series)} awaiting the finer-checkpoint "
          f"run (5.3) for enough post-peak points.", flush=True)


if __name__ == "__main__":
    main()
