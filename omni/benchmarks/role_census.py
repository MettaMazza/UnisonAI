"""THE ROLE CENSUS -- A1's registered verification (the recipe map is a role map).

THE FORCED CLAIM UNDER TEST (Steps 309, 316, 317, all closed):
  a store's FAMILY is not chosen by the recipe; it follows the store's ROLE --
  geometry -> colour (smooth), selection -> binary (dyadic), and a pure
  performer stores no family (quiet). What looked like a recipe-to-recipe
  "lean" is a ROLE map. This census reads each store's role (architectural,
  forced) alongside its measured family and checks family-follows-role.

WHAT IS MEASURED vs FORCED:
  - ROLE is forced and unambiguous, read from architecture:
      embedding    -> geometry store   (holds placement)
      expansion    -> selection store  (the gating up/gate projection)
      attention    -> performer        (selects, stores nothing -- Step 315)
      contraction  -> performer        (the return projection)
  - FAMILY is the ALREADY-RECORDED atlas reading (results.jsonl, round-2 with
    the slant arm). Step 316 forces the discriminator to be the VALUE signature
    (whole=binary vs {1/4,3/4}=colour); the atlas read is concentration-margin,
    a loudness proxy. This census uses the loudness reading as-is and NAMES,
    per class, whether that shape suffices -- the value-signature re-read is the
    registered refinement where it does not. The slant arm (dyadic family, it
    tracks Walsh) is counted with Walsh as BINARY; DCT as COLOUR.

VERDICT RULE (fixed before the run): family-follows-role is SUPPORTED on the
  recorded reading iff, among woken stores, geometry stores are colour in a
  majority AND performer stores are quiet in ALL; the selection->binary cell is
  reported as its own count (concentration/slant is the muddy class -- the
  S316 value-signature re-read is the refinement, GGUF-dependent for the
  giants). Recorded whichever way it lands.

No model weights are loaded here -- this reads the committed ledger. It is a
comparison-only, behind-the-wall measurement; it forces nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run

import json

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "results.jsonl")
WAKE = 2.0   # the atlas "woken" threshold (margin >= 2x a shuffle null)

ROLE = {
    "embedding": "geometry",
    "expansion": "selection",
    "attention": "performer",
    "contraction": "performer",
}
FORCED_FAMILY = {           # Step 317: family forced by role
    "geometry": "colour",
    "selection": "binary",
    "performer": "quiet",
}

REG = {
    "name": "role-census",
    "objects": ["the committed basis-atlas round-2 rows (11 public models x 4 "
                "tensor classes, slant arm included) from results.jsonl"],
    "statistic": "per (model, class): ROLE (architectural, forced) x FAMILY "
                 "(recorded loudest basis: dct=colour; walsh/slant=binary; "
                 "not-woken=quiet), cross-tabulated and matched to the forced "
                 "role->family assignment (geometry->colour, selection->binary, "
                 "performer->quiet)",
    "verdict_rule": "family-follows-role SUPPORTED on the recorded reading iff "
                    "woken geometry stores are colour in a majority AND all "
                    "performer stores are quiet; selection->binary reported as "
                    "its own count (the S316 value-signature re-read is the "
                    "refinement where concentration is muddy). Recorded either way",
    "margin_clause": "wake threshold 2.0x fixed; majority = strictly more than "
                     "half of woken stores in the class; slant counted with "
                     "walsh as the binary (dyadic) family, dct as colour",
}


def family_of(row):
    margins = {
        "walsh": row.get("walsh_margin", 0.0),
        "dct": row.get("dct_margin", 0.0),
        "haar": row.get("haar_margin", 0.0),
        "slant": row.get("slant_margin", 0.0),
    }
    if max(margins.values()) < WAKE:
        return "quiet", margins
    loudest = max(margins, key=margins.get)
    if loudest == "dct":
        return "colour", margins
    if loudest in ("walsh", "slant"):
        return "binary", margins
    return "quiet", margins   # haar-loudest but woken: no law-family


def main():
    rows = [json.loads(l) for l in open(LEDGER)
            if '"instrument": "basis-atlas"' in l and '"slant_margin"' in l]
    if not rows:
        print("no round-2 (slant) atlas rows found in the ledger", flush=True)
        return
    # dedupe to one row per (model, class): the round-2 slant run
    latest = {}
    for r in rows:
        latest[(r["model"], r["tensor_class"])] = r

    run = Run(REG)
    # cross-tab counts
    cells = {}          # (role, family) -> count
    per_class = {}      # role -> {family: count, woken: n}
    lines = []
    for (model, cls), r in sorted(latest.items()):
        role = ROLE.get(cls, cls)
        fam, margins = family_of(r)
        forced = FORCED_FAMILY[role]
        match = (fam == forced)
        cells[(role, fam)] = cells.get((role, fam), 0) + 1
        pc = per_class.setdefault(role, {"colour": 0, "binary": 0, "quiet": 0, "woken": 0})
        pc[fam] += 1
        if fam != "quiet":
            pc["woken"] += 1
        run.record(instrument="role-census", model=model, tensor_class=cls,
                   role=role, measured_family=fam, forced_family=forced,
                   follows_role=bool(match),
                   loudest=r.get("loudest"), woken=bool(r.get("woken")))
        lines.append(f"  {model:16s} {cls:11s} role={role:9s} "
                     f"family={fam:6s} forced={forced:6s} {'OK' if match else 'x'}")

    print("\n".join(lines), flush=True)

    # verdict
    geo = per_class.get("geometry", {})
    perf = per_class.get("performer", {})
    sel = per_class.get("selection", {})
    geo_colour_majority = geo.get("colour", 0) > geo.get("woken", 0) / 2 if geo.get("woken", 0) else False
    perf_all_quiet = (perf.get("colour", 0) + perf.get("binary", 0)) == 0
    sel_binary = sel.get("binary", 0)
    sel_total = sel.get("colour", 0) + sel.get("binary", 0) + sel.get("quiet", 0)
    supported = geo_colour_majority and perf_all_quiet

    print("\n--- census ---", flush=True)
    print(f"  geometry:   colour {geo.get('colour',0)}/{geo.get('woken',0)} woken "
          f"(binary {geo.get('binary',0)}, quiet {geo.get('quiet',0)})  colour-majority={geo_colour_majority}", flush=True)
    print(f"  selection:  binary {sel_binary}/{sel_total}  "
          f"(colour {sel.get('colour',0)}, quiet {sel.get('quiet',0)})  "
          f"-- the S316 value-signature re-read is the refinement", flush=True)
    print(f"  performer:  all quiet = {perf_all_quiet}  "
          f"(colour {perf.get('colour',0)}, binary {perf.get('binary',0)}, quiet {perf.get('quiet',0)})", flush=True)
    run.record(instrument="verdict", object="role-census",
               geometry_colour=geo.get("colour", 0), geometry_woken=geo.get("woken", 0),
               performer_all_quiet=bool(perf_all_quiet),
               selection_binary=sel_binary, selection_total=sel_total,
               family_follows_role_supported=bool(supported))
    print(f"\nROLE CENSUS: family-follows-role {'SUPPORTED' if supported else 'NOT SUPPORTED'} "
          f"on the recorded reading (geometry->colour majority + performers all quiet); "
          f"selection->binary at {sel_binary}/{sel_total}, the value-signature re-read pending.", flush=True)


if __name__ == "__main__":
    main()
