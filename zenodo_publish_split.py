"""Publish the corrected split papers to Zenodo (Maria's directive, 2026-07-16).

1) NEW VERSION v5.0 on the architecture record lineage (concept 10.5281/zenodo.21217278):
   the corrected, split architecture paper (papers/UnisonAI_Architecture.md) — same flow as
   zenodo_new_version.py (newversion -> replace file -> version/date -> publish).
2) NEW RECORD for the decode/interpretability paper (papers/Fold_Decode_Interpretability.md),
   metadata matching the existing record's conventions (creators, license, type), cross-
   linked to the theory DOI, both GitHub repos, and the architecture concept DOI.

Token from ZENODO_TOKEN or ~/.zenodo_token (never printed). --dry-run prints intentions.
Publishing is irreversible.
"""
import os, sys, time
import requests

DRY = "--dry-run" in sys.argv
API = "https://zenodo.org/api/deposit/depositions"
ARCH_RECID = 21364144
PAPERS_DIR = "/Users/mettamazza/Desktop/Unison AI/papers"

DECODE_DESCRIPTION = """<h3>The fold decode campaign — headline findings</h3>
<ul>
<li><b>The token embedding is the universal law-carrying class: 11/11 models wake</b>, every training recipe, 4B to 1T parameters.</li>
<li><b>The loud band is the function</b>: matched-budget ablation destroys the model at ~150x the damage of random deletion.</li>
<li><b>The deposition curve</b> read from public checkpoints: the embedding wakes first (step 256), peak near step 4000, consolidation to a plateau; the optimizer is the discriminating ingredient (gradient stream loud by step 4).</li>
<li><b>Two spectral families, forced</b>: exactly two (one per generator), selection forced by store role — <i>family follows role, and the role is architecture</i> — closed to the corpus's finished form-closure standard and verified by the role census.</li>
<li>Training data and reasoning are readable back out of trained weights (provenance ranking, memorization echo, counted reasoning signatures) — registered calibrations, clean nulls.</li>
</ul>
<p>Split from the combined record (v4.x lineage) as its own paper. Companion architecture paper: the UnisonAI record (concept DOI 10.5281/zenodo.21217278). Theory: DOI 10.5281/zenodo.21182469. Every number is from committed, timestamped campaign records; instruments and ledgers: github.com/MettaMazza/UnisonAI (omni/benchmarks/).</p>"""


def token():
    t = os.environ.get("ZENODO_TOKEN")
    if not t and os.path.exists(os.path.expanduser("~/.zenodo_token")):
        t = open(os.path.expanduser("~/.zenodo_token")).read().strip()
    if not t:
        sys.exit("No token: export ZENODO_TOKEN=... or write it to ~/.zenodo_token")
    return t


def main():
    P = {"access_token": token()}
    who = requests.get(API, params={**P, "size": 1}, timeout=30)
    if who.status_code != 200:
        sys.exit(f"Token check failed (HTTP {who.status_code}): {who.text[:200]}")
    print("token OK")

    # existing record's conventions (public API)
    rec = requests.get(f"https://zenodo.org/api/records/{ARCH_RECID}", timeout=30).json()
    subtype = (rec["metadata"].get("resource_type") or {}).get("subtype") or "preprint"

    # ---- 1) architecture paper -> v5.0 (her flow, verbatim) ----
    arch_bytes = open(os.path.join(PAPERS_DIR, "UnisonAI_Architecture.md"), "rb").read()
    print(f"\n=== arch lineage {ARCH_RECID} -> v5.0 ({len(arch_bytes)} bytes) ===")
    if DRY:
        print("  [dry-run] newversion -> replace file -> version=v5.0 -> publish")
    else:
        r = requests.post(f"{API}/{ARCH_RECID}/actions/newversion", params=P, timeout=60)
        if r.status_code == 403 and "latest" in r.text:
            latest = requests.get(f"https://zenodo.org/api/records/{ARCH_RECID}",
                                  timeout=30).json()["links"]["latest"].rstrip("/").split("/")[-1]
            print(f"  {ARCH_RECID} not latest; using {latest}")
            r = requests.post(f"{API}/{latest}/actions/newversion", params=P, timeout=60)
        r.raise_for_status()
        draft = requests.get(r.json()["links"]["latest_draft"], params=P, timeout=60).json()
        did = draft["id"]
        print(f"  draft {did}")
        for f in draft.get("files", []):
            requests.delete(f"{API}/{did}/files/{f['id']}", params=P, timeout=60).raise_for_status()
        up = requests.put(f"{draft['links']['bucket']}/UnisonAI_The_Forced_Language_Architecture.md",
                          data=arch_bytes, params=P, timeout=300)
        up.raise_for_status()
        md = draft["metadata"]
        md["version"] = "v5.0"
        md["publication_date"] = time.strftime("%Y-%m-%d")
        md.pop("doi", None)
        requests.put(f"{API}/{did}", params=P, json={"metadata": md}, timeout=60).raise_for_status()
        pub = requests.post(f"{API}/{did}/actions/publish", params=P, timeout=120)
        pub.raise_for_status()
        print(f"  PUBLISHED v5.0: doi {pub.json().get('doi')}")

    # ---- 2) the decode paper -> NEW record v1.0 ----
    dec_bytes = open(os.path.join(PAPERS_DIR, "Fold_Decode_Interpretability.md"), "rb").read()
    meta = {
        "title": "The Law Inside Trained Weights: The Fold Decode Campaign",
        "upload_type": "publication",
        "publication_type": subtype,
        "description": DECODE_DESCRIPTION,
        "creators": [{"name": "Smith, Maria", "affiliation": "Ernos Labs"}],
        "license": "cc-by-4.0",
        "version": "v1.0",
        "publication_date": time.strftime("%Y-%m-%d"),
        "related_identifiers": [
            {"identifier": "10.5281/zenodo.21182469", "relation": "isSupplementTo", "scheme": "doi"},
            {"identifier": "10.5281/zenodo.21217278", "relation": "isSupplementedBy", "scheme": "doi"},
            {"identifier": "https://github.com/MettaMazza/Smithian-Fold-Theory-Of-Everything",
             "relation": "isSupplementedBy", "scheme": "url"},
            {"identifier": "https://github.com/MettaMazza/UnisonAI",
             "relation": "isSupplementedBy", "scheme": "url"},
        ],
        "keywords": ["interpretability", "Walsh-Hadamard", "spectral analysis",
                     "Smithian Fold Theory", "neural network weights", "zero parameters"],
    }
    print(f"\n=== NEW record: decode paper v1.0 ({len(dec_bytes)} bytes) ===")
    if DRY:
        print(f"  [dry-run] create deposit ({subtype}) -> upload Fold_Decode_Interpretability.md -> publish")
        print(f"  title: {meta['title']}")
    else:
        r = requests.post(API, params=P, json={"metadata": meta}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        did = dep["id"]
        print(f"  deposit {did}")
        up = requests.put(f"{dep['links']['bucket']}/Fold_Decode_Interpretability.md",
                          data=dec_bytes, params=P, timeout=300)
        up.raise_for_status()
        pub = requests.post(f"{API}/{did}/actions/publish", params=P, timeout=120)
        pub.raise_for_status()
        rec2 = pub.json()
        print(f"  PUBLISHED NEW: doi {rec2.get('doi')}  concept {rec2.get('conceptdoi')}")

    print("\nDONE")


if __name__ == "__main__":
    main()
