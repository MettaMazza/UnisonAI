"""Mint architecture paper v6.0 on the existing Zenodo lineage (Maria's directive:
fact-checked, cross-referenced, 'then pushed'). Same flow as zenodo_new_version.py /
zenodo_publish_split.py: newversion -> replace file -> version/date -> publish.
Token from ~/.zenodo_token (never printed)."""
import os, time, requests

API = "https://zenodo.org/api/deposit/depositions"
ARCH = 21364144
tok = open(os.path.expanduser("~/.zenodo_token")).read().strip()
P = {"access_token": tok}

content = open("/Users/mettamazza/Desktop/Unison AI/papers/UnisonAI_Architecture.md", "rb").read()
print(f"minting v6.0 ({len(content)} bytes)")
r = requests.post(f"{API}/{ARCH}/actions/newversion", params=P, timeout=60)
if r.status_code == 403 and "latest" in r.text:
    latest = requests.get(f"https://zenodo.org/api/records/{ARCH}",
                          timeout=30).json()["links"]["latest"].rstrip("/").split("/")[-1]
    print(f"  {ARCH} not latest; using {latest}")
    r = requests.post(f"{API}/{latest}/actions/newversion", params=P, timeout=60)
r.raise_for_status()
draft = requests.get(r.json()["links"]["latest_draft"], params=P, timeout=60).json()
did = draft["id"]
print(f"  draft {did}")
for f in draft.get("files", []):
    requests.delete(f"{API}/{did}/files/{f['id']}", params=P, timeout=60).raise_for_status()
requests.put(f"{draft['links']['bucket']}/UnisonAI_The_Forced_Language_Architecture.md",
             data=content, params=P, timeout=300).raise_for_status()
md = draft["metadata"]
md["version"] = "v6.0"
md["publication_date"] = time.strftime("%Y-%m-%d")
md.pop("doi", None)
requests.put(f"{API}/{did}", params=P, json={"metadata": md}, timeout=60).raise_for_status()
pub = requests.post(f"{API}/{did}/actions/publish", params=P, timeout=120)
pub.raise_for_status()
print("PUBLISHED v6.0: doi", pub.json().get("doi"))
