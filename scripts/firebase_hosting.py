"""Give the Cloud Run service a short URL: https://<site>.web.app via Firebase Hosting (free).

    python scripts/firebase_hosting.py                       # tries site names crossverse, crossverse-app, ...
    python scripts/firebase_hosting.py --site my-name --service crossverse --region europe-west1

Firebase Hosting forwards every path to the Cloud Run service (a "rewrite"), so the app, its API and
shared links work under the short domain. Uses the logged-in gcloud account and project and the Firebase
REST APIs (no Node.js / firebase CLI needed). Safe to re-run: it reuses an existing site.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import requests

FIREBASE = "https://firebase.googleapis.com/v1beta1"
HOSTING = "https://firebasehosting.googleapis.com/v1beta1"


def gcloud(*args: str) -> str:
    sys.path.insert(0, __file__.rsplit("scripts", 1)[0] + "scripts")
    from deploy_cloudrun import gcloud as find_gcloud

    out = subprocess.run([find_gcloud(), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", action="append", help="site name(s) to try, first free one wins")
    p.add_argument("--service", default="crossverse")
    p.add_argument("--region", default="europe-west1")
    args = p.parse_args(argv)
    names = args.site or ["crossverse", "crossverse-app", "crossverse-recs", "crossverse-movies-games"]

    project = gcloud("config", "get-value", "project")
    token = gcloud("auth", "print-access-token")
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "x-goog-user-project": project,
                      "content-type": "application/json"})

    # 1. make the Google Cloud project a Firebase project (once)
    if s.get(f"{FIREBASE}/projects/{project}").status_code == 404:
        print(f"adding Firebase to {project} ...", flush=True)
        r = s.post(f"{FIREBASE}/projects/{project}:addFirebase", json={})
        r.raise_for_status()
        op = r.json()["name"]
        while not s.get(f"{FIREBASE}/{op}").json().get("done"):
            time.sleep(3)

    # 2. a Hosting site with a free <name>.web.app domain (reuse one we already own)
    owned = {x["name"].rsplit("/", 1)[-1] for x in s.get(f"{HOSTING}/projects/{project}/sites").json().get("sites", [])}
    site = next((n for n in names if n in owned), None)
    for name in names:
        if site:
            break
        r = s.post(f"{HOSTING}/projects/{project}/sites", params={"siteId": name}, json={})
        if r.ok:
            site = name
        else:
            print(f"  {name}.web.app not available ({r.status_code})", flush=True)
    if not site:
        print("no site name available; pass --site <name>", file=sys.stderr)
        return 1

    # 3. a release whose only job is to forward everything to the Cloud Run service
    config = {"rewrites": [{"glob": "**", "run": {"serviceId": args.service, "region": args.region}}]}
    version = s.post(f"{HOSTING}/sites/{site}/versions", json={"config": config})
    version.raise_for_status()
    vname = version.json()["name"]
    s.patch(f"{HOSTING}/{vname}", params={"update_mask": "status"}, json={"status": "FINALIZED"}).raise_for_status()
    s.post(f"{HOSTING}/sites/{site}/releases", params={"versionName": vname}, json={}).raise_for_status()
    print(f"\nShort URL: https://{site}.web.app  (also https://{site}.firebaseapp.com)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
