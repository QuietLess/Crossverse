"""Schedule the new-release refresh on Google Cloud: a Cloud Run job, run twice a week by Cloud Scheduler.

    python scripts/deploy_refresh_job.py                  # print the gcloud commands (changes nothing)
    python scripts/deploy_refresh_job.py --apply          # run them
    python scripts/deploy_refresh_job.py --apply --run-now   # ... and start one refresh right away

What it sets up, in the project and region of the service (scripts/deploy_cloudrun.py):

1. a private Cloud Storage bucket for the published new-release index, and two service accounts so the
   public service and the job don't share an identity: crossverse-refresh (writes the bucket, reads the
   keys, is the scheduler's identity) and crossverse-web (the service; reads the bucket, nothing else);
2. the TMDB and IGDB keys from .env as Secret Manager secrets (values go through stdin, never argv or logs);
3. a Cloud Run job "crossverse-refresh" on the service's own image (code, model and sentence model inside)
   running `crossverse refresh-new-releases --out /mnt/new-releases` with the bucket mounted;
4. a Cloud Scheduler trigger (default Monday and Thursday 05:00 UTC);
5. the service mounts the bucket read-only and reloads a newly published index within 10 minutes.

The job fetches TMDB films/series, IGDB games and AniList anime released since data/new_releases.SINCE,
with a fresh response cache each run, and refuses to publish an index under half the live one. It doesn't
retrain: the behaviour data is frozen, so new titles are matched by story and franchise only.

Costs (estimate, check current pricing): a few minutes of job CPU twice a week, a few MB of storage and
three secrets, all within or close to the free tiers; Cloud Scheduler includes 3 free jobs per billing
account. Needs the Run, Cloud Scheduler, Secret Manager and Cloud Storage APIs enabled.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deploy_cloudrun import gcloud, new_releases_mount  # noqa: E402

from crossverse.config import load_env  # noqa: E402

SECRETS = {"TMDB_TOKEN": "crossverse-tmdb-token", "IGDB_CLIENT_ID": "crossverse-igdb-client-id",
           "IGDB_CLIENT_SECRET": "crossverse-igdb-client-secret"}


def out(*args: str) -> str:
    return subprocess.run([gcloud(), *args], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--service", default="crossverse")
    p.add_argument("--region", default="europe-west1")
    p.add_argument("--job", default="crossverse-refresh")
    p.add_argument("--bucket", default=None, help="default: <project>-crossverse-new-releases")
    p.add_argument("--schedule", default="0 5 * * 1,4", help="cron, in --time-zone (default Mon+Thu 05:00)")
    p.add_argument("--time-zone", default="Etc/UTC")
    p.add_argument("--apply", action="store_true", help="run the commands (default: print them)")
    p.add_argument("--run-now", action="store_true", help="with --apply: start one refresh now")
    args = p.parse_args(argv)

    env = load_env()
    missing = [k for k in SECRETS if not env.get(k)]
    if missing:
        print(f"missing in .env: {', '.join(missing)}", file=sys.stderr)
        return 1
    project = out("config", "get-value", "project")
    job_sa, web_sa = (f"crossverse-{n}@{project}.iam.gserviceaccount.com" for n in ("refresh", "web"))
    image = out("run", "services", "describe", args.service, "--region", args.region,
                "--format=value(spec.template.spec.containers[0].image)")
    bucket = args.bucket or f"{project}-crossverse-new-releases"
    mount = ["--add-volume", f"name=new-releases,type=cloud-storage,bucket={bucket}",
             "--add-volume-mount", "volume=new-releases,mount-path=/mnt/new-releases"]
    secrets = ",".join(f"{k}={v}:latest" for k, v in SECRETS.items())
    run_uri = (f"https://run.googleapis.com/v2/projects/{project}/locations/{args.region}/jobs/{args.job}:run")

    steps: list[tuple[list[str], str | None, bool]] = [  # (gcloud args, stdin, may fail because it exists)
        (["iam", "service-accounts", "create", "crossverse-refresh",
          "--display-name=CrossVerse new-release refresh job"], None, True),
        (["iam", "service-accounts", "create", "crossverse-web", "--display-name=CrossVerse public service"],
         None, True),
        (["storage", "buckets", "create", f"gs://{bucket}", f"--location={args.region}",
          "--uniform-bucket-level-access", "--public-access-prevention"], None, True),
        (["storage", "buckets", "add-iam-policy-binding", f"gs://{bucket}", f"--member=serviceAccount:{job_sa}",
          "--role=roles/storage.objectAdmin"], None, False),
        (["storage", "buckets", "add-iam-policy-binding", f"gs://{bucket}", f"--member=serviceAccount:{web_sa}",
          "--role=roles/storage.objectViewer"], None, False),
    ]
    for key, name in SECRETS.items():
        steps += [(["secrets", "create", name, "--replication-policy=automatic", "--data-file=-"], env[key], True),
                  (["secrets", "add-iam-policy-binding", name, f"--member=serviceAccount:{job_sa}",
                    "--role=roles/secretmanager.secretAccessor"], None, False)]
    steps += [
        (["run", "jobs", "deploy", args.job, "--image", image, "--region", args.region, "--command", "crossverse",
          "--args", "refresh-new-releases,--out,/mnt/new-releases", "--set-secrets", secrets,
          "--cpu", "2", "--memory", "4Gi", "--task-timeout", "3600", "--max-retries", "1",
          "--service-account", job_sa, *mount], None, False),
        (["run", "jobs", "add-iam-policy-binding", args.job, "--region", args.region,
          f"--member=serviceAccount:{job_sa}", "--role=roles/run.invoker"], None, False),
        (["scheduler", "jobs", "create", "http", f"{args.job}-schedule", f"--location={args.region}",
          f"--schedule={args.schedule}", f"--time-zone={args.time_zone}", f"--uri={run_uri}", "--http-method=POST",
          f"--oauth-service-account-email={job_sa}"], None, True),
        (["run", "services", "update", args.service, "--region", args.region, "--service-account", web_sa,
          *new_releases_mount(bucket)], None, False),
    ]
    if args.run_now:
        steps.append((["run", "jobs", "execute", args.job, "--region", args.region], None, False))

    for cmd, stdin, may_exist in steps:
        shown = "gcloud " + " ".join(shlex.quote(c) for c in cmd) + ("  < (secret from .env)" if stdin else "")
        print(shown, flush=True)
        if not args.apply:
            continue
        done = subprocess.run([gcloud(), *cmd], input=stdin, capture_output=True, text=True)
        if done.returncode and may_exist and "already exists" in done.stderr.lower():
            if stdin is not None:  # an existing secret gets a new version with the current value
                name = cmd[2]
                subprocess.run([gcloud(), "secrets", "versions", "add", name, "--data-file=-"], input=stdin,
                               capture_output=True, text=True, check=True)
            print("  (exists)")
        elif done.returncode:
            print(done.stderr.strip(), file=sys.stderr)
            return done.returncode
    if not args.apply:
        print("\nNothing changed. Re-run with --apply to create these resources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
