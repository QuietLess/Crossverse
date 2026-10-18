# Next steps

Where things stand on 2026-10-18 and how to continue. On `main` and pushed (CI green): API hardening,
the TMDB logo, anime as a tag, and the scheduled new-release refresh. **The public site still runs the
older code**: none of this is live until the steps below are done.

## 1. Prepare the deploy machine

It needs things that aren't in the repository (on purpose):

| what | where it comes from |
|---|---|
| the trained model: `artifacts/models/<version>/` and `artifacts/models/PRODUCTION` | the machine that trained it (the old Windows machine), or retrain (README › Quickstart) |
| `data/external/metadata.parquet` (catalog franchises, staged into the image for the refresh job) | the same machine, or `python pipelines/enrich.py` |
| `.env` with `TMDB_TOKEN`, `IGDB_CLIENT_ID`, `IGDB_CLIENT_SECRET` | your TMDB and Twitch developer accounts |
| Google Cloud SDK, logged in, project set | `gcloud auth login` and `gcloud config set project <PROJECT_ID>` |

```bash
pip install -e ".[dev,web,semantic]" -c constraints.txt
pytest                      # expect 313 passed, 3 skipped (Postgres/Redis tests need servers)
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
    cloudscheduler.googleapis.com secretmanager.googleapis.com storage.googleapis.com iam.googleapis.com
```

## 2. Go live (in this order)

The refresh job runs on the service's image, so the service must run the new code before the job is
created.

1. **Redeploy the site with the new code** (logo, Anime chip, request limits):

   ```bash
   python scripts/deploy_cloudrun.py
   ```

   Check https://crossverse-app.web.app : the TMDB logo is in the footer; `/health` answers.

2. **Create the scheduled refresh.** First look at the commands, then run them:

   ```bash
   python scripts/deploy_refresh_job.py                    # prints only, changes nothing
   python scripts/deploy_refresh_job.py --apply --run-now  # creates everything and runs one refresh
   ```

   This creates the bucket `<PROJECT_ID>-crossverse-new-releases`, the service accounts
   `crossverse-refresh` (job) and `crossverse-web` (site), three secrets, the job, the schedule
   (Monday and Thursday 05:00 UTC; change with `--schedule` and `--time-zone`), and mounts the bucket on
   the site. Estimated cost: a few cents a month (check current pricing). If a step fails right after
   the service accounts are created ("does not exist"), wait a minute and run it again.

3. **Check the first refresh** (it takes a few minutes; the TMDB and IGDB parts have never run live
   with the new code, so watch this one):

   ```bash
   gcloud run jobs executions list --job crossverse-refresh --region europe-west1
   gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="crossverse-refresh"' --limit 50
   curl -s https://crossverse-app.web.app/health      # "new_releases": {"version": "2026...", "titles": ...}
   ```

   The site picks up the new index within 10 minutes. Search an anime from this season: it should show
   under Movies or TV series with an Anime chip.

4. **From now on, redeploy with the bucket**, so the job moves to each new image and rebuilds the index
   for the new model (an index is only served by the model it was built for):

   ```bash
   python scripts/deploy_cloudrun.py --new-releases-bucket <PROJECT_ID>-crossverse-new-releases
   ```

## 3. If something goes wrong

| problem | what to do |
|---|---|
| the job keeps failing | the site keeps its last good index (or the model's own); read the job logs; pause the schedule with `gcloud scheduler jobs pause crossverse-refresh-schedule --location europe-west1` |
| "shrank ... not published" in the job log | a source returned far fewer titles (expired key, outage). Fix the cause; `crossverse refresh-new-releases --force` publishes anyway |
| a bad index went live | point `CURRENT` in the bucket at an earlier version folder (3 are kept) |
| the new site version misbehaves | `gcloud run services update-traffic crossverse --region europe-west1 --to-revisions <OLD_REVISION>=100` |

Try it locally without Google Cloud: `make refresh-new-releases`, then
`CROSSVERSE_NEW_RELEASES_DIR=artifacts/new_releases make serve`.

## 4. Afterwards

* **Keep TMDB's written permission** with the project records (docs/ROADMAP.md › TMDB terms).
* **Follow-ups from the security reviews** (all LOW, none blocks): docs/ROADMAP.md › Next › Refresh
  follow-ups. The most useful: alert on failed job runs, and stop serving an index older than 6 months.
* **Rate limiting** for the public API, and optionally a **durable feedback log** (hosted Postgres, a
  recurring cost): docs/ROADMAP.md › Next.
* **Uncommitted local change**: `.gitignore` adds `claude.md`, which on macOS also ignores `CLAUDE.md`
  (the project instructions). Commit it if that's intended, or drop the line.
