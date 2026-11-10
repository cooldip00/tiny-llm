# Progress

## Current
_Rewritten 2026-11-09. The previous block is kept unchanged at the end of this file ("Old Current block"). This block holds only the state and one pending list: edit it, don't append to it. Details and sub-plans stay in the Log and in the 2026-11-08/09 notes under 6.4._

- **Phase:** 6, shipping v1. Phase 4 is finished for v1 except 4.7, which is optional and not on the v1 path. Phase 5 and Phase 7 are deferred (D-021).
- **Git:** last pushed `a83e980` (CI green). `git status -sb` shows any local commits not pushed yet.
- **Live:** public router Space `CoolDip00/tiny-llm` with two private replicas (`-replica-1`, `-replica-2`), all cpu-basic, sleep after 48 h. Model `CoolDip00/tiny-llm-tinystories-30m`. Private log dataset `CoolDip00/tiny-llm-space-logs`. W&B project public, run `ugd124lz` linked from the README. Test Space `CoolDip00/tiny-llm-dmtest`: don't open its page before the 11-10 check.
- **How we work:** one item at a time, report and wait for a go-ahead. Sub-plan first, saved here or in the Log, before and after each piece of work. Ask before commit, push, any GPU run, or anything published to or changed on HF.
- **Blockers:** none
- **Open decisions:** (1) after launch, 6.9 first (agreed 2026-11-09) or Phase 5 first (PLAN 7.1 says first in line); decide after launch.
- **Daily:** ~30 min explaining the day's work out loud (feeds 6.6).

**Next action:** the user edits `blog-draft.local.md` into their own words (6.5d step 3) and starts 6.6. Tuesday items follow the list below.

**Pending before launch:**
1. [ ] **Blog 6.5d:** sub-plan, I draft, the user edits. Home is GitHub Pages and it is the only link shared. Optional dev.to copy about a week later with a canonical link; skip Medium. Include "What I learned" and the plain-language explainer (PARKING_LOT README polish items 6-7). Say `/v1/completions`, not a broad "OpenAI-compatible". No claim about waking a sleeping private replica until the dmtest check. Scan the final text for secrets and personal data.
2. [ ] **6.6 self-check (user):** not shortened. If it isn't ready Tuesday, post Wed or Thu evening.
3. [x] **Plain-text keys file:** on 2026-11-09 the user had the whole old folder moved to `D:\Private\tiny-llm` (12 files, counts and bytes matched), outside the project folder, to keep it as an old copy. `wandb key.txt` there also holds other project keys, so it is NOT deleted; never open it from a session. A password manager is still the better home if the user wants it later.
4. [ ] **Tue 11-10 after 09:00 UTC (2:30 pm IST):** dmtest sleep check (one runtime read, one timed request), plus one test prompt through the live demo.
5. [ ] **Fresh-clone check:** clone, install, tests, one CPU serve command. No retraining.
6. [ ] **GitHub profile (user), Tue daytime:** name, photo, one-line bio, LinkedIn and blog links, review the 23 public repos. No fancy profile README.
7. [ ] **GitHub Pages on; social preview** (I make it, the user uploads: repo Settings, General, Social preview; 1280x640, under 1 MB).
8. [ ] **Tue 7:30-9 pm IST launch:** blog live, LinkedIn post 1 (link in first comment), README Status line 23 to v1, tag v1 (ask first). Attach `runs/bench/v1b` to the release only after a privacy check (ask first).
9. [ ] **Resume and applications (separate track, job folder):** check the ATS fixes and use the `/v1/completions` wording in `main.tex`. Keep applying; this doesn't wait for launch.

**After launch:**
- Thu 11-12 r/LocalLLaMA. Tue 11-17 LinkedIn 2 + r/MachineLearning [P]. Tue 11-24 LinkedIn 3. Check each subreddit's self-promotion rules. 6.5g outreach is ongoing (user).
- After the dmtest check: `docs/SERVING.md` reply table and still-open update, sleeping-Space row and test in `classify`, delete dmtest (ask first).
- By 2026-12-02: downgrade W&B to Free, then reopen the run link logged out. Keep HF PRO while applying: Docker Spaces need it, so cancelling breaks the demo.
- Work order after v1: profile why vLLM is 1.35x on one stream (proposed), 6.9 release strategies (v1.1, ~9-10 h), CUDA graphs for paged KV plus the `--kv paged` HTTP load test, then the rest of PARKING_LOT.
- _Housekeeping cancelled 2026-11-09: the user keeps the old doc copies in `D:\Private\tiny-llm`. Don't delete them._

---

## Log
Newest first. One entry per work block: what was done, numbers, next.

### 2026-11-07 — 6.5b step 5, items 0–1: push step 4; HF Docker Space facts
- **Item 0:** pushed `4af0783`, CI green.
- **Item 1 (HF docs: spaces-overview, spaces-sdks-docker, spaces-gpus, spaces-config-reference):**
  - **CPU basic:** 2 vCPU, 16 GB RAM, 50 GB disk, **not persistent** (lost on every restart), free. Sleeps after **48 h** without visitors; a visit wakes it; custom sleep time only on paid hardware. Every push to the Space repo rebuilds and restarts it; changing secrets or variables restarts it (forum).
  - **Build args:** Space **variables are passed as `--build-arg`** and are also env vars at runtime. So `ARG GIT_COMMIT` in our Dockerfile picks up a Space variable `GIT_COMMIT`. Secrets are env vars at runtime only (at build only via `RUN --mount=type=secret`); `HF_TOKEN` as a secret is what `huggingface_hub` reads, so log shipping needs no code change.
  - **Built-in env:** `SPACE_ID`, `SPACE_HOST`, `SPACE_AUTHOR_NAME`, `SPACE_REPO_NAME`, `SPACE_TITLE`, `SPACE_CREATOR_USER_ID`, `CPU_CORES`, `MEMORY`, `ACCELERATOR`, and `SPACE_SIGNING_SECRET` (random, secret, **stable across restarts, rebuilds and replicas**).
  - **Container:** uid 1000 (our image already does this), `app_port` default 7860, outbound traffic only on ports 80 / 443 / 8080 (the Hub upload uses 443).
  - **SIGTERM / grace period:** not documented. A forum thread shows Docker Spaces do get SIGTERM; the grace period is unknown → measure in item 6 (restart the Space; do `stopped` + `logs uploaded` arrive?).
  - **X-Forwarded-For:** not documented → test in item 6: with `--rate-limit` on, send more than the burst with different spoofed `X-Forwarded-For` values; if none get 429, the leftmost address is client-controlled and item 2's fix is needed.
- **Consequences for the plan:** (a) `GIT_COMMIT` and `LOG_DATASET` as Space variables work. Set them before the first upload, since a variable change restarts or rebuilds the Space. (b) Salt, decided 2026-11-07: the user sets `TINYLLM_LOG_SALT` (the same value in every Space, so D-031 replicas hash a visitor alike). `SPACE_SIGNING_SECRET` is per Space ("replicas" in HF's docs are copies of one Space, not separate Spaces), so it is only the fallback: salt = `TINYLLM_LOG_SALT`, else a hash of `SPACE_SIGNING_SECRET`, else random per start (item 2). Tokens: nothing until item 5; then the user makes one fine-grained HF token (write on `CoolDip00/tiny-llm-space-logs` only) → Space secret `HF_TOKEN`, plus the salt → `TINYLLM_LOG_SALT`. Order: create the Space + variables → user adds the secrets → push code. The local WSL token creates the Space; check its permissions read-only first (edit, not a new token, if it can't). Later: D-031 reuses the token and salt; the GitHub → Space sync Action needs one more token (after D-031). (c) Cold start after sleep downloads the model again (non-persistent disk); measure it in item 6.
- **Item 2 (image changes), done 2026-11-07:**
  - `server.py`: `forwarded_ip(xff, hops)` takes the client address `--proxy-hops` entries from the **right** of `X-Forwarded-For` (default 1), never the leftmost (client-controlled). `resolve_salt()`: `TINYLLM_LOG_SALT` → hash of `SPACE_SIGNING_SECRET` → random; `server ready` logs `salt` (the source, never the value) and `proxy_hops` (0 = proxy not trusted).
  - `Dockerfile`: `--trust-proxy --proxy-hops "$PROXY_HOPS"` only when `SPACE_ID` is set; new ENV `MAX_QUEUE=32 RATE_LIMIT=0 RATE_BURST=10 PROXY_HOPS=1` (server defaults for now; item 3 sets the Space values).
  - Tests: `forwarded_ip` cases, a spoofed-XFF visitor still gets 429 at the burst, salt order. CPU suite 314 passed, 148 skipped (462 tests).
  - Docker: plain `docker run` → smoke check passes, no `--trust-proxy` in the command, `salt: random, proxy_hops: 0`. Fake Space (`SPACE_ID`, `SPACE_SIGNING_SECRET`, `RATE_LIMIT=60 RATE_BURST=2`) → `--trust-proxy --proxy-hops 1`, one visitor faking a new left entry each time gets `200 200 429`, another visitor 200, `salt: SPACE_SIGNING_SECRET`, the secret never in the log.
  - Still open for item 6: is 1 the right hop count on HF? (Phone on mobile data vs laptop must hash differently; a spoofed entry must not.)
- **Item 3, started 2026-11-07 (power mode Performance, fans Max):**
  - `load_test.py` now refuses to overwrite an existing output (checked before the run; test). Commit items 2–4 together (user, 2026-11-07).
  - WSL has 7 GB RAM, so the container gets `--memory 6g` (the model needs < 1 GB; HF's 16 GB doesn't matter here).
  - **Finding: PyTorch ignores the CPU cap.** Under `--cpus 2` (cgroup `cpu.max` = `200000 100000`) the container sees 16 cores and torch starts 8 threads. Probe, SLOTS=4, 128 tokens, 30 s per rate (`runs/bench/load/space-cpu/probe-threads-{default,2}.json` + `.log`):
    - 8 threads: ITL p50 105 ms, ~29 tok/s, TTFT p95 25.6 s already at 0.5 req/s, 24 failures at 2 req/s; rated load none.
    - 2 threads (`OMP_NUM_THREADS=2`): ITL p50 21 ms, 166 tok/s at 2 req/s, TTFT p95 75 ms / 413 ms at 0.5 / 1 req/s; rated load 1.0 req/s.
    - So the server must set torch's threads to the CPU limit (cgroup `cpu.max`, else HF's `CPU_CORES`), or the Space runs ~5× slower. Fix proposed to the user before the full test.
  - **Thread fix, done 2026-11-07 (user approved):** `server.py` `cpu_limit()` reads cgroup v2 `cpu.max`, then cgroup v1 `cpu.cfs_quota_us / cpu.cfs_period_us`, then `CPU_CORES` (HF); on CPU the server sets `torch.set_num_threads(min(limit, torch default))`; `--threads N` overrides (≥ 1); `model loaded` logs `threads`. Test `test_cpu_limit` (v2 cap, 1.5 → 2, no cap, v1 none, `CPU_CORES`, v1 beats it, junk). CPU suite 316 passed, 148 skipped (464 tests). Image: `--cpus 2` → 2 threads; no cap → 8 (torch's default); `-e CPU_CORES=2` → 2; `--cpus 2 -e OMP_NUM_THREADS=1` → 1; `--threads 0` → usage error. Noted in SERVING.md "Constraints".
  - **Pass 1 (2026-11-07 15:40–16:00 UTC, Performance, fans Max):** `--cpus 2 --memory 6g`, queue 32, 128 tokens, 60 s per rate, `--slo space` (TTFT p95 ≤ 2 s); threads 2 in every run. Files `runs/bench/load/space-cpu/pass1-slots{2,4,8}-q32.{json,log,server.jsonl}`.
    - SLOTS 2: max 0.92 req/s (118 tok/s); rated load 0.5 req/s (TTFT p95 553 ms); at 1 req/s p95 4.6 s.
    - SLOTS 4: max 1.92 req/s (246 tok/s); rated 1.0 req/s (p95 187 ms); at 2 req/s p95 9.3 s.
    - SLOTS 8: max 2.16 req/s (276 tok/s); rated **2.0 req/s** (p95 500 ms, ITL p50 18 ms); at 4 req/s p95 16.7 s with queue 32, 68 of 238 shed. 4 → 8 slots: +12% max throughput, ITL under saturation 15 → 24 ms.
    - Choice: SLOTS=8. Queue 32 lets admitted requests wait ~15 s at overload → pass 2 sizes it from throughput: 2 s × ~2.2 req/s ≈ 4.
    - Caveat: the laptop's cores are likely faster than HF's shared vCPUs; item 6's live test checks.
  - **Pass 2 (16:04–16:14 UTC):** SLOTS 8, rates 1–4, files `pass2-slots8-q{4,8}.*`.
    - Queue 4: 2 req/s p95 332 ms, 0 shed; 3 req/s p95 1.20 s, 7% shed, 2.71 req/s served; 4 req/s p95 1.45 s, 20% shed, 3.03 req/s / 388 tok/s.
    - Queue 8: 3 req/s p95 2.08 s (miss); 4 req/s p95 2.52 s.
    - Pass 1's 2.16 req/s at queue 32 was understated (wall time includes draining the long queue).
  - **Chosen and set in the Dockerfile:** `SLOTS=8 MAX_QUEUE=4 RATE_LIMIT=20 RATE_BURST=5`. Rated load 2.0 req/s. Results table in docs/SERVING.md "Results: Space defaults on 2 CPUs"; PLAN item 3 ticked; PARKING_LOT "Queue sized from the SLO" noted as applied to the Space.
- **Item 4 done (2026-11-07):** `space/README.md` (YAML: docker, port 7860, MIT, short description; demo / status / API / GitHub + model links; limits incl. 20 req/min and sleep; what is logged) and `scripts/push_space.py`.
  - Dry run by default, `--yes` to act (like `upload_model.py`); `--create --yes` makes the public Space (docker, cpu-basic, `exist_ok=False`) with `LOG_DATASET` set at creation; owner checked against `whoami`.
  - Push: allow-list read from git at HEAD (not the working tree), `__pycache__`/`.pyc` skipped; refuses uncommitted tracked changes or HEAD on no remote branch; one Space commit, stale Space files deleted (`.gitattributes` kept); record in `runs/space/push_<time>.json` (never overwritten).
  - **Change from the plan:** the commit id goes into the uploaded Dockerfile's `ARG GIT_COMMIT` default instead of a `GIT_COMMIT` Space variable. A variable change restarts the Space on its own: set before the upload, the old code would run under the new id; after, the new code under the old id. A leftover `GIT_COMMIT` variable is deleted (it would override).
  - Real-repo dry run: 35 files (Dockerfile, .dockerignore, LICENSE, 32 under `src/tinyllm/` incl. `web/`, the tokenizer) + README; refuses now because `space/README.md` isn't committed (exit 1, as intended).
  - `tests/test_push_space.py`, 6 tests, no network: a temp git repo with checkpoints / data / runs / wandb / hf / NOTES.local.md / .env / a .pyc committed on purpose; none goes up. Full suite 322 passed, 148 skipped (470); hygiene clean.
- **Committed + pushed `6eef970`** (items 2–4, 13 files, no AI attribution); CI success.
- HF login check (read-only): fine-grained token `tiny-llm-laptop`, user scope with `repo.write`; `CoolDip00/tiny-llm` free; log dataset exists.
- **Item 5a blocked (2026-11-07):** Claude Code's auto mode refused to run `--create --yes` (it publishes), so the user ran it. HF answered **402 Payment Required**: "Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription." Nothing was created (checked read-only). Item 1's reading of the docs missed this. It affects 6.5b step 5 and D-031 (replicas + router Spaces), and D-033 (Docker Space).
- **Host research (2026-11-07, web, not verified on provider consoles):**
  - HF: confirmed in HF's docs (`huggingface_hub` PR #5037, forum threads): Docker + Gradio on cpu-basic need PRO since ~July 2026; static Spaces stay free.
  - Google Cloud Run: free per billing account (us-central1 etc.) 180k vCPU-s, 360k GiB-s, 2M requests / month (request-based billing; instance-based 240k / 450k). Card needed. Scales to 0 (cold start), CPU throttled between requests in request-based mode (affects the 10-min log upload thread), SIGTERM with 10 s grace. Its own load balancer and autoscaling overlap D-031's router.
  - Oracle Always Free Ampere A1 VM: reportedly halved to 2 OCPU / 12 GB in June 2026 (official page still showed 4 / 24); card; capacity shortages; idle reclaim; we'd run Docker + a TLS proxy ourselves.
  - Render free: 512 MB, shared ~0.1 CPU, sleeps after 15 min: too small for torch. Koyeb / Northflank: free tiers unclear or card-gated. Fly.io: no free tier since 2024. Railway: credits only.
- **Next:** the user picks the host; nothing runs until then.
- D-031 check on HF PRO (web research, 2026-11-07; nothing created):
  - Creating the 3 Spaces: PRO allows Docker Spaces and the docs give no cap on how many; cpu-basic has no hourly cost, so router + 2 replicas cost only the $9 PRO. HF's own "replicas" feature is paid hardware only and billed per replica: not our route.
  - Router -> replica calls: allowed (outbound 80/443/8080; HF documents Space-to-Space calls). Risk: on 2026-07-08/09 every in-Space request to *.hf.space returned 503 for about a day (HF regression, fixed 07-10). The router must report a clear error when no replica answers.
  - Sleep: cpu-basic sleeps after 48 h idle, cannot be changed even on PRO. One report says a sleeping Space answers a POST without waking and a GET / wakes it (about 30 s). The router wakes replicas with GET /, falling back to restart_space with a token.
  - Visitor identity: replicas see the router, not the visitor, so a per-IP rate limit at the replicas would put all visitors in one bucket. Design change: rate limit at the router; replicas private (router sends a read token) and trust only the router's visitor header.
  - Policy: content policy bans "proxies ... to bypass restrictions"; HF's automated abuse checks have paused Spaces described as proxies. Spreading load over free Spaces is a grey area. Keep to 2 replicas, say so on the README, keep traffic low.
  - Streaming through two HF proxies is untested; logs need per-Space file names (SPACE_ID) and the shared salt.
  - Verdict: no hard blocker found; two unknowns (wake behaviour, policy) need a 30-minute smoke test with Dev Mode before any router code. D-031 needs an amendment for these changes.
  - Budget (the user, 2026-11-07): PRO at $9/month is the limit; a CPU Upgrade Space (~$31/month with PRO) is too much. If the Dev Mode test fails or HF flags anything, D-031 moves to v1.1 (its own cut order), not to paid hardware.
  - Decision (the user, 2026-11-07): go with HF PRO ($9/month) for the demo while applying to inference / ML-infra roles. Plan for 2026-11-08, one item at a time with a go-ahead each:
    1. The user subscribes to PRO; I check `isPro` read-only.
    2. Commit + push the uncommitted PLAN/PROGRESS notes (ask), CI green.
    3. 5a: the user runs `push_space.py --create --yes`; I verify the Space. 5b: the user adds secrets HF_TOKEN + TINYLLM_LOG_SALT (never in chat). 5c: dry run, then `--yes` (ask). 5d: watch build + start.
    4. Item 6 live checks; item 7 scaling design; item 8 links + model-card re-upload (ask) + commit/push (ask). 6.5b done.
    5. D-031: Dev Mode test (private replica copy: token, asleep, streaming) -> go/no-go. Go: DECISIONS amendment (router rate limit, private replicas, GET / wake, per-Space logs), remote-replica Router, create replicas + router (ask), 1-vs-2 load test. No-go or flagged: v1.1.
    6. Then 6.1 README + diagram, 6.3 resume bullets, 6.4 PARKING_LOT review, 6.5c W&B public, 6.5d-g drafts, 6.6 the user's self-check. Tick 4.9 (all sub-steps done).
  - 2026-11-08: step 1 done. The user subscribed to PRO (first invoice $7, prorated for November; $9/month from December); `whoami` shows CoolDip00 isPro=True.
  - 2026-11-08: step 2 done (`b4d7d1f`, CI success). 5a done: the user ran `push_space.py --create --yes`; checked read-only: public, sdk docker, requested cpu-basic, sleep 172800 s (48 h), stage NO_APP_FILE, variable LOG_DATASET=CoolDip00/tiny-llm-space-logs, files .gitattributes + HF's default README.md (replaced by ours in 5c), log dataset exists. Next: 5b, the user adds the two secrets.
  - 2026-11-08: 5b done: the user made a fine-grained token (Read + Write on the dataset `CoolDip00/tiny-llm-space-logs` only, no user/org permissions) and added the Space secrets `HF_TOKEN` and `TINYLLM_LOG_SALT` in the browser; the settings page lists both, and LOG_DATASET is still the only variable. Values were never in chat; the user keeps a private copy of the salt. The secrets can't be read back through the API, so 5d checks them (startup log: salt set; log upload works). Next: commit this note, then 5c (`push_space.py` dry run, then `--yes`).
  - 2026-11-08: note committed (`75e4579`, CI success). 5c done: dry run listed 36 files (35 allow-listed + space/README.md as README.md, 868,455 B), no stale deletes, `ARG GIT_COMMIT=75e4579`; then `--yes` uploaded Space commit `d656a308` (record `runs/space/push_20261108T054633Z.json`, not committed). Next: 5d, watch the build and start read-only (expect `model loaded` threads 2, `server ready`, salt set).
  - 2026-11-08: 5d done: stage RUNNING on cpu-basic, runtime sha `d656a308` (our upload). Startup log: `log upload on` (every 10 min) and a clean `log retention` pass (token reads the dataset); `model loaded` device cpu, threads 2, slots 8, load 3.52 s, version `tiny-llm-tinystories-30m@4efa5367+75e4579`; `server ready` with `salt: TINYLLM_LOG_SALT` (secret in use), proxy_hops 1. First live request already logged: 200, TTFT 93 ms, ITL p50 25 ms, 120 tokens in 3.1 s, client field hashed. Not yet proven: token write (first upload ~10 min after start), checked in item 6. Order change agreed with the user: the D-031 Dev Mode test runs after item 6 and before items 7-8, because SERVING.md scaling text and README depend on its go/no-go. Next: item 6 live checks.
  - 2026-11-08: item 6 so far: `smoke_server.py --url https://cooldip00-tiny-llm.hf.space --wait 600` passed 8/8; `/stats` version `...@4efa5367+75e4579`, norm_backend torch; spoofed `X-Forwarded-For` (1.2.3.4, 203.0.113.77, none) all hashed to the same client id, so the header can't change identity; log dataset confirmed private. The user's screenshots of the demo and status pages look right, but one bug: the status latency chart draws each series as one SVG path, so a minute with no neighbours is an `M` with no `L` (zero length, not drawn) and only the newest point gets a dot; the user's 3.13 s whole-request p95 minute was missing. Sparse demo traffic makes most minutes isolated. Fixing now (user go-ahead): draw a small dot for each isolated point in `src/tinyllm/web/status.html`, check locally, then re-push the Space (ask). Pending: first log upload to the dataset (token write).
  - 2026-11-08: token write proven: Space log upload at 05:57:35 UTC, dataset commit `cc9a7c73`, file `logs/2026-11-08/20261108T054735Z-tiny-llm-05.jsonl` (name carries the Space name, as D-031 needs). Chart fix done in `status.html`: each line series keeps its path, and a point with no neighbour on either side gets an r3 dot (the newest point keeps its r4 dot, no double). Checked by running the page's `drawChart` in node with stubs: lone/gap/last, lone in the middle, continuous (unchanged path), single point, all lone, empty; all as expected. Full suite 470 passed. Next: commit + push (ask), CI, then `push_space.py --yes` again (ask) and re-check `/status`.
  - 2026-11-08: committed `a2fd365` (CI success); dry run changed only status.html; `--yes` uploaded Space commit `9cc801ff` (record `runs/space/push_20261108T060724Z.json`); rebuild to RUNNING in 31 s; smoke 8/8 passed; live `/stats` version `...@4efa5367+a2fd365`; live `/status` serves the fix. The restart cleared the in-memory history, so the visual check needs two requests a few minutes apart. Next: the user's visual check of `/status`, then the D-031 Dev Mode test.
  - 2026-11-08: visual check passed: the user's screenshots show a line across adjacent minutes and a separate dot for the lone minute; live `/stats/history` matched (minutes_ago 4: 2 req, latency p95 1.19 s; 3: 1 req, 1.68 s; 1: 1 req, 1.25 s). Item 6 done. Finding (cosmetic): minute buckets slide (`obs.py` history: `ago = int((now - t) // 60)`), so a request can move to the next bucket between refreshes and groupings differ between screenshots. Recommended: PARKING_LOT (pending the user's call). Next: present the D-031 Dev Mode test plan and wait for a go-ahead (it creates a private Space on HF).
  - 2026-11-08: the user's next screenshots showed the same 4 requests as 1-1-1-1 then 1-2-1; the user chose to fix it now (it may hint that something is wrong). Plan: `history()` buckets by wall-clock minute (`self._wall`), `covered` to match; a new test refreshes at two moments and checks a request keeps its minute; full suite; then commit + push (ask), `push_space.py --yes` (ask), re-check `/status`. The D-031 Dev Mode test plan (steps 0-5) is presented and waits for its own go-ahead after this.
  - 2026-11-08: fix done locally: `history()` maps each request's monotonic age onto the wall clock and buckets by clock minute; `covered` = the server started before that minute ended; the oldest minute can be partial (hour window pruned by age). New test `test_minutes_follow_the_clock_not_the_refresh` fails on the old code (`{1: 1, 0: 1} != {1: 2}`) and passes now; full suite 471 passed; hygiene + diffcheck clean. Next: commit + push (ask), CI, `push_space.py --yes` (ask), re-check `/status`.
  - 2026-11-08: committed `5eac619` (CI success); dry run: HEAD 5eac619, `ARG GIT_COMMIT=5eac619`, 36 files, checks ok. `--yes` was blocked by the Claude Code permission check, so the Space still runs `a2fd365`. Next: the user runs `push_space.py --yes` (or allows it), then I re-check `/status`.
  - 2026-11-08: the user ran `push_space.py --yes`: Space commit `015487fd` (record `runs/space/push_20261108T062725Z.json`); RUNNING on sha 015487fd; smoke passed; live `/stats` version `...@4efa5367+5eac619`. Live check: 3 requests 06:29:06-06:29:59 UTC, then 8 `/stats/history` refreshes over ~100 s (across two minute boundaries) all gave the same groups {06:28: 1 (smoke), 06:29: 2, 06:30: 1}; closed minutes never changed. A request is bucketed by when it finishes (the 06:29:59 one landed in 06:30). Fix done. Next: the user's look at `/status`, then the D-031 Dev Mode test (steps 0-5, own go-ahead; step 0 creates a private Space).
  - 2026-11-08: D-031 Dev Mode test plan (the user said go for step 0; each later step that changes HF asks again): 0) duplicate `CoolDip00/tiny-llm` as private `CoolDip00/tiny-llm-dmtest`, cpu-basic, no secrets, no LOG_DATASET (logs off); pass = RUNNING + Dev Mode toggle in settings. 1) the user makes two fine-grained tokens scoped to that Space only (read, write) and adds them as its secrets REPLICA_TOKEN / REPLICA_ADMIN_TOKEN in the browser. 2) from WSL: no token -> 401/404, token -> 200 `/health` + a completion. 3) Dev Mode on (restarts); in its shell call its own `*.hf.space` URL with $REPLICA_TOKEN, plain + streamed; pass = outbound OK, auth OK, tokens arrive one by one. 4) pause, see what a router would get, `restart_space` with the write token, time to 200 (pass: clear error, <~2 min). 5) Dev Mode off; delete or keep idle to 2026-11-10 for a real 48 h sleep check (ask); go/no-go in PROGRESS. Go = 2-4 pass and no HF flag; else D-031 -> v1.1. Starting step 0.
  - 2026-11-08: step 0 done (pending the user's look for the Dev Mode toggle): `duplicate_repo(..., repo_type="space", private=True, space_hardware="cpu-basic")` (hub 2.4.0 has no `duplicate_space`) created `CoolDip00/tiny-llm-dmtest`, private, sha 015487fd; no variables or secrets were copied (logs off); sleep 172800 s. BUILDING -> RUNNING in 62 s; log: model loaded `...+5eac619` threads 2, server ready, salt falls back to HF's `SPACE_SIGNING_SECRET`, no log-upload line. Step 1 change: only the read token now (secret REPLICA_TOKEN); the write token waits for step 4, where the user runs the restart in their own terminal with the token typed into a hidden prompt (the paused Space can't run it).
  - 2026-11-08: the user turned Dev Mode on: BUILD_ERROR, `git: not found` (Dev Mode's own layer runs `git config`). HF docs (hub-docs `spaces-dev-mode.md`, Docker Spaces): needs bash, curl, wget, procps, git, git-lfs; code in /app owned by uid 1000; a CMD. Our slim image has no git and WORKDIR makes /app root-owned. Fix for the test copy only (repo Dockerfile unchanged, the live image stays lean): scratch `Dockerfile.devmode` = HEAD Dockerfile + `apt-get install bash git git-lfs wget curl procps` + `mkdir -p /app && chown user /app`, `ARG GIT_COMMIT=5eac619`; upload to dmtest only (ask). Token: the user's form had the Read-Only preset on all of CoolDip00's repos; must be customised to only `CoolDip00/tiny-llm-dmtest` read.
  - 2026-11-08: upload done (user OK, "nothing lost"): `create_commit` to dmtest only, pinned with `parent_commit`; test commit `1c8b2a89` replaced only Dockerfile (37 files, same set); live `CoolDip00/tiny-llm` sha 015487fd before and after. Rebuild RUNNING in 47 s, runtime sha 1c8b2a89, devMode True; log: model loaded, server ready (Dev Mode's daemon restarted the app once, 06:45:39 UTC). Token form: the user still had the user-level "Read contents of your repos" + "public gated repos" ticked; told to untick both, keep only the dmtest repo read. Next: the user creates the token, adds secret REPLICA_TOKEN, sends a screenshot; then step 2.
  - 2026-11-08: step 1 done: the user created the fine-grained token (read on `spaces/CoolDip00/tiny-llm-dmtest` only, no user/org scopes) and added it as dmtest **Secret** `REPLICA_TOKEN` (no variables). Settings also show "Synchronize duplicated Space: 1 commit ahead": never click it (it would pull the live Dockerfile over the Dev Mode one). Next: step 2 from WSL (read-only: /health without auth must be refused, with the owner login 200 + one completion).
  - 2026-11-08: step 2 PASS (WSL, read-only, owner login via `build_hf_headers`, never printed): dmtest RUNNING sha 1c8b2a89 devMode True; `/health` without auth -> 404 (HF page, private Space hidden), with auth -> 200 `{"status":"ok"}`; greedy completion 24 tokens -> 200 in 1.6 s (TinyStories text). Next: step 3, the user runs curl in the Dev Mode VS Code terminal with `$REPLICA_TOKEN` (plain + streamed) and sends a screenshot.
  - 2026-11-08: step 3 partial (user, Dev Mode VS Code web terminal, shell is /bin/sh): `$REPLICA_TOKEN` set; from inside the Space to its own hf.space URL: /health no token 404, with token 200 `{"status":"ok"}`; plain completion 200, same greedy text as step 2. Streamed check failed only on my script (`${l:0:70}` is bash-only: "Bad substitution"); re-running with a sh-safe line.
  - 2026-11-08: step 3 PASS: sh-safe streamed re-run from the Dev Mode terminal: SSE chunks one per token, 06:59:22.050 -> .368 UTC (~15-20 ms apart, not batched), ends `data: [DONE]`. So a Space can call another private Space's hf.space URL with a read-only fine-grained token, plain and streamed. Next: step 4 (pause, what a router sees, restart with a dmtest-scoped write token, time to 200): plan shown to the user, waiting for go-ahead.
  - 2026-11-08: step 4a done (user OK): paused dmtest from WSL (owner login) -> PAUSED (devMode flag still True). What a router sees, ~1.1 s each: no auth -> 404 HF HTML page; with auth -> **503 text/html "The space is paused, ask a maintainer to restart it"** (/health and POST /v1/completions alike). So the router must treat 503 + text/html (not our JSON) as "replica down", fail over, and only then try `restart_space`. Live `CoolDip00/tiny-llm` RUNNING 015487fd before and after. Next: the user creates the dmtest-scoped write token and runs scratchpad `wake.py` (hidden prompt), pastes the output.
  - 2026-11-08: step 4b PASS: the user made a fine-grained token (`CoolDip00/tiny-llm-dmtest-write`: dmtest only, read + write contents/settings; HF auto-ticks "view access requests", harmless, Space not gated; no user/org scopes) and ran scratchpad `wake.py` (hidden prompt). PAUSED -> `restart_space` accepted -> 21.6 s /health ReadTimeout (20 s timeout) stage APP_STARTING -> 28.3 s **502** stage RUNNING -> **35.1 s 200**. So a replica-scoped token can wake its own replica, ~35 s cold (model load). Router rules from this: short timeouts; 503 text/html = paused, 502 = starting, hang = starting; fail over at once, never wait on a waking replica inside a request. Next: step 5 (Dev Mode off, sleep check / delete, token cleanup, go/no-go): ask the user.
  - 2026-11-08: **D-031 verdict: GO** (steps 1-4 pass, nothing flagged by HF). Step 5 agreed with the user: (1) both test tokens deleted (`CoolDip00/tiny-llm-dmtest`, `-dmtest-write`; kept `tiny-llm-space-logs-writer` = live HF_TOKEN, `tiny-llm-laptop` = WSL login, unrelated `Langchain_course`). The dead `REPLICA_TOKEN` secret stays on dmtest (removing it would restart the Space and reset the sleep clock) until the Space is deleted. (2) Dev Mode off: still on at 07:14 UTC (dmtest RUNNING 1c8b2a89; live RUNNING 015487fd untouched). (3) Then no visits to the dmtest page; (4) 2026-11-10 after ~09:00 UTC: check SLEEPING, one request, time it, then delete dmtest (ask). Real replicas must be duplicated from `CoolDip00/tiny-llm`, never from dmtest (test-only Dev Mode Dockerfile).
  - 2026-11-08: Dev Mode off (user), tabs closed. 07:15 UTC: dmtest RUNNING 1c8b2a89 devMode False, sleep 172800 s; live RUNNING 015487fd untouched. Sleep clock starts ~07:15 UTC -> expect SLEEPING from ~2026-11-10 07:15 UTC; check after 09:00 UTC (one runtime read + one timed request, then ask to delete dmtest). No more dmtest checks until then. D-031 test otherwise done. Next: item 7 (scaling design in SERVING.md) with the GO result.
  - 2026-11-08: **item 7 started (user: "yes", plan below; a/b/c taken as my recommendations: private replicas, per-replica HF capacity measured in D-031's load test, item 7 committed on its own).** Files: docs/SERVING.md (main), DECISIONS.md (D-031 amendment, original text untouched), PLAN.md + PROGRESS.md. No code, no Space changes. Sub-plan:
    0. PROGRESS note (this). 
    1. Read-only research, short report to the user before writing: (a) HF terms on several Spaces serving one app (D-031 requires it before replicas exist); (b) current HF hardware prices (paid CPU tiers, T4) for the scale-up option (shown only; the $9 budget rules them out).
    2. Rewrite SERVING.md "Scaling beyond one server": one replica's capacity (rated 2.0 req/s ~235 tok/s, SLOTS 8 / queue 4; CPU decode is the limit, not memory; concurrent users + visitors/day with the formula; caveat laptop Docker numbers, HF vCPUs measured in D-031's load test) / free-tier limits (48 h sleep; cold start 35 s paused->200 measured today, sleep-wake after 11-10; no autoscaling; shared CPU) / options: scale up (prices) vs scale out (D-031) / D-031 test results table (steps 1-4) / router design from the test: poll /ready + /stats every few s with short timeouts; reply rules 503 HTML = paused, 502 or hang = starting, 404 HTML = token/config bug (alert, no retry), 429 JSON = busy -> other replica; shortest queue; one retry on another replica only before the first byte; wake in background, rate-limited, never inside a request; X-Replica + pass X-Model-Version; rate limit + visitor hash at the router, hashed id forwarded to replicas; router's own SLO = user-facing, status shows per-replica state; autoscaling = queue-depth rule, documented only (free tier: wake/pause); capacity = load / (2 req/s x headroom) / still open: sleep-wake (11-10), per-replica HF capacity. Small edits: "What is deployed where" (router + replicas planned); laptop caveat line -> points to D-031 load test (item 6 ran smoke checks, not a load test).
    3. DECISIONS D-031 amendment: test results; replicas private; one router token, read + write on the replica Spaces only, as a router Space secret; reply rules; real replicas duplicated from CoolDip00/tiny-llm, never from dmtest.
    4. Checks: hygiene + diffcheck (no personal paths/tokens), full test suite, show the user the SERVING.md diff before commit.
    5. PLAN tick item 7, PROGRESS note, commit + push (ask). Estimate 1.5-2 h. Then item 8, then build D-031.
  - 2026-11-08: **item 7 step 1 done (research, read-only).** Read: ToS (eff. 2022-09-15), Content Policy (eff. 2025-04-10), Supplemental Terms index (eff. 2025-04-28) + its Hub Premium Features PDF (eff. 2026-11-14: billing/credits only), docs spaces-overview, spaces-gpus, spaces-zerogpu, pricing. (a) Terms: nothing forbids several Spaces or a router Space; no Space-count limit for PRO. Risks: Content Policy s.5 'Abuse or interference' lists proxies 'to bypass restrictions' and 'excessive bulk activity'; ToS allows suspension without cause; HF's own replicas feature is 'only available for upgraded (paid) hardware'. Design consequences: no keep-alive pings, replicas sleep normally, wake only on real traffic, small fixed replica count (2), framed as a learning demo, not a way round paid replicas. (b) Prices (per hour): cpu-basic 2 vCPU/16 GB free; cpu-upgrade 8 vCPU/32 GB $0.03 (~$22/mo); T4 small 4 vCPU/15 GB/16 GB GPU $0.40; T4 medium 8/30/16 $0.60; L4 8/30/24 $0.80. Billed only Starting/Running; paid hardware can set custom sleep, sleep and pause not billed. ZeroGPU: Gradio SDK only per its doc (pricing page says Docker too; conflict noted). PRO $9 gives no compute-hour credit. Waiting for the user's go-ahead on step 2.
  - 2026-11-08: **item 7 step 2 started (user: "yes carefully"; the three terms limits go into the design).** Editing docs/SERVING.md only: rewrite "Scaling beyond one server"; small edits to Constraints (line 9 says no load balancer), "What is deployed where" (router + replicas planned) and the laptop-caveat line in the 2-CPU results. Extra points found while reading the code: two kinds of 429 (rate_limit_error vs server_busy: retry only on busy); polling must stop when there is no visitor traffic, or the router itself would keep replicas awake; replicas need the router's hashed client id, or every request looks like one client. Nothing committed until step 4 shows the diff.
  - 2026-11-08: **item 7 step 2 done (docs/SERVING.md edited, uncommitted).** Scaling section rewritten; Constraints line, deployed-where row and the laptop-caveat line updated. Next: step 3 (D-031 amendment in DECISIONS.md), after the user's go-ahead.
  - 2026-11-08: **item 7 step 3 started (user: "yes carefully").** Adding a dated Amendment bullet to D-031 in DECISIONS.md (original text untouched; same form as the amendments on D-0xx at lines 66, 315, 337, 357). Also one line in SERVING.md "Still open": which public link the README and model card use once the router exists (recommendation: item 8 links CoolDip00/tiny-llm now, switch to the router Space once D-031 works, tiny-llm stays up as fallback). Replica count confirmed by the user: 2 fixed. Spaces after D-031: router (public), 2 replicas (private), tiny-llm (stays), dmtest deleted after the 11-10 check.
  - 2026-11-08: **item 7 step 3 done (DECISIONS.md D-031 amendment + SERVING.md "Still open" line, uncommitted).** D-031's original text untouched. Next: step 4 (hygiene + diffcheck, full test suite, show the user the diff), after the user's go-ahead.
  - 2026-11-08: **item 7 step 4 checks passed.** Hygiene: no CRLF, no token-like strings; the only path hit is the old committed line 90 of DECISIONS.md (D-0xx text, not from this work). Diffcheck on added lines: clean. Tests with the GPU hidden (CUDA_VISIBLE_DEVICES empty): 323 passed, 148 skipped in 48.38 s. SERVING.md diff shown to the user. Next: step 5 (PLAN tick, PROGRESS note, commit + push), after the user's go-ahead.
  - 2026-11-08: **item 7 done (user approved the SERVING.md diff: "yes carefully").** After the checks, three sentences in SERVING.md were tightened (what is measured; HF doesn't state the CPU is shared; the no-auto-creation limit is ours, not HF's); docs only, diffcheck re-run clean, tests not re-run. PLAN item 7 ticked; items 5 and 6 still have no done mark in PLAN (tick them with item 8). Commit + push of DECISIONS, PLAN, PROGRESS, docs/SERVING.md: see git log. Next: item 8 (links + docs), then build D-031. Dmtest sleep check: 2026-11-10 after 09:00 UTC.
  - 2026-11-08: Item 7 pushed as `31e259d`, CI green. **Item 8 started (user: "yes carefully"); then a D-031 sub-plan for approval.** Sub-plan:
    0. PROGRESS note (this).
    1. README: "Try it" link to the live demo, CoolDip00/tiny-llm.
    2. hf/MODEL_CARD.md: Space link. Re-upload of the card changes the public model page: show the change and ask first.
    3. PARKING_LOT: README item 3 done.
    4. PLAN + PROGRESS: tick items 5, 6, 8, then step 5 and 6.5b (including the 4.9 tick). The 11-10 sleep check stays pending as a D-031 task.
    5. Checks: hygiene + diffcheck, full test suite with the GPU hidden, show the user the diff.
    6. Commit + push (ask), wait for CI.
    D-031 after that, in three stages: (1) router code + CPU tests against fake replicas, no HF changes, can start before the 11-10 check; (2) create 2 private replicas + router Space (ask; the user creates the router token in the browser); (3) 1-vs-2 load test through the router (ask). Detailed sub-plan for stage 1 goes to the user and into PROGRESS before any code.
  - 2026-11-08: **item 8 edits made (uncommitted).** README "Try it" line above Highlights (says the speed numbers are the laptop GPU's, per D-033); MODEL_CARD Links gains the Space; PARKING_LOT README item 3 done; PLAN ticks 4.9, 6.5b, step 5, items 5, 6, 8 (6.5b's tick says the D-031 router and the 11-10 sleep check stay open; item 6's says the gentle live load test moved to D-031). Card upload is one file, README.md, with a one-off script after the commit (ask); `upload_model.py` would resend all six files. Safe for the Space: the Dockerfile pins the model at Hub commit `b11100b`.
  - 2026-11-08: Item 8 pushed as `e9be9e3`, CI green (about 91 s; before the commit 323 passed, 148 skipped with the GPU hidden). **Still pending: the model card upload.** The user approved it, but the permission check blocked it from the agent, so the user runs the one-off script themselves. It uploads only README.md to the model repo and stops if the Hub card differs from `hf/MODEL_CARD.md` by anything except the new Live demo line. Next after that: D-031 stage 1 sub-plan for approval, before any code.
  - 2026-11-08: **Item 8 done; 6.5b step 5 complete.** The user ran the card upload: Hub commit `fd9804e` on `CoolDip00/tiny-llm-tinystories-30m`, one line added, only README.md changed, Hub card equals `hf/MODEL_CARD.md`, and the Space's pinned revision `b11100b` still resolves. Next: D-031 stage 1 sub-plan for approval, before any code.
  - 2026-11-08: **D-031 stage 1 sub-plan (APPROVED 2026-11-08, separate `router.py`).** Design: docs/SERVING.md "Router design". No HF changes in this stage. CPU only. One step at a time: report, then wait.
    Structure: new `src/tinyllm/router.py` (class `ReplicaRouter`), its own process, no torch, instead of the "remote-replica mode" PLAN.md describes: server.py imports torch and its handler drives a local worker. `server.Router` (variant picking, 6.9) stays. Tests: `tests/test_router.py`, scripted fake replicas on real uvicorn threads, injected clock.
    0. PROGRESS note (this).
    1. Groundwork, no behaviour change: move the torch-free helpers (`RateLimiter`, `forwarded_ip`, `resolve_salt`, `HeadersMiddleware`, `_page`, `_error_response`) from server.py to a new torch-free module, imported back so existing tests still pass. httpx moves from dev to runtime deps (uv.lock). Test: importing the router does not load torch.
    2. Replica side: `--trust-router` (off by default), server takes the visitor hash from a router header only when on, format-checked. Off is essential: the public `tiny-llm` Space runs this image and a visitor could otherwise send the header to dodge the rate limit. Dockerfile env `TRUST_ROUTER`. Nothing live changes until a push_space.
    3. Pure logic: `classify(reply)` as one table (the SERVING.md reply table; the 11-10 sleep result becomes one row + one test), and pick (polled outstanding + sent since poll, ties round-robin, excludes the replica already tried).
    4. Request path, POST /v1/completions only: rate limit + hash at the router, connect timeout 3 s, at most one retry before the first byte (429 server_busy, 503, 502, connection error), stream passthrough, `X-Replica`, `X-Model-Version`, `X-Request-ID`. A failure after the first byte ends the stream with the server's own SSE error event. Visitor leaves: replica request closed (streaming and non-streaming). Safety: no redirects followed, visitor Authorization never forwarded, replica token and private URLs never in a response, body size cap (413), only allow-listed routes.
    5. Polling and waking: polling starts only on a POST /v1/completions and stops 10 min after the last one; the first request after idle awaits one poll round (2 s max). Pages, /stats, /stats/history, /v1/models, /health, /ready are passive: they never poll, wake, or count as traffic, so an open status tab can't keep replicas awake. Wake is an injected function (default `restart_space` in a thread), background, max once per replica per 5 min, only when no replica is ready or the ready ones shed and another is asleep; never on 404. No replica ready: 503 + `Retry-After: 30`, type `waking_up` (the demo page already handles 503).
    6. Router pages and metrics: same demo + status pages; `Metrics` at the router (TTFT = first token sent to the visitor; non-streaming = the full response, which is when the visitor gets its first token; a waking 503 counts against availability); engine chart fed from polls; status page gains a replica table (state, queue, poll age, or "not polled"), shown only when the field is present; /v1/models from the last poll or `--model-label`; same `--log-dir/--log-dataset` as the server. CLI: `python -m tinyllm.router --replica name=url` (twice), token from env `TINYLLM_REPLICA_TOKEN` (never a CLI flag).
    7. Real end-to-end on the laptop CPU: router + 2 real `tinyllm.server` processes (`--device cpu`, HF cache), `smoke_server.py --url` against the router, kill one replica mid-run and check the other takes over. No GPU.
    8. Docs (SERVING.md: built vs not; PLAN), checks (hygiene, diffcheck, full tests GPU hidden), show the diff, commit + push (ask), CI.
    Tests planned: every reply-table row; pick + tie order + stale count; retry happens once, never after the first byte, never on 400/504 or rate_limit_error; stream equals direct against a real `serve()` replica; disconnect reaches the replica; spoofed hash ignored with the flag off, kept with it on; token/URL never leak; status tab causes no polls; polling stops at 10 min; wake throttle, no wake on 404; all down gives 503 + Retry-After.
    Not in stage 1: Dockerfile for the router, push script, creating Spaces, tokens, load test (stages 2 and 3); whether replicas ship logs (stage 2). Estimate 3-4 h.
  - 2026-11-08: D-031 stage 1 **step 1 done (uncommitted).** `RateLimiter`, `forwarded_ip`, `resolve_salt`, `HeadersMiddleware`, `read_page` (was `_page`), `error_response` (was `_error_response`) moved unchanged to new `src/tinyllm/web_common.py` and imported back into server.py; new `tests/test_router.py` proves the module loads without torch; httpx moved from dev to runtime deps (uv.lock: only that entry moved); README layout names the module. No behaviour change. Next: step 2 `--trust-router`.
  - 2026-11-08: D-031 stage 1 **step 2 done (uncommitted).** `--trust-router` / `ServerConfig.trust_router` / Dockerfile `TRUST_ROUTER` (off when empty). Header `X-Tinyllm-Client`, 16 lowercase hex characters (`web_common.CLIENT_HEADER`, `router_client`); a missing or malformed value falls back to the address. Tests: header ignored by default, honoured with the flag on, malformed falls back. Live Space unchanged until a push. Next: step 3 classify + pick.
  - 2026-11-08: D-031 stage 1 **step 3 done (uncommitted).** `router.py`: `Verdict`, `RETRYABLE`, `Reply`, `classify`, `Replica`, `Picker`; no I/O. Beyond the SERVING.md table: 401/403 = MISCONFIGURED; HTML or other non-JSON pages that aren't listed = DOWN (never woken, never shown to a visitor), including a 200 HTML page, in case a sleeping Space answers that way. SERVING.md's table gets those rows in step 8. Next: step 4 request path.
  - 2026-11-08: D-031 stage 1 **step 4 done (uncommitted).** `router.py`: `RouterConfig`, `ReplicaRouter`, `create_router_app`, POST /v1/completions only. Also `web_common.client_hash` (shared with server.py), 3xx = DOWN. Rules: ReadTimeout = 504, no retry; any BUSY seen = 429 `server_busy`; otherwise 503 `waking_up`, `Retry-After: 30`. Next: step 5 polling and waking (no wake is triggered yet). Step 6 CLI must call uvicorn with `proxy_headers=False`: by default uvicorn believes X-Forwarded-For from loopback, and `client_hash` does its own parsing.
  - 2026-11-08: D-031 stage 1 **step 5 done (uncommitted).** `router.py`: `traffic` (starts polling, first request after idle waits for one round), `_poll_loop`, `poll` (/ready + /stats at once, 2 s in total), `wake_asleep`, `hf_wake(token)`. Only a request that passed the rate limit and body cap is traffic. Wake only for verdict ASLEEP (restarting a running Space would kill its requests), from the request path only, when no replica is ready or one shed; all paused replicas are woken, each at most once per 5 min. /ready 200 with unreadable /stats stays ready and keeps its load estimate. `create_router_app(wake=None)` never wakes: only the step 6 CLI passes `hf_wake`, and it must give each replica its `space_id` (`Replica.space_id`), so `--replica` needs the Space id as well as the URL. Stage 2 note: `huggingface_hub` comes in only through `tokenizers`, so the router image must install it. Next: step 6 pages, metrics, CLI.
  - 2026-11-08: D-031 stage 1 **step 6 done (uncommitted).** `router.py`: `/`, `/status`, `/health`, `/ready`, `/v1/models`, `/stats`, `/stats/history` (all passive: never poll, wake or count as traffic), `describe` adds `variants`, `polling`, `replicas` (names and states only, never a URL or Space id), `_snapshot` feeds the load chart from polls (no snapshot when no replica is ready: a gap is the truth), CLI `python -m tinyllm.router --replica name=url[,owner/space]` (token only from `$TINYLLM_REPLICA_TOKEN`, never a flag; token set needs every Space id; `proxy_headers=False`). `status.html`: Replicas panel, hidden on a single server. Tests: 137 router, full suite 462 passed, 148 skipped (GPU hidden); router tests 3 runs in a row; hygiene and diffcheck clean; status.html script parses under node. Next: step 7 real end-to-end on laptop CPU (router + 2 real `tinyllm.server --device cpu`, `smoke_server.py --url`, kill one replica).
  - 2026-11-08: D-031 stage 1 **step 7 done** (no code change; script not in the repo, code still uncommitted). Router + 2 real `tinyllm.server --device cpu --ckpt checkpoints/main/best.pt --trust-router` on localhost, no token: 20/20 checks. `smoke_server.py --url` passed through the router; streaming ended in `[DONE]`; 12 concurrent requests all 200 and split 6/6; SIGKILL of r2: the next 8 requests + 1 stream all succeeded (one internal retry), `/stats` showed r2 down and capacity 4; restarted r2 came back to ok by itself; both killed: 503 + Retry-After 30 in 0.01 s, router `/health` still ok; prompt marker in no log. Clean SIGTERM exits -15 (uvicorn 0.54 re-raises the signal after shutdown), with `stopped` as the last log line; a stage 2 health check must not read -15 as a crash. The HF cache on the laptop has only the model README, not the weights, so `--offline --hf-model` fails here (hence `--ckpt`). Not exercised: waking (needs HF), a browser view of the Replicas panel. Known wording: with replicas truly down and no waker the 503 still says `waking up`. Next: step 8 docs (SERVING.md built vs not, new table rows; PLAN), checks, show diff, ask before commit + push.
  - 2026-11-08: D-031 stage 1 **step 8 done (uncommitted).** `docs/SERVING.md`: section intro says built but not deployed, reply table rows (401/403, other HTML or non-JSON, 3xx, other JSON), waking note, new "Built so far". `PLAN.md`: stage 1 line. `README.md`: layout row. Full suite 462 passed, 148 skipped; hygiene and diffcheck clean. Next: show the diff, ask before commit + push, then CI. Then stage 2 (ask).
  - 2026-11-08: **D-031 stage 1 committed and pushed: `a16d5b3`, CI passed.** The live Space is unchanged (GitHub only). Next: write the stage 2 sub-plan (ask first; it changes HF). Stage 2 must remember: the live Space's code has no `--trust-router`, so replicas duplicated from it need this commit's code pushed; decide whether replicas ship logs; router image installs `huggingface_hub`; `SLO_TTFT_S=2` for the router.
  - **D-031 stage 2 + 3 sub-plan (rewritten 2026-11-08 after the user's decisions; report and ask after each step; anything that changes HF is run or approved by the user; never paste a token or secret in chat).** Cost: free `cpu-basic` only, no paid hardware.
    - **Decided by the user (2026-11-08):** the end state is **three Spaces**: `CoolDip00/tiny-llm` (public; visitors use it; it becomes the router and serves the demo + status pages), `CoolDip00/tiny-llm-replica-1` and `-2` (private, run the model). No separate front-end Space, no separate fallback demo. README and model card links never change. Replicas are created empty and get this repo's code by push, not duplicated from `tiny-llm` (duplicating copies old code without `--trust-router` and adds a 'Synchronize' button). Only the router ships logs, to the existing private dataset as `router`; replicas ship nothing and hold no secret. All building and testing happens in a **private temporary Space `CoolDip00/tiny-llm-router`**, so the public demo is untouched until the takeover; that Space is deleted afterwards (ask).
    - **Step 1 (repo only):** `Dockerfile.router`: no torch, pins from uv.lock (fastapi, starlette, pydantic, uvicorn, httpx, huggingface-hub), fails the build if torch can be imported; `--trust-proxy` on a Space, `--slo-ttft-s 2`, `--log-replica router`, log upload only when `LOG_DATASET` is set. Replica list from `$REPLICAS` (space-separated `name=url,owner/space`), a Space **secret** so the private URLs are never in a public file or setting. `space/router/README.md` (card of the public `tiny-llm` after takeover: a learning demo of routing, what is logged) and `space/replica/README.md`.
    - **Step 2 (repo only):** `scripts/push_space.py --role server|router|replica`, **role required** (after the takeover a default of `server` could silently undo the router). Today the script always sets `LOG_DATASET` and uploads `space/README.md`; the replica role must never set `LOG_DATASET`. `--create` for a replica: private, variables `TRUST_ROUTER=1`, `RATE_LIMIT=0`. `--create` for the router: private. Router role uploads `Dockerfile.router` as `Dockerfile`, `space/router/README.md` as `README.md`, no tokenizer. Tests for each role's file list and variables.
    - **Step 3 (local):** build the router image, run it against 2 local CPU servers as in stage 1 step 7, smoke check, confirm no torch in the image. Then docs, checks, show diff, ask, commit + push, CI (`push_space.py` refuses a dirty tree, so PROGRESS must be committed first).
    - **Step 4 (HF, replicas):** dry run shown first. User runs `--create` and the push for both replicas (the classifier blocks these for me). Wait for the builds, then check `/ready` and `/stats` of each with the laptop login (no token on any command line): right commit, `norm_backend` torch, private (401/404 without a token).
    - **Step 5 (HF, private router):** user creates in the browser one fine-grained token, read + write on the 2 replica Spaces only. `tiny-llm-router` created private; the user sets its secrets in the browser, never in chat: `TINYLLM_REPLICA_TOKEN`, `REPLICAS`, `TINYLLM_LOG_SALT` (same salt as `tiny-llm`), `HF_TOKEN` (`tiny-llm-space-logs-writer`, or a new token if its value is not at hand). Variable `LOG_DATASET`. Then push and wait for the build.
    - **Step 6 (HF, real checks through the private router):** smoke check; requests reach both replicas (`X-Replica`); visitor hash forwarded; pause both replicas, first request gets a 503 with `Retry-After`, the router wakes them with its token, a later request gets 200 (first real test of waking and of the token's write access); status page; no URL or token in any public route; a `router` log file in the dataset with no prompt text. User looks at the Replicas panel in a browser. Commit notes.
    - **Stage 3 (HF, load test through the private router, ask first):** goodput with 1 vs 2 replicas, labelled a routing demo. Needs the load generator to send the laptop login to a private Space (check it can, without a token on the command line), and the temporary router's `RATE_LIMIT=0` for the run, otherwise the 20/min limit is what gets measured. Each mode its own output file, JSON behind every number (memory: never overwrite), one session per table. Results in SERVING.md, commit.
    - **Takeover (HF, public, ask first, after stage 3):** add secrets `REPLICAS` and `TINYLLM_REPLICA_TOKEN` to `tiny-llm` (user, browser; `HF_TOKEN`, `TINYLLM_LOG_SALT`, `LOG_DATASET` are already there). `push_space.py --role router --space CoolDip00/tiny-llm`. Smoke check on the public URL, X-Forwarded-For spoof check, status page. Undo: `push_space.py --role server` from a commit, one rebuild, a short outage. Then delete `tiny-llm-router` (ask, look at it first). Docs: SERVING.md 'What is deployed where', 'Built so far', 'Still open', the 'duplicated' line; `tiny-llm` card; PROGRESS; commit + push, CI.
    - **Known limits:** how a *sleeping* (not paused) private Space answers is unknown until the 2026-11-10 dmtest check, so that `classify` row comes after it; replicas can't sleep before then anyway (48 h). Don't touch dmtest or `class-grader`. Then 6.1 → 6.3 → v1.
  - 2026-11-08: D-031 stage 2 **step 1 done (nothing on HF).** `Dockerfile.router`, `space/router/README.md` (card of the public `tiny-llm` after the takeover), `space/replica/README.md`, README layout row. Built locally as `tinyllm-router:s2step1`: about 59 MB, torch not importable (the build fails if it ever is); empty `REPLICAS` refuses to start (exit 2); `REPLICAS` expands to one `--replica` each; `/stats`, `/health`, `/status` answer in the container; stops on `docker stop`. Next: step 2 `push_space.py --role`.
  - 2026-11-08: D-031 stage 2 **step 2 done (nothing on HF).** `scripts/push_space.py`: `--role server|router|replica` and `--space` are both required now. replica: server image, `space/replica/README.md`, created private, variables `TRUST_ROUTER=1` + `RATE_LIMIT=0`, deletes `LOG_DATASET`, `--log-dataset` refused, upload refused if the Space is not private. router: `Dockerfile.router` uploaded as `Dockerfile`, `space/router/README.md`, no tokenizer, created private, `LOG_DATASET`. server and router delete `TRUST_ROUTER`; every role deletes `GIT_COMMIT`. `role` and `variables` recorded in `runs/space/push_*.json`. Tests: `tests/test_push_space.py` 27 passed (file lists, variables, refused arguments, private guard, router pins match `uv.lock`); full suite 483 passed, 148 skipped; hygiene and diffcheck clean. Dry runs for all 4 Spaces shown in a throwaway clone (no network): server and replica 38 files, router 37. Next: step 3 (local router image against 2 CPU servers, docs incl. SERVING.md line 23, commit + push after asking).
  - 2026-11-08: D-031 stage 2 **step 3 done (nothing on HF).** Router image against 2 real CPU servers: 25/25 checks. 59 MB, uid 1000, no torch or tokenizers; CMD expands correctly off and on a Space (`--trust-proxy --proxy-hops 1`, `--log-replica router`); smoke; streaming; 12 concurrent split 6/6; no replica URL in public routes; `kill -9` of one unseen by clients, restart found by itself; both gone: 503 with `Retry-After: 30` in about 4 s; `docker stop` clean (exit 0 in 0.6 s); prompt marker in no log. Local test note: this machine runs Docker Desktop, so `--network host` stays inside its VM and port 9100 is taken on Windows; the test published 18100 and reached the servers at the WSL address. Docs: SERVING.md deploy section and 'Built so far'. Steps 1-3 were committed and pushed together with this note. Next: step 4 (dry run shown first; the user runs the HF commands).
  - 2026-11-08: D-031 stage 2 **step 4 done.** Commit `7ba225f` pushed, CI green. The user created and pushed `CoolDip00/tiny-llm-replica-1` (Space commit `eda80d8`) and `-2` (`2a115ca`). Checks with the laptop login, 16/16: both RUNNING on cpu-basic, private, variables exactly `TRUST_ROUTER=1` + `RATE_LIMIT=0`, `/ready` 200, `/stats` version `tiny-llm-tinystories-30m@4efa5367+7ba225f`, `norm_backend` torch, 8 slots, one 12-token completion each; without a token `/`, `/ready`, `/stats`, `/status` all 404. Next: commit this note (`push_space.py` refuses a dirty tree), then step 5: the user makes one fine-grained token (read + write on the 2 replica Spaces only), runs `--role router --space CoolDip00/tiny-llm-router --create --yes`, sets its 4 secrets in the browser, runs the push.
  - 2026-11-08: D-031 stage 2 **step 5 done (note uncommitted: commit PROGRESS before any later push_space).** The user created the token `tiny-llm-router-replicas` (read + write on the 2 replica Spaces only), the private temporary Space `CoolDip00/tiny-llm-router` (Space commit `24b5a5f`, repo commit `f5f6d5c`) and its 4 secrets. First start failed with exit 2 because the `REPLICAS` secret held the wrong text (a paste slip); the user fixed the value, no code change. Read-only checks 12/12: RUNNING on cpu-basic, private, only `LOG_DATASET`, `/health` 200, sees r1 and r2, not polling before traffic, no replica URL or token in public routes, 404 without a token; startup log: `wake` true, salt from `TINYLLM_LOG_SALT`, `proxy_hops` 1. No completion sent through it yet. Next: step 6 real checks (ask first; it pauses both replicas).
  - 2026-11-08: D-031 stage 2 **step 6 done, 22/22, and the user viewed the status page (note uncommitted: commit PROGRESS before any later push_space).** Through the private router with the laptop login: requests and streaming 200, both replicas served (`X-Replica`), 16 slots, version `...@4efa5367+7ba225f`, torch; visitor hash forwarded (same 16-hex value in router and replica logs); prompt marker in no run log; no IP in request lines; demo and status page with replica data; no replica URL, Space id or token in public routes. Paused both replicas with the laptop login: first request a 503 `waking_up` with `Retry-After: 30` in 0.35 s; the router restarted both with its own token (log: `waking replica` r1 and r2, no failure; first real proof of that token's write access); a request 22.9 s after the first got 200; both back to ok and serving. Router log uploaded to the private dataset (`logs/2026-11-08/20261108T111951Z-router-11.jsonl`, 10 request lines) with no prompt text and no IP. The user's browser view of the router's status page shows r1 and r2 Ready, 8 slots each, and agrees with `/stats/history`: 11 requests, 9 served, 2 failed (the two `waking_up` answers from the deliberate pause), availability 81.82%, burn rate 18.2x. Availability shows Missed only because of that pause, on a very small count. Next: stage 3 (ask first).
  - **D-031 stage 3 sub-plan (written 2026-11-08, waiting for the user's go-ahead; report and ask after each step; the user makes every HF change in the browser; never paste a token or secret in chat).** Goal: goodput (highest rate that meets the SLOs) with 2 replicas vs 1, through the private router, labelled a routing demo. Free `cpu-basic` only. No GPU: the laptop only sends requests. `tiny-llm` (public) is not touched. Estimate: about 1 h for step 1, about 45 min for steps 2-6.
    - **Facts checked in the code:** `scripts/load_test.py` sends no auth header, so it cannot reach a private Space yet. `Dockerfile.router` passes `RATE_LIMIT` to `--rate-limit`, and 0 disables the limiter. **One replica cannot be simulated by pausing r2:** the router wakes a paused replica as soon as r1 sheds a request (`_route` -> `wake_asleep`), which would bring r2 back in the middle of the run. So 1-replica mode means the router's `REPLICAS` secret lists only r1.
    - **Step 1 (repo only):** `load_test.py --hf-login`: reads the laptop login with `huggingface_hub.get_token()`, sends it as a header, refuses any URL that is not `https://*.hf.space` (the token can't go elsewhere) and refuses if there is no login; records only that the flag was on, never the token. Summary gains `by_replica`, counted from `X-Replica`. Tests in `tests/test_load_test.py`. Full tests, hygiene, diffcheck, show the diff, commit + push (ask), CI. No Space push is needed: this script doesn't run on a Space.
    - **Step 2 (user, browser):** add variable `RATE_LIMIT=0` on `tiny-llm-router` only (restarts the router). Check: RUNNING, 16 slots, both replicas ok. It stays until that private Space is deleted at the takeover; the public `tiny-llm` never gets it and keeps 20/min.
    - **Step 3 (pilot):** `--rates 1 --duration 20`, about 24 requests. Confirms the login works, both replicas appear in `by_replica`, and HF's own proxy doesn't refuse one client. Report before going on.
    - **Steps 4-6 (one session, one table):** Run A, both replicas. The user changes `REPLICAS` to r1 only; check 8 slots; Run B. The user restores `REPLICAS` to both; check 16 slots; Run C. A-B-A costs no extra edits (the secret has to be restored anyway) and shows whether HF's shared CPUs drifted between runs. Each run: `--rates 1,2,3,4,6 --duration 60 --max-tokens 128 --slo space --keep-requests --power-mode client-only --hf-login`, GPU hidden, about 6 min, 960 requests. Files in `runs/bench/load/space-router/<date>-{pilot,a-2rep,b-1rep,c-2rep}.{json,log}`, plus `.stats.json` (router `/stats` and `/stats/history` after the run). `load_test.py` refuses to overwrite.
    - **Stop and report if:** a replica isn't ok or the slot count is wrong before a run; a warmup fails; any 429 or error that isn't the router's own JSON (HF's proxy refusing us); failures at 1 req/s.
    - **Step 7 (repo):** table in `docs/SERVING.md` (rate, served, 429, failed, req/s, tok/s, TTFT p95, split by replica), every number from those JSON files; replaces the 'Capacity on HF's hardware is measured in D-031's 1-vs-2 load test' caveat. Show the diff, commit + push (ask), CI. Then the takeover (ask first).
    - **How to read the results:** TTFT is measured on the laptop, so it includes the internet path to HF (the same in every run). 2 replicas need not give 2x: the router and HF's proxy can limit first, and the CPUs are shared. Report what is measured. About 2,900 test requests land in the router's log in the private dataset (no prompts, deleted after 30 days), so usage reports for this date include them.
  - 2026-11-08: D-031 stage 3 **step 1 done, committed `55f5a7a`, CI passed.** `scripts/load_test.py --hf-login`: reads the login with `get_token()`, refuses any URL that is not `https://*.hf.space` before reading it, refuses with no login (exit 2); `workload.hf_login` records only the flag. Summary and the per-rate line gain `by_replica` from `X-Replica`, each request `replica`. `tests/test_load_test.py` +9: header sent on every request, token not in the output file, both replicas counted, 6 refused URLs, no login, empty on a single server. Nothing on HF.
  - 2026-11-08: D-031 stage 3 **step 2 done, step 3 pilot started.** The user added variable `RATE_LIMIT=0` on the private `tiny-llm-router` only (temporary: it goes when that Space is deleted at the takeover; `REPLICAS` still lists both). Results in `runs/bench/load/space-router/` (git-ignored).
  - 2026-11-08: D-031 stage 3 **step 3 pilot done, no stop rule hit.** Pre-check: router RUNNING, `RATE_LIMIT=0`, r1 and r2 ok, 16 slots. 1 req/s for 20 s: 17 sent, 17 served, 0 shed, 0 failed, split r1 9 / r2 8, TTFT p95 1.37 s, 103 tok/s; login works, HF's proxy refused nothing. Files `runs/bench/load/space-router/20261108-pilot.{json,log,stats.json}`. Next: Runs A/B/C after the user's go-ahead.
  - 2026-11-08: D-031 stage 3 **step 4 Run A started** (user approved): both replicas, rates 1,2,3,4,6, 60 s each, to `runs/bench/load/space-router/20261108-a-2rep.{json,log,stats.json}`.
  - 2026-11-08: D-031 stage 3 **step 4 Run A done, no stop rule hit.** Two replicas, 16 slots: rated 4.0 req/s (477 tok/s, TTFT p95 1.86 s); at 6 req/s 297 of 393 served, 96 shed (all the router's own `server_busy`), 0 failed, TTFT p95 3.12 s, 590 tok/s. Split about even at every rate. Next: user sets `REPLICAS` to r1 only on `tiny-llm-router`, check 8 slots, Run B to `20261108-b-1rep`.
  - 2026-11-08: D-031 stage 3 **step 5 Run B started**: user set `REPLICAS` to r1 only on `tiny-llm-router` (MUST be restored to both before Run C and the takeover). Rates 1,2,3,4,6, 60 s each, to `runs/bench/load/space-router/20261108-b-1rep.{json,log,stats.json}`.
  - 2026-11-08: D-031 stage 3 **step 5 Run B done, no stop rule hit.** One replica, 8 slots: rated 1.0 req/s (102 tok/s, TTFT p95 1.26 s); 2 req/s missed on TTFT p95 2.34 s with 1 shed; served peak 2.31 req/s, 295 tok/s at 4 req/s; 383 sheds over the run, all the router's own `server_busy`, 0 failed. Next: user restores `REPLICAS` to both, check 16 slots, Run C to `20261108-c-2rep`.
  - 2026-11-08: D-031 stage 3 **step 6 Run C started**: user restored `REPLICAS` to both on `tiny-llm-router`. Rates 1,2,3,4,6, 60 s each, to `runs/bench/load/space-router/20261108-c-2rep.{json,log,stats.json}`.
  - 2026-11-08: D-031 stage 3 **step 6 Run C done, no stop rule hit; all three runs finished in one session.** Two replicas, 16 slots: rated 3.0 req/s (360 tok/s, TTFT p95 1.32 s); 4 req/s missed on TTFT p95 2.30 s with 1 shed; 6 req/s 279 of 393 served, 114 shed, 0 failed, 546 tok/s. A and C disagree at 4 req/s (A met at 1.86 s), so the figure to quote for two replicas is 3 req/s met in both runs, 4 borderline. Peak served: two replicas 4.27 to 4.61 req/s, one replica 2.31 req/s (1.85x to 2.0x). `REPLICAS` is back to both (16 slots confirmed); `RATE_LIMIT=0` stays until `tiny-llm-router` is deleted. Next: step 7 (table in `docs/SERVING.md`, replacing the caveat), after the user's go-ahead.
  - 2026-11-08: D-031 stage 3 **step 7 done, stage 3 complete, committed `cf598f4`, CI passed**: results table into `docs/SERVING.md` (new section "Results: 1 vs 2 replicas on HF's CPUs", every number generated from the three JSON files), replacing the caveat's pointer and the "Still open" item. Tests 492 passed, 148 skipped; hygiene and diffcheck clean. Next: the takeover (sub-plan first, ask before anything on HF). The takeover also updates the stale "not deployed yet" and "Not done yet" lines in SERVING.md.
  - 2026-11-08: CI: the `docker` job now has `needs: test` (user's request), so the image is only built after the tests pass; before this the two jobs ran in parallel. Committed `bfa6bd7`, CI passed; job times confirm docker started after test finished.
  - 2026-11-08: **D-031 takeover sub-plan** (written before any HF change; nothing done yet). Public `CoolDip00/tiny-llm` becomes the router in front of the two private replicas. Facts checked: `tiny-llm` RUNNING, Space commit `015487f` (running version `...+5eac619`), variables only `LOG_DATASET`; replicas RUNNING; `git diff` from the load-tested router commit `f5f6d5c` and replica commit `7ba225f` to HEAD is empty for `src`, Dockerfiles, `.dockerignore`, `space/router`, `space/replica`, so the router pushed from HEAD is the code that was measured; dry run of the push passes (37 files, 352 KB). The old server does a last log upload during its graceful stop.
    - **Step 1 (user reads):** `space/router/README.md`, which becomes the public page of `tiny-llm`. Wording changes, if any, are committed before the push because the script uploads from HEAD.
    - **Step 2 (user, browser):** on `CoolDip00/tiny-llm` settings add two secrets: `REPLICAS` (both replicas, same value as on `tiny-llm-router`) and `TINYLLM_REPLICA_TOKEN` (value of the `tiny-llm-router-replicas` token; if the user has no copy, stop and decide first). `HF_TOKEN`, `TINYLLM_LOG_SALT`, `LOG_DATASET` are already there. Each save restarts the old server for a minute or so. Never add `RATE_LIMIT` here: the public Space keeps 20/min, burst 5. Check: RUNNING, still the old server, answering.
    - **Step 3 (user, WSL; goes public):** `cd ~/llm && uv run python scripts/push_space.py --role router --space CoolDip00/tiny-llm --yes`. Rebuild, a short outage. Paste the output.
    - **Step 4 (checks, no HF changes):** RUNNING, Space commit recorded, variables still only `LOG_DATASET`; `/stats` shows r1 and r2 ok, 16 slots, version `...+7ba225f`; `scripts/smoke_server.py` against the public URL; `/` and `/status` return 200; rate limit is on: burst of small requests with changing `X-Forwarded-For` headers still gets 429 (proves a visitor can't dodge it, about 30 requests); no private replica URL or Space name in `/stats`, `/stats/history`, `/status`, or the public card; a router log file appears in the private dataset within about 10 min, and the old server's last upload is there.
    - **Stop rule and undo:** if a step 4 check fails and can't be fixed at once, undo with `push_space.py --role server --space CoolDip00/tiny-llm --yes` from HEAD (one rebuild, short outage). The two added secrets are unused by the server and can stay.
    - **Step 5 (ask, look first; irreversible):** delete `CoolDip00/tiny-llm-router` (its `RATE_LIMIT=0` goes with it). Before: confirm it's the right Space, private, and its logs were uploaded. Keep the token `tiny-llm-router-replicas`; `tiny-llm` now uses it.
    - **Step 6 (repo):** `docs/SERVING.md` ('What is deployed where', 'not deployed yet', 'Built so far' and 'Not done yet', 'Still open' public-link item, 'duplicated' line); PROGRESS. Show the diff, commit + push (ask), CI. Then 2026-11-10 dmtest sleep check, then 6.1.
  - 2026-11-08: D-031 takeover **step 1 done**: user read `space/router/README.md`, no wording changes; user has a copy of the `tiny-llm-router-replicas` token. Step 2 started (user adds the two secrets on the public `CoolDip00/tiny-llm`, settings URL `https://huggingface.co/spaces/CoolDip00/tiny-llm/settings`).
  - 2026-11-08: D-031 takeover **step 2 done**: secrets `REPLICAS` and `TINYLLM_REPLICA_TOKEN` added on the public `tiny-llm`; it restarted and is still the old server (`...+5eac619`, Space commit `015487f`), ready, variables only `LOG_DATASET`. Next: step 3, the user runs the router push.
  - 2026-11-08: D-031 takeover **step 3 done, public `tiny-llm` is now the router** (user pushed from `9e83e17`; Space commit `b446ce4`). **Step 4 live checks passed:** public, variables only `LOG_DATASET`, r1 and r2 ok, 16 slots, replicas on `7ba225f`, `smoke_server.py` passed on the public URL, `/` `/status` `/stats` `/stats/history` 200; 30 requests with faked `X-Forwarded-For` got 5 served and 25 `rate_limit_error`; no private replica name or token in public pages, README, Dockerfile or run log. Still to check: first router log file in the private dataset. Then step 5 (delete `tiny-llm-router`, ask first), step 6 docs.
  - 2026-11-08: D-031 takeover **step 4 done, all checks passed.** Router log `20261108T133722Z-router-13.jsonl` reached the private dataset: 32 request rows (7 served, 25 rate limited, matching the check traffic), no prompt or text field, no raw or faked IP, no private replica URL; `salt` is only the source's name. Old server's last upload `20261108T133104Z-tiny-llm-13.jsonl` ends with `shutting down: draining` and `stopped`. Next: step 5 (delete `tiny-llm-router`, ask first).
  - 2026-11-08: D-031 takeover **step 5 done**: temporary private `CoolDip00/tiny-llm-router` deleted by the user in the browser (it was on `24b5a5f`, idle). API confirms it is gone; public `tiny-llm` still serves through r1 and r2 (16 slots). Token `tiny-llm-router-replicas` kept, `tiny-llm` uses it. End state: three Spaces. Next: step 6, stale lines in `docs/SERVING.md`.
  - 2026-11-08: D-031 takeover **step 6 sub-plan (not started, waiting for go-ahead).** `docs/SERVING.md`: 'What is deployed where' row becomes live (line 18); 'not deployed yet' paragraph (262); 'Replicas are duplicated from' becomes pushed with `push_space.py` (326); 'tested on the laptop, not deployed' (381); 'Not done yet' list keeps only what is still open (383); remove the two answered 'Still open' items (394: replicas ship no logs, they have no secrets, the router logs requests; 395: public link stays `CoolDip00/tiny-llm`, which is now the router, no fallback Space). `DECISIONS.md`: one dated amendment under D-031 (the existing public Space became the router, three Spaces in the end, the temporary private router is deleted). PROGRESS note. Then show the diff, hygiene and tests, commit + push (ask), CI.
  - 2026-11-08: D-031 takeover **step 6 edits written (uncommitted)**: `docs/SERVING.md` now describes the deployed router and replicas, the two answered 'Still open' items are gone, `DECISIONS.md` D-031 has the 2026-11-08 amendment. Next: commit + push (ask), CI, then the 2026-11-10 dmtest sleep check.
  - 2026-11-08: D-031 takeover **done**: step 6 committed and pushed as `5203a86`, CI success. Next: 6.1 now; the 2026-11-10 dmtest sleep check stays on its date and only touches `docs/SERVING.md` and `classify`.
  - 2026-11-08: **6.1 sub-plan (not started, waiting for go-ahead).** README + diagrams. No HF change, no GPU run.
    - **Step 1 (read only):** metric names and where each is measured in the server code, README layout and Reproduce sections, model card wording about the Space.
    - **Step 2 (diagrams, Mermaid like the two already in the README):** (a) one replica: client -> HTTP app -> queue -> engine thread -> slots / KV, labelled with where each metric is measured (PLAN 6.1, D-028); (b) small: visitor -> public router Space -> 2 private replica Spaces (D-031). Both in the README 'How it works' and in `docs/SERVING.md`. Show the user before step 3.
    - **Step 3 (README text, outside the `bench:` markers only):** 'Try it' line (router + 2 private replicas; replicas demonstrate routing, the traffic doesn't need them: D-031 caveat; speed numbers still from the laptop GPU: D-028 caveat; no wake-up timing until the 11-10 check); one Highlights bullet and a short 1-vs-2 replica table (numbers from the load-test JSONs: two replicas 3 req/s, 4 borderline; one 1 req/s; peak gain 1.85x to 2.00x; 0 failed); 2-3 'why' sentences for the user to approve; Repository layout (`router.py`, `Dockerfile.router`, `space/`, `push_space.py`); Reproduce (router command).
    - **Step 4 (checks):** script compares every new number with the JSONs; `git diff` shows nothing inside the `bench:` blocks; tests; hygiene.
    - **Step 5:** user reads the diff; commit + push (ask); CI; user looks at the rendered README on GitHub (Mermaid renders only there); then tick PLAN 6.1.
    - **Left out on purpose:** the 'Status: v1-preview' line (changes at the v1 tag), 6.3 resume bullets, the Space's own README and the model card unless step 1 finds something wrong (HF changes need their own approval).
  - 2026-11-08: 6.1 **step 1 done (read only).** Metrics for diagram (a), from `server.py` / `obs.py`: all timed from `job.arrival` in the HTTP app; `queue_wait_s` = arrival to the start of the step giving the first token; `ttft_s` = arrival to first token (stamped in the engine thread right after the step); `itl_p50/p95_s` = gaps between tokens; `latency_s` = arrival to last token; every ~5 s the engine thread snapshots queue length, busy slots, tok/s, KV used; a full queue answers 429 at the HTTP app. Behind the router, the SLOs visitors see are measured at the router, with the retries and extra hop. Model card needs no change (its link and 'free CPU Space' still hold), so no HF change in 6.1. Also fix in step 3: README layout says router 'tested locally, not deployed yet'; `tests/` says 470 tests, now 640 collected; `scripts/` row lacks `push_space.py`. Next: step 2, the two diagrams (ask first).
  - 2026-11-08: 6.1 **step 2 written (uncommitted)**: user chose simple static Mermaid over interactive or GIF diagrams (an optional short GIF of the live demo is decided after step 3; the user would record it). Two diagrams, rendered with Mermaid 11 in headless Chrome in light and dark before going in: one server with where each metric is measured (README 'How it works', SERVING 'Observability'), and visitor to public router Space to 2 private replicas (README, replaces the ASCII sketch in SERVING 'Router design'). Next: user checks them, then step 3.
  - 2026-11-08: 6.1 **step 3 written (uncommitted)**: README 'Try it', a 'Why' paragraph (user to approve), Deployment highlight, new section 'Results: 1 vs 2 replicas on Hugging Face's CPUs' (numbers from the three JSONs), layout rows, router command in Reproduce. Next: step 4 checks.
  - 2026-11-08: 6.1 **step 4 checks passed (uncommitted)**: bench: blocks unchanged from HEAD; table rows match the three JSONs; README router command tried on CPU (replicas ready, requests alternate r1/r2, / and /status 200, processes stopped); pytest 492 passed, 148 skipped; hygiene clean. Next: user reads the diff and the 'Why' paragraph, decides on the optional GIF, then step 5 commit + push (ask).
  - 2026-11-08: 6.1 step 5: user approved the 'Why' paragraph as written, no GIF for now (can be added later), commit + push approved. Next: CI, user checks the rendered README and diagrams on GitHub, then tick PLAN 6.1.
  - 2026-11-08: 6.1 step 5: pushed `7ecee4f`, CI success. Waiting on: user checks the rendered README and diagrams on GitHub, then tick PLAN 6.1.
  - 2026-11-08: 6.1 step 5 follow-up (approved): GitHub's fixed zoom buttons sit over the bottom-right of each Mermaid diagram on the README's narrow column (serving-engine diagram's last box fully hidden; 'Slots and KV cache' and 'Private HF dataset' cut off; 'out' touched). Fix: invisible spacer box under the bottom-right node in all 4 README diagrams; SERVING.md looks fine at its wider width, leave it. Check in local preview with the button area marked, push, user looks again (and at the Results table), then tick PLAN 6.1.
  - 2026-11-08: 6.1 step 5 follow-up pushed `b1ee9b4`, CI success. On GitHub: Tokens, Slots and KV cache, and Private HF dataset are now clear of the zoom buttons, but the spacer moved 'out' (model block) up under the top-right expand button and its text is clipped. Proposed (awaiting go-ahead): one small invisible box above 'out' in that diagram. Still to see rendered: the 1 vs 2 replicas Results table. PLAN 6.1 not ticked yet.
  - 2026-11-08: 6.1: user saw the 1 vs 2 replicas Results table rendered on GitHub, fine. Approved one more invisible box above 'out' in the model block diagram (local preview with both button areas marked is clear). Next: push, CI, user looks once more at the model block, then tick PLAN 6.1.
  - 2026-11-08: 6.1 **done**: `32784bb` pushed, CI success, user's GitHub screenshot shows 'out' clear; all four README diagrams readable. PLAN 6.1 ticked. Next: 2026-11-10 after 09:00 UTC dmtest sleep check, then 6.3.
  - 2026-11-08: **6.3 sub-plan (proposed, waiting for the user's go-ahead).** 3-4 resume bullets with real numbers. No GPU, no HF changes. Does not wait for the 11-10 dmtest check, as long as no bullet claims waking a sleeping private replica.
    1. **Evidence table:** a read-only script takes every number in the bullets from its source (runs/bench/v1b, runs/bench/load/4.10c, runs/bench/load/space-router JSONs, training log), not from README prose, and prints number, source file, match. Table saved here in PROGRESS (PLAN done-when: every bullet backed by a number in PROGRESS).
    2. **Draft 4 bullets, aimed at inference-engineering roles (PLAN 6.5g):** engine (137 tok/s at batch 1, 71% of the 192 tok/s ceiling, 3.1x HF generate); serving (continuous batching 1.37x static, paged KV 25 vs 8 sequences in 64 MiB, 1.64x; load-tested against vLLM and llama.cpp); training (30M params, 996M tokens, 2 h 27 min, 56% MFU, 8 GB laptop GPU); deployment (own router over two free-CPU Spaces, 3 vs 1 req/s within SLO, peak 1.85x-2.00x, 0 failed).
    3. **Honesty pass:** never imply we beat vLLM (it is 1.35x faster on one stream and holds more load); bounded TTFT comes from refusing requests with 429s, so say so; 3 req/s not 4 for two replicas; laptop GPU numbers and Space CPU numbers kept apart; no Phase 5 claims (D-021); no wake-up claim before the dmtest check.
    4. **User review:** user picks and edits the wording; anything they can't defend out loud comes off (6.6).
    5. **Close:** tick PLAN 6.3, PROGRESS note, commit + push (ask). Open question for the user: final wording in the public PROGRESS or only in git-ignored NOTES.local.md (recommended: the evidence table public, the wording private).
  - 2026-11-08: 6.3: user chose: evidence table public here, bullet wording only in git-ignored NOTES.local.md. **Step 1 done (uncommitted):** all numbers recomputed from the source files, 0 mismatches (table below; 'limit' rows are what the bullets must not overstate). Found while tracing: (a) the HTTP load tests used contiguous slots, so the paged cache's 1.64x comes from the offline 64-request workload, not the server; the server has --kv paged but it was not load-tested; (b) 1.37x holds at 8 slots, at 32 slots continuous is only 1.05x static; (c) paged has no CUDA graphs yet, contiguous + graphs is 2.9x paged on the 30M model; (d) the deployment runs hit the temporary router Space (since deleted) with the per-visitor limit off, so 3 req/s is capacity, not what one visitor gets; (e) the 4.10c and space-router JSONs record uncommitted changes in the tree when measured, the v1b JSONs do not. Serving bullet reworded for (a) and (b). Draft wording is in NOTES.local.md. Next: user review (step 4).

    | Bullet | Number | Claimed | From source | Raw | Source (under runs/bench unless a full path) | Match |
    |---|---|---|---|---|---|---|
    | engine | decode tok/s | 137 | 137 | 136.5 | v1b/tinyllama/kv-graph.json, B=1, prompt 128 | yes |
    | engine | % of ceiling | 71 | 71 | 71 | v1b/tinyllama/kv-graph.json, B=1, prompt 128 | yes |
    | engine | ceiling tok/s | 192 | 192 | 192.3 | v1b/tinyllama/kv-graph.json, B=1, prompt 128 | yes |
    | engine | vs HF generate | 3.1 | 3.1 | 3.076 | v1b/tinyllama/kv-graph.json, B=1, prompt 128 / hf.json | yes |
    | engine | uncommitted changes when measured | False | False | False | v1b/tinyllama/kv-graph.json, B=1, prompt 128 | yes |
    | serving | continuous / static, TinyLlama, 8 slots | 1.37 | 1.37 | 1.371 | v1b/tinyllama/serving.json | yes |
    | serving | paged / contiguous, 30M, 64 MiB | 1.64 | 1.64 | 1.644 | v1b/30m/serving.json | yes |
    | serving | paged concurrent sequences | 25 | 25 | 25 | v1b/30m/serving.json | yes |
    | serving | contiguous concurrent sequences | 8 | 8 | 8 | v1b/30m/serving.json | yes |
    | serving | same KV memory MiB (paged) | 64 | 64 | 64 | v1b/30m/serving.json | yes |
    | serving | same KV memory MiB (contiguous) | 64 | 64 | 64 | v1b/30m/serving.json | yes |
    | limit | vLLM / ours, one stream | 1.35 | 1.35 | 1.353 | load/4.10c/vllm-8seqs-b1.json / ours-8slots-b1.json | yes |
    | limit | continuous / static, TinyLlama, 32 slots | 1.05 | 1.05 | 1.046 | v1b/tinyllama/serving.json | yes |
    | limit | contiguous + graphs / paged (no graphs yet), 30M | 2.9 | 2.9 | 2.947 | v1b/30m/serving.json | yes |
    | limit | ours 8 slots, TTFT p95 at 16 req/s, s | 5.6 | 5.6 | 5.633 | load/4.10c/ours-8slots-sweep.json | yes |
    | limit | ours 8 slots, % refused at 16 req/s | 60 | 60 | 0.6019 | load/4.10c/ours-8slots-sweep.json | yes |
    | limit | vLLM 8 seqs, TTFT p95 at 16 req/s, s | 46 | 46 | 46.32 | load/4.10c/vllm-8seqs-sweep.json | yes |
    | training | parameters, M | 29.9 | 29.9 | 29893120 | checkpoints/main/train.log | yes |
    | training | tokens, M | 996 | 996 | 996.1 | checkpoints/main/train.log | yes |
    | training | mean K tok/s (logged steps after step 0) | 115 | 115 | 115.3 | checkpoints/main/train.log | yes |
    | training | mean MFU % | 56 | 56 | 56.13 | checkpoints/main/train.log | yes |
    | training | wall time | 2 h 27 min | 2 h 27 min | 147.1 | wandb dir name 20261103_144030 to train.log mtime | yes |
    | training | final val loss | 1.2825 | 1.2825 | 1.282 | checkpoints/main/train.log | yes |
    | deployment | run a-2rep: highest rate meeting every SLO | 4 | 4 | 4 | load/space-router/20261108-a-2rep.json | yes |
    | deployment | run b-1rep: highest rate meeting every SLO | 1 | 1 | 1 | load/space-router/20261108-b-1rep.json | yes |
    | deployment | run c-2rep: highest rate meeting every SLO | 3 | 3 | 3 | load/space-router/20261108-c-2rep.json | yes |
    | deployment | run a-2rep: peak served req/s | 4.61 | 4.61 | 4.607 | load/space-router/20261108-a-2rep.json | yes |
    | deployment | run b-1rep: peak served req/s | 2.31 | 2.31 | 2.307 | load/space-router/20261108-b-1rep.json | yes |
    | deployment | run c-2rep: peak served req/s | 4.27 | 4.27 | 4.268 | load/space-router/20261108-c-2rep.json | yes |
    | deployment | peak gain, low | 1.85 | 1.85 | 1.85 | load/space-router: c-2rep / b-1rep | yes |
    | deployment | peak gain, high | 2.00 | 2.00 | 1.997 | load/space-router: a-2rep / b-1rep | yes |
    | deployment | failed requests, all three runs | 0 | 0 | 0 | load/space-router: all rows | yes |
  - 2026-11-08: 6.3 **done**: user approved the four bullets (final wording in git-ignored NOTES.local.md). PLAN 6.3 ticked. Next: 2026-11-10 after 09:00 UTC dmtest sleep check; before that, 6.4 PARKING_LOT review can start. A bullet may mention waking a sleeping replica only after the dmtest check.
  - 2026-11-08: 6.3 committed locally (`2cddabb`, not pushed; user: push together with 6.4, one CI run).
  - 2026-11-08: **6.4 sub-plan (proposed, waiting for the user's go-ahead).** Docs only: PARKING_LOT.md, PLAN.md, PROGRESS.md. Nothing is deleted from PARKING_LOT; closed items are struck through with a pointer, as the file already does.
    1. **Check the unclear items (read-only):** serve the exported HF folder (decided in 6.5b?), 7-day SLO window after a restart (did the router change it?), queue sized from the SLO.
    2. **Close what is already done or promoted:** Docker Space, CI smoke deploy, failure injection, serving diagram, eval gate, SLOs + error budget, README polish items 1 and 3.
    3. **Add the 6.3 finding as a new item:** the HTTP server was never load-tested with `--kv paged`.
    4. **Add a short 'Reviewed 2026-11-08' block at the top:** v1.1 shortlist proposed: (a) CUDA graphs for the paged engine (contiguous + graphs is 2.9x paged; README roadmap names it); (b) prefix caching, option A, ~3-4 h (vLLM and llama.cpp both have it); (c) CUDA graphs for prefill, ~2-3 h (ITL p99 30-45 ms vs p50 9 ms); (d) `bench_all` throttle warning fix, ~30 min. Tied to other steps: blog link and animated explainer (6.5d). Stay parked: all research and Phase 5 / 7 items (D-021). Before v1: no code. Optional for the user: GitHub social preview image (Settings, about 2 min).
    5. **Close:** tick PLAN 6.4, PROGRESS note, hygiene, commit locally, then push 6.3 + 6.4 together and check CI (ask).
  - 2026-11-08: 6.4 **done**: user approved the shortlist, social preview moved to the 6.5d group. Checks: serving the export is done for our story model, 7-day SLO window still open, queue auto-sizing still parked. PARKING_LOT.md: review block added, finished items struck through with pointers, nothing deleted, one new item (`--kv paged` load test). PLAN 6.4 ticked. Next: push 6.3 + 6.4 (ask), then 2026-11-10 after 09:00 UTC dmtest sleep check; 6.5c-g and 6.6 remain.
  - **v1 schedule (agreed 2026-11-09):** Mon 11-09: 6.5c, blog draft (6.5d), user starts 6.6. Tue 11-10 after 09:00 UTC: dmtest sleep check (README line 7 says the Spaces wake for the next visitor) and one test prompt through the live demo. Tue daytime: finish 6.6, GitHub Pages on (assumed; user to confirm), social preview. **Tue 7:30-9 pm IST: blog live, LinkedIn post 1, README status line, tag v1 (ask first).** Thu 11-12 r/LocalLLaMA. Tue 11-17 LinkedIn 2 + r/MachineLearning. Tue 11-24 LinkedIn 3. After v1: SERVING/classify update and dmtest delete (ask first); 6.5g outreach, ongoing. If 6.6 is not ready on Tuesday, post Wed or Thu evening instead; 6.6 is not shortened.
  - **6.5c sub-plan (2026-11-09; waiting for the user's go-ahead):** Read-only check found that both W&B runs (`cel67ssj`, `ugd124lz`, project `cooldip00-personal/tinyllm`) store the user's email, the laptop host name and the home path in `wandb-metadata.json` and `config.yaml`, which a public project would show in each run's Files tab. Steps: 1. Confirm the local copies of those files exist under `wandb/` (git-ignored). 2. Delete those two files from both runs on W&B (user approves). 3. Re-read through the API that email and host are gone. 4. User sets the project to public. 5. User opens the run in a private window and checks Overview and Files. 6. Add the run link to the README training section; local commit, push when asked. Fallback if anything personal still shows: keep the project private and put a loss-curve PNG drawn from `checkpoints/main/train.log` in the README.
  - 2026-11-09: 6.5c steps 1-3 **done** (user approved): the two files copied to a folder outside the repo and checked against the laptop and the server (SHA-256), then `wandb-metadata.json` and `config.yaml` deleted from both runs. Read back: exactly those 2 removed per run (29→27, 84→82), config, summary and last history step (1999, 7599) unchanged; no email or host name in the API record or in any of the 59 remaining non-media files. **Still there:** the WSL home path in the `local_path` field of 50 of the 53 artifact manifests. Left in place: it is only the WSL user name, already in public DECISIONS.md and docs/SETUP.md, and deleting manifests could break the sample tables. Next: step 4 user sets the project public; step 5 private-window check (Overview may still show the host name); step 6 README link.
  - 2026-11-09: 6.5c steps 4-5 **done**: user turned off the account privacy block and set the project to Public. Logged-out private-window check of run `ugd124lz`: Overview shows only the username, run path and start time; Files has no metadata or config file; no email, host name or full name on those pages. Raw config of both runs read through the API: clean. W&B trial ends about 2026-12-02: downgrade to Free, check no card is charged, then re-open the run link logged out. Next: step 6 README link (waiting for the user).
  - 2026-11-09: 6.5c step 6 **done, so 6.5c is done**: run link added under the loss chart in the README training section, PLAN ticked. Charts tab of the run also checked logged out by the user (loads, nothing personal). **Email decision:** old commits are NOT rewritten (it would change all 95 hashes and need a force-push, and clones and old commits by hash would still show it). From now on: GitHub "Keep my email addresses private" and "Block command line pushes that expose my email" are on, this repo's git `user.email` is the GitHub noreply address, and docs/SETUP.md shows a placeholder. The block checks the most recent commit, so every commit made here must use the noreply address.
  - 2026-11-09: quick pre-launch items: top of this file rewritten, old block kept at the end. README line 20 and the resume bullet in NOTES.local.md now say OpenAI-compatible `/v1/completions` (only that route and `/v1/models` exist; no `/v1/chat/completions`, no `stop` or per-request `seed`, logprobs always null). `runs/bench/v1b` copied to `D:\ResumeProject\backups\bench-v1b-2026-11-09` (26 files, 140 MB, sha256 of every file matched). Public HF scan, anonymous, all commits: model repo (3 commits, 7 text blobs, weights skipped) and Space `tiny-llm` (5 commits, 48 text blobs): no tokens, assigned secrets, private keys, email or home paths, and none in commit messages. Anonymous listing shows only those two repos plus `class-grader` as public (not scanned; not part of this project).
  - **6.5d blog sub-plan (2026-11-09; approved by the user):** Site: a separate public repo `cooldip00.github.io` (user creates it) with plain Jekyll, not Pages from this repo's `docs/`, which would publish SETUP.md and the other working docs. Draft lives in `blog-draft.local.md` (git-ignored by `*.local.md`), so nothing is public until launch. About 2,500 words, one post, first person. Outline: (1) hook: 137 tok/s, 71% of the 192 ceiling, 3.1x HF, demo link; (2) why build it: libraries hide where time goes; (3) training, short: 29.9M params, 996M tokens, 2 h 27 min, 56% MFU, val 1.2825, W&B link; (4) plain-language explainer: what a KV cache is, why decode is memory-bandwidth bound, how the ceiling is computed; (5) reaching 71%: CUDA graphs 54 to 137, decode profile chart; (6) many users: continuous batching 1.37x at 8 slots (1.05x at 32), paged KV 25 vs 8 sequences, 1.64x offline only, contiguous + graphs 2.9x paged; (7) vs vLLM, honestly: vLLM 1.35x on one stream; at 16 req/s ours TTFT p95 5.6 s with 60% refused, vLLM 46 s; a trade, not a win; (8) two free CPU Spaces behind our own router: 1.85-2.00x, 0 failed of 2,940, measured on a temporary router with the per-visitor limit off; (9) what didn't work and what I learned, linking key D-xxx entries; (10) what's next, links. Steps: 1 I write the draft with charts from `docs/img`; 2 I check every number against README/RESULTS in a table, as in 6.3; 3 the user edits it into their own words and cuts anything they can't defend (with 6.6); 4 Tue daytime: user creates the site repo, I prepare the site files, preview, secrets and personal-data scan of the final text; 5 Tue 7:30-9 pm IST: post goes live (ask first), README and resume links, tick PLAN 6.5d. Rules: `/v1/completions` wording; no claim about waking a sleeping private replica until the dmtest check; keep the employer generic; no email or home paths.
  - 2026-11-09: 6.5d steps 1-2 **done**: `blog-draft.local.md` written (about 2,500 words, git-ignored, not published), every number checked, results in `blog-numbers.local.md` (git-ignored). Facts settled while checking: the 192 ceiling applies to bf16 weights only (INT8 Triton reaches 200 tok/s at B=1); the 2,940 requests are 980 per replica run, of which 2,346 served, 594 refused with the router's 429, 0 failed (`runs/bench/load/space-router`); the headline 137 tok/s is TinyLlama, not the user's own 30M model, and the post says so in its opening. Next is step 3: the user rewrites it in their own words and fills the two [YOU: ...] places (remove both markers before publishing), cutting anything they can't defend.
  - 2026-11-10: 6.5d review fixes **done** (user approved after the Codex and Gemini reviews were checked against the repo): new title "I built an LLM inference engine from scratch and measured it against my laptop GPU's limit" (the engine is the main story); opening with one number (71%), 192/137/3.1x moved into "Getting to 71%"; Triton norm introduced (1.06x, 1,164 to 827 GPU ops; both the 54 and 137 rows include it, per D-032); correctness split into HF check (eager path) and graph vs eager (bit-identical to the static eager step, bf16-close to plain eager, 98.97%); 1.05x reason (64 requests = 2 waves of 32); 30M model at 15% of its ceiling; Docker thread story (~5x, 105 vs 21 ms); network TTFT p50 1.18-1.24 s at 1 req/s; caveat moved under the vLLM table; 'slots' throughout. Line 14 (sleep) waits for today's dmtest check. Rejected from the reviews: the 137-vs-162 split, a torch.compile mechanism, '54 to 137 together', INT8 'breakthrough', the 503/35 s sleep wording (that was the paused test), and the encoding 'fix' (file is valid UTF-8). Backup of the draft before this edit: the session scratchpad, blog-draft.before-review-fixes.md. Number check rerun: 48 claims, 0 failed (10 new); 2,831 words. Line 14 still waits for the dmtest check.


### 2026-11-07 — 6.5b step 4: log shipping, privacy, usage report
- **Shipping** (`src/tinyllm/logship.py`): `--log-dir DIR [--log-dataset REPO_ID] [--log-replica]` writes a filtered JSONL copy of the log, one file per server start per UTC hour (`<date>/<start>-<replica>-<HH>.jsonl`), and uploads it with a `CommitScheduler` subclass every 10 min plus once after the drain (`create_app(..., shipper)`). Shipped: `request` / `slow request`, snapshots thinned to 30 s, lifecycle lines (`server ready` now carries `slo_ttft_s`), errors without tracebacks; never debug. No token or no access: one warning, logs stay local. Retention on the uploader's thread, daily: delete dataset files > 30 days old in one commit; squash history when its oldest commit is > 30 days old. Dockerfile: `LOG_DATASET=""` (off); when set, CMD adds `--log-dir /tmp/tinyllm-logs --log-dataset`.
- **Privacy:** audit test sends a marker through 200, stream, 400 (validation ×2, too long), 429, 504 and a disconnect; the shipped file has every status and no marker, no IP, no completion text. Demo footer "What is logged" `<details>`; SERVING.md "Log shipping" + privacy rules (salt secret, pseudonymous hash, 30 days, hard kill loses ≤ 10 min).
- **Report:** `scripts/analyze_logs.py --dir | --dataset [--since | --days] [--json]` (markdown; JSON to a new `runs/logs/report_<UTC>.json`, never overwritten). Same SLO maths as `obs.py`; corrupt / unknown lines counted.
- **Tests:** `tests/test_logs.py` (8, incl. the real `CommitScheduler` subclass against a fake `HfApi`), `tests/test_analyze_logs.py` (4). CPU suite: 311 passed, 148 skipped (459 tests). Docker: smoke check passes with logging off; with `LOG_DATASET` set and no token: "log upload off: logs stay local", hourly file written, marker count 0, `stopped` reached, report renders.
- **Real proof on HF (item 5):** created the private dataset `CoolDip00/tiny-llm-space-logs`. Ran the container with `LOG_DATASET` set and the WSL HF login mounted read-only (`HF_TOKEN_PATH`; the token is never in a command or env value). Traffic: 8 concurrent requests with a marker, one 400, one stream; then `docker stop`. Container log: `log upload on` → `log retention` → `stopped` → `logs uploaded` (~1 s after stop). Dataset: still private, 2 commits (initial + one scheduled), one file `logs/2026-11-07/20261107T143701Z-local-14.jsonl`, 6.7 KB, 15 lines; marker 0, container IP 0. `analyze_logs.py --dataset … --days 1 --json`: 10 requests (9 × 200, 1 × 400), 1 clean stop, SLOs met, JSON in `runs/logs/`. Measured line sizes: request 574 B, snapshot 278 B, so the size estimate holds. HF shows `usedStorage` 0 B right after upload (not yet computed).
- **Next:** see Current → Next action.

### 2026-11-07 — 6.5b step 3: demo page, status page, SLOs and error budget
- **Pages:** `GET /` demo (story continuation streamed over SSE; length / temperature sliders, Stop, TTFT and tokens/s, friendly 429 / 503) and `GET /status` (polls `/stats/history` every 5 s; hand-drawn SVG charts: requests / min by outcome, TTFT p50 / p95 + latency p95 against the SLO line, busy slots + queue; crosshair tooltips, keyboard, table view, light and dark). Static HTML + JS in `src/tinyllm/web/`, no CDN, relative URLs (for the D-031 router).
- **Server:** `GET /stats/history` (computed at most once a second): the last 60 min per minute, the last hour of engine snapshots, each SLO over a rolling 7-day window with compliance, budget left and the 1 h burn rate (fast burn > 14.4× with ≥ 10 requests). Hourly counters in memory, so the window starts at the last restart (parked: rebuild from the step-4 logs). `--slo-ttft-s` (default 1 s; the Docker image sets 2 s). 400 / 499 count for no SLO; page and stats requests aren't counted.
- **Leading space fixed** (PARKING_LOT): the completion is decoded behind the prompt's last token, so TinyLlama returns `' Max was happy.'` like vLLM / llama.cpp; the 30M tokenizer already did.
- **Tests:** `tests/test_status.py` (10: SLO and budget maths, burn rate, 7-day pruning, history shape, both pages, polling not counted), `tests/test_detokenize.py` (5). `smoke_server.py` also checks `/`, `/status`, `/stats/history`. CPU suite: 299 passed, 148 skipped (447 tests).
- **Docs:** SERVING.md "Demo and status pages" section; README line; PLAN tick.
- **Next:** see Current → Next action.

### 2026-11-07 — 6.5b: 30M model on the Hugging Face Hub
- **Upload:** `upload_model.py --yes` → public https://huggingface.co/CoolDip00/tiny-llm-tinystories-30m, commit `b11100b` (log `runs/export/upload_20261107T134020Z.json`). Downloaded back at that commit into a fresh cache: all 5 files' sha256 and the README (= `hf/MODEL_CARD.md`) match.
- **Image:** Dockerfile `REVISION` pinned to `b11100b…`. Plain `docker run` (no mount) downloads from the Hub, loads in 30.4 s, smoke check passed ("…a little girl named Lily…", TTFT 58–78 ms on CPU for a 4-token prompt), `/stats` version `tiny-llm-tinystories-30m@4efa5367+485e951`, `docker stop` → drain → exit 0.
- **Next:** see Current → Next action.

### 2026-11-07 — 6.5b steps 1–2: model card, upload script, CPU Docker image (local, not published)
- **Server:** `--hf-model DIR_OR_REPO` (+ `--revision`) loads our 30M model from a 4.8b export folder or a Hub repo (`snapshot_download` of the 5 export files only; `--offline` = HF cache only). Refuses a non-story export (Llama-style BOS/EOS → "use `--model tinyllama`") and a tokenizer / config EOS mismatch. 6 tests (`tests/test_server_hf.py`, tiny random model from `scripts/make_tiny_hf_model.py`).
- **Image:** `Dockerfile` (python 3.12-slim, torch 2.14.1 CPU wheel + the server's pinned packages from uv.lock, asserts no CUDA; uid 1000; no weights, model from the Hub at start or a mounted folder via `MODEL=/model`), allow-list `.dockerignore` (package + tokenizer only). `tinyllm:dev` 1.52 GB. `scripts/smoke_server.py` (stdlib): `/ready`, `/health`, one completion, `/stats` counts it and reports `norm_backend: "torch"`. Passed in the container with the tiny model and with the real 30M export mounted ("…a little girl named Lily…"); `docker stop` exits 0 (graceful drain).
- **CI:** new `docker` job: build, tiny model, serve, smoke check, graceful stop, server log on failure.
- **HF:** `hf/MODEL_CARD.md` (usage verified locally with Transformers; numbers from the 3.7 run and the 4.8b export check). `scripts/upload_model.py`: dry run by default; checks the exact file set, export check PASS and sizes, card names the repo, login owner; one commit; saves `runs/export/upload_<time>.json`. Dry run OK (weights 119,580,576 B). Re-export at clean `6756629` (`runs/export/tinystories-30m-6.5b`, PASS, 121.9 s): every file identical to the 4.8b export.
- **Docs:** README Reproduce gains the docker build / run / curl lines and a CPU-only note; layout table row. Tests 432 passed.
- **Next:** see Current → Next action.

### 2026-11-07 — 4.10c done: every GPU table rerun with the Triton norm (D-032)
- **Benchmarks:** `runs/bench` backed up (outside the repo, verified); `bench_all --run --dir runs/bench/v1b`, all 17, commit `3928889`, performance-fans-max-room-fan-overhead, cool to 60 °C; 09:25–11:09 UTC (1 h 44 min), exit 0 (log `runs/bench/4.10c/bench_all_20261107T092546Z.log`). Every model JSON records `norm_backend: triton` (HF: `hf`).
  - TinyLlama kv-graph B=1 127 → 137 tok/s (66% → 71% of the ceiling, 3.1× HF); B=8 814 → 886; B=32 1,984 → 2,138. KV eager 49 → 54, naive 31 → 35. INT8 Triton 1.45× → 1.47× bf16 at B=1. Serving static / continuous / +graphs 189 / 261 / 581 → 218 / 299 / 637 tok/s.
  - Logits vs HF bf16: max |Δ| 0.34 prefill / 0.28 decode, top-1 same at 100% / 99.24% (v1: bit-identical). With `--norm-backend torch` (new flag in `tinyllama_vs_hf.py`, 51 s, `runs/bench/v1b/tinyllama/logits_vs_hf_torchnorm.json`): bit-identical again, so the gap is the norm's summation order. RESULTS shows both; the README sentence says so only when that file is present (`results.EXTRAS`).
  - Heat matched v1. Two `bench_all` warnings, neither rerun: `30m-serving` clock median 1,155 MHz (light configs clock down at 52–57 °C; v1 1,267 MHz) and `tl-int8-graph` hardware slowdown flags (also in v1; rounds within ~2%). The check misfires on mixed loads → PARKING_LOT.
- **Load sweep** (same configs, rates and client as 4.8a; one engine at a time, cooled to 60 °C before each; 11:10–12:22 UTC; `runs/bench/load/4.10c/`, 12 JSON; prefix caching off ×3, llama.cpp 8 × 2048 slots, both checked in the logs):
  - One stream: ours 111.7 → 119.6 tok/s (62.2%); vLLM 161.8 (84.1%); llama.cpp bf16 153.4 (79.8%), Q8_0 236.2. vLLM's lead 1.44× → 1.35× (D-032 expected ~1.35×), llama.cpp's 1.35× → 1.28×.
  - Rated loads unchanged (ours 4 / 8 at 8 / 32 slots, vLLM default 32, llama.cpp 4) except vLLM 8 seqs 6 → 8 req/s: TTFT p95 0.95 s at 8 against the 1 s SLO (4.8a: 1.07 s), saturation 8.5 → 8.7 req/s. Documented as the SLO edge, not a gain.
  - Heat: max 83–86 °C, SM medians 1,575–1,875 MHz; ours-32slots also logged hw slowdown.
- **Code/docs:** `extract_load_sweep.py` took `"source"` from a hardcoded 4.8a path; now from `--dir` (default 4.10c) and `docs/data/load_sweep.json` regenerated. `--dir` defaults moved to `v1b` (`bench_all.py`, `results.py`, `plots.py`). Logit wording in the README / RESULTS renderer; new test. README vs vLLM / llama.cpp section, SERVING.md 4.10c table (4.8a kept), INFERENCE.md 4.10c section + notes on 4.9c and 4.9d. The "~1.5–2 h" run time was right; left as is. Tests 426 passed.
- **Next:** push (ask), then 6.5b.

### 2026-11-07 — 4.10a done; Triton norm is the default (D-032); what v1 ships (D-033)
- **Kernels (`891814c`):** `src/tinyllm/kernels/rmsnorm.py`: `rmsnorm` and `add_rmsnorm` (residual add + norm, returns both), one program per row, fp32 statistics, rounding as in `model.RMSNorm`; `h` bitwise equal to `x + a`. `RMSNorm(backend=...)`, `set_norm_backend`; `Block.forward` uses `add_forward` for the post-attention norm. Triton only on CUDA with no gradient; otherwise PyTorch.
  - Tests: the fp16 cases first failed the default tolerance (20 of 8.4M elements off by 1–2 ulps: only the fp32 sum's order differs). Not a kernel bug; the tests now allow 3 ulps and require > 99% bitwise equal. GPU suite 424 passed (log `runs/bench/4.10a/gpu_tests_20261107T080417Z.log`).
- **Session** (`bench_all --only tl-rmsnorm`, performance-fans-max-room-fan-overhead, 1,620 MHz median, 85 °C, software throttling only; log `runs/bench/4.10a/rmsnorm_20261107T081404Z.log`; JSON `runs/bench/v1/tinyllama/rmsnorm_graph.json`):
  - kv-graph decode, PyTorch → Triton norm: B=1 128 → 137 tok/s (1.07×), B=8 844 → 897 (1.06×), B=32 2,073 → 2,162 (1.04×).
  - B=1 graph step: 1,164 → 827 GPU ops, GPU busy 7.74 → 7.25 ms (both 1,875 MHz).
  - add+norm alone: 1 row 18.2 → 2.1 µs; 4,096 rows 872 → 164 µs at 97% of 423 GB/s.
  - Greedy rows identical to the PyTorch norm 1/8; teacher-forced argmax agreement 99.28%.
- **Default switched (D-032), after discussion:** vLLM and llama.cpp both fuse their norms, so a comparison table with our norm unfused undersells the engine. Training and CPU (the Space) are unchanged by construction. Condition: **every GPU table is rerun in one session** before 6.5b (PLAN 4.10c, into `runs/bench/v1b`, plus the load sweep), so no table mixes the two norms; the ablation table stays. Expected (not measured): one stream ~112 → ~120 tok/s, vLLM's lead 1.44× → ~1.35×.
  - Code: `DEFAULT_NORM_BACKEND = "triton"`; `norm_backend(model=None)` → `"triton"` / `"torch"` / `"mixed"` (what actually runs); recorded in every bench script's provenance and in `/stats` per variant. New test `test_default_backend_is_triton`. CPU suite 277 passed, 148 skipped. The GPU suite has not been run on the new default yet.
  - Side effects: first-use Triton compile (~1 s per width/dtype, cached; absorbed by warm-ups and graph capture); GPU val loss under `no_grad` may differ in the 4th–5th decimal. The trained 30M checkpoint is unaffected (same weights and `state_dict`).
- **What v1 ships (D-033):** live demo = the 30M model on a free CPU Space (no GQA / Triton / graphs there); HF model repo = 30M weights + card; GitHub = source + **CPU-only Docker image**, with 30M and TinyLlama and every GPU path. No 30M retrain with GQA (tiny KV at 30M; GQA already shown on TinyLlama in 4.9a) → PARKING_LOT, Phase 5. CUDA image → PARKING_LOT, v1.1. No paid GPU Space.
- **Docs:** INFERENCE.md 4.10a section + Triton table row; PLAN 4.10a ticked, 4.10c added, 6.5b note; DECISIONS D-032, D-033; PARKING_LOT 2 items; README test count 301 → 425, kernels row.
- **Next:** GPU test suite (ask), push (ask), then 4.10c.

### 2026-11-07 — 4.10b done: final session, docs (PLAN 4.10b ticked)
- **Spill fix (`55e8595`):** `bench_quant` times each variant with only its own weights on the GPU (int8 + int8-triton share tensors and move together; `_DECODERS` cleared; `timed_alone` in the JSON); 3 tests.
- **Wiring (`2fdd1dc`):** `tl-int8-graph` + `tl-int8-gemv` in `results.BENCHES`; RESULTS INT8 section in three parts; README INT8 line now reports the Triton kernel.
- **Final session** (`bench_all --only tl-int8-gemv,tl-int8,tl-int8-graph`, commit `2fdd1dc`, performance-fans-max-room-fan-overhead, each job started ≤ 50 °C; log `runs/bench/4.10b/final.log`; backup `runs-bench-2026-11-07-final.tar.gz`; old `int8.json` kept as `.throttled-2026-11-07`):
  - gemv: decode-step matmuls M=1 bf16 5.67 / torch 16.66 / triton 3.04 ms (1.87×); M=32 1.80×; M=64 0.99×. Software throttling only, 81 °C.
  - tl-int8 (eager): int8 0.74–0.75× bf16 at B=1/8/32; B=32 int8 1,085 tok/s (was 533, the spill). Software throttling only, 85 °C.
  - tl-int8-graph: bf16 / int8 / int8-triton B=1 128 / 50 / **186 (1.45×)**; B=8 839 / 355 / **1,063 (1.27×)**; B=32 2,053 / 1,064 / **2,175 (1.06×)**. ⚠ hw slowdown flags seen (median 1,545 MHz, 85 °C); kept because it matches the first session and the B=32-alone run within 3%.
- **results.py:** the RESULTS Run line counts only the main run's commits; `--only` reruns are grouped by date, commit and NitroSense mode (fixes "⚠ 2 commits" and undated rerun times). Test updated.
- **Docs:** INFERENCE.md 4.10b section + the INT8 row of the Triton table; PLAN 4.10b ticked.
- **Open:** int8 TTFT 10–130 ms over bf16 (prefill on the PyTorch path); at B=32 the step saves 0.9 ms of the 3.0 ms the matmul bench predicts (not explained).
- **Next:** push, then 4.10a.

### 2026-11-07 — 4.10b Triton INT8 kernel: built, first GPU session (stop rule passed; not final)
- **Kernel (`3bf1212`):** `src/tinyllm/kernels/int8_gemv.py`, one Triton kernel reading int8 weights directly (`tl.dot`, M padded to 16, autotuned per N, K, M bucket), for M ≤ 64 on CUDA; `Int8Linear(backend="torch"|"triton")`, falls back to torch on CPU / fp32 / prefill / no Triton; `set_backend`. Tests: error vs fp64 ≤ the torch path's on all TinyLlama shapes, M 1–64; CUDA-graph replay; fallbacks. 342 tests pass.
- **Bench scripts (`317bee7`):** `bench_quant --engine kv|kv-graph --triton` (variant int8-triton shares int8's tensors in a separate module tree, so the per-model GraphDecoder cache can't mix them; default names carry the mode). `scripts/bench_int8_gemv.py`: kernel alone vs torch path vs bf16, CUDA-graph replays over weight copies > L2.
- **Safety:** `runs/bench` backed up to `D:\ResumeProject\backups\runs-bench-2026-11-07.tar.gz` (re-extracted, `diff -r` identical); old `v1/tinyllama/int8.json` kept as `int8.json.throttled-2026-11-06` (cmp identical). Session log `runs/bench/4.10b/run.log`, power mode in `session.txt`.
- **GPU session (performance mode, fans Auto, each job started ≤ 50 °C, commit `317bee7`):**
  - Microbench (`runs/bench/4.10b/int8_gemv.json`, software throttling only, 83 °C): decode-step matmuls M=1 bf16 5.68 / torch-int8 16.68 / triton 3.10 ms (1.83× bf16); M=32 6.76 / 17.90 / 3.76 ms (1.80×); M=64 6.17 / 17.76 / 6.50 ms (0.95×: tile limit). Large layers at 88–97% of 423 GB/s.
  - kv-graph comparison (`runs/bench/4.10b/int8_graph_triton.json`, ⚠ hw thermal slowdown, 86 °C, min 525 MHz; variants interleaved per round so ratios are fair, absolutes not): decode tok/s bf16 / int8 / int8-triton: B=1 129 / 49 / **186 (1.45×)**, rounds 186/186/184; B=8 827 / 349 / **1080 (1.31×)**; B=32 2021 / 619 / 1621 (0.80×, rounds 1807/1560/1621). Val loss int8 = int8-triton = bf16 + 0.0010.
  - **Stop rule passed** (int8 beats bf16 at batch 1 with graphs on).
  - `tl-int8` rerun via bench_all (`v1/tinyllama/int8.json`): ⚠ hw thermal slowdown again (87 °C). Its README/RESULTS render (INT8 0.84× / 0.35×) was **not** committed; docs restored.
- **Found:** (a) every ~20-min TinyLlama run hits hw thermal slowdown with fans on Auto, even starting at 50 °C → fans on Max for benchmark sessions. (b) At B=32 the int8 variants reserve ~8.7 GB on the 8 GB card (prefill TTFT 5.4–6.8 s vs 0.34 s bf16): WSL spills to host memory. Causes: the bf16 model stays resident while int8 is timed (benchmark artifact; the v1 baseline had it too), prefill computes logits for every position (int8: bf16 + fp32 copies), torch path's per-call bf16 weight copy. B=32 loss may be spill and/or heat; not claimed either way.
- **Next:** see Current → Next action. Rerun in the final session: jobs 2 and 3 (must), job 1 (2 min, same manifest).

### 2026-11-07 — Plan: both Triton kernels and HF replicas in v1 (D-016 amendments, D-031)
- **Triton (D-016 amendments, 11-06 and 11-07):** 4.10b INT8 GEMV first, then 4.10a fused RMSNorm (~1–1.5 h); no longer gated on 4.7; before 6.3; Triton capped at one day. Judged with CUDA graphs on, after a cool `tl-int8` rerun (the v1 run was throttled). Stop rule: if int8 doesn't beat bf16 at batch 1 within the half day, stop and record. No retraining: both kernels run on the existing weights.
- **Replicas (D-031):** after 6.5b's single Docker Space, two replica Spaces plus a router Space running our `Router` (shortest queue from `/stats`, retry on 429 or a dead replica, wake a sleeping one). Spaces only; a Vercel frontend was rejected (its 4 h / month of function CPU, a second codebase). About +5–7 h, before 6.3.
- **6.1 diagrams** move after the replicas: the single-server diagram becomes one replica, plus a small router → replicas diagram.
- **Housekeeping:** CI green for `5a0ce75` and `e8a20c4`; repo pinned (6.5a done); `sweep.log` backed up outside WSL (hash checked).
- **Next:** 4.10b, starting with the cool `tl-int8` baseline.

### 2026-11-06 — README completeness: vs vLLM / llama.cpp, the server, the export (D-030 amendment)
- **Why:** the README audit after 6.2 found results and features that never reached it: the 4.8a comparison with vLLM and llama.cpp, the 4.13 server, the 4.8b export, and modules / docs missing from the layout. It's the third time something was left out, so there is now a test and a PLAN rule.
- **README:**
  - New section "Results: vs vLLM and llama.cpp": the 5-row table from SERVING.md, 4 bullets (one stream, seats set capacity, past saturation, caveats), and the load plot.
  - Highlights: an HTTP server bullet (429s past capacity keep TTFT bounded; links the section). How it works: one server bullet linking SERVING.md. Training table: an Export row (HF logits within 1.4e-5, identical tokenizer ids).
  - Layout: `server.py`, `obs.py`, `logger.py`, `logit_compare.py`, `provenance.py`, `results.py`, `plots.py`; scripts row names `load_test.py` and `export_hf.py`; the docs row links all 8 docs. Reproduce: start the server, run the load test. Status line and Roadmap no longer list the comparison as to do.
- **Load plot** (`docs/img/load_sweep.svg`, D-030 amendment): `scripts/extract_load_sweep.py` → `docs/data/load_sweep.json` (2 KB, 8 rates × 3 engines, no local paths); two panels, completed req/s and TTFT p95 (log) vs offered rate. Checked as PNG on white and `#0d1117`. Our TTFT line is flat past saturation only because the full queue rejects with 429s, and TTFT counts served requests only, so the panel is titled "served requests" and the line end says it's capped; otherwise it reads as faster.
- **Prevention:** `tests/test_readme.py` (every `src/tinyllm` module in the layout, every `docs/*.md` linked, every README image exists); PLAN rule: a step's done-when includes the README.
- **Checks:** `bench_all.py` twice: "1 of 6 changed", then "0 of 6 changed", README / RESULTS / SVGs byte-identical. Tests +4 (load extractor, 3 README checks). Full suite **301 passed**.
- **Next:** push both commits (one CI run), check the 6 SVGs on GitHub in light and dark, then 6.1 / 6.3.

### 2026-11-06 — 6.2 plots for the README and RESULTS.md (D-030, 6.2 done)
- **What:** `src/tinyllm/plots.py` draws five SVGs with matplotlib 3.11.2 (new dev dependency); `bench_all.py` writes them to `docs/img/` with the tables, from the same `runs/bench/v1/` JSON:
  - README: `decode_b1` (B=1 by engine against the 192 tok/s ceiling, top of Highlights), `batch_scaling` (ours vs HF at B = 1 / 8 / 32, log scales, ceiling × batch), `train_loss` (training section).
  - RESULTS.md: `decode_b1`, `batch_scaling`, `serving` (TinyLlama scheduling; 30M contiguous vs paged in the same 64 MiB), `decode_profile` (GPU idle 76% eager vs 3% with graphs).
- **Design:** one transparent SVG per plot for GitHub light and dark (checked as PNG previews on white and `#0d1117`); blue = ours, orange = HF, gray = context; values on every mark; alt text with the numbers, generated from the JSON.
- **Determinism:** fixed SVG hash salt, no date, text as paths. `bench_all.py` twice: the second pass prints "0 of 5 changed" and the README is unchanged.
- **Training curve:** `scripts/extract_train_curve.py` → `docs/data/train_curve.json` (761 train / 39 val points, 11 KB, with the log's SHA-256), since `checkpoints/` is git-ignored. `.gitignore`'s `data/` also matched `docs/data/`; re-included with `!docs/data/` (`data/train.bin` still ignored).
- **Tests:** +4 (`tests/test_plots.py`: deterministic render, missing row names its benchmark, `bench_all.render` writes the plots then changes nothing, log parsing keeps a resumed step's last value). Full suite **297 passed** (README count updated).
- **Moved / parked:** the ablation plots to 5.5 (PLAN). An animated paged-KV explainer to PARKING_LOT (blog / Space; the README can't run JS).
- **Next:** push, check the SVGs on GitHub in both themes, then 6.1 / 6.3.

### 2026-11-06 — 6.8 GPU run: every results table from one run (6.8 done)
- **Run:** `bench_all.py --run --power-mode performance`, 11:29–12:57 UTC, commit `a765817`, clean tree, NitroSense **performance**, cool-down to ≤ 60 °C before each benchmark. **14 / 14 OK.** RTX 3070 Ti Laptop, driver 610.91, torch 2.14.1+cu130. JSON, `run.log` and `manifest.json` in git-ignored `runs/bench/v1/`.
- **Reruns** (`--only`, cool-down to ≤ 50 °C; marked in the manifest and listed in the RESULTS.md header): `30m-serving` (start 43 °C) and `tl-int8` (start 50 °C). Both reproduced the first run, so their ⚠ is the laptop, not a bad session:
  - `tl-int8`: bf16 46 / 374 / 1,482 tok/s vs int8 35 / 279 / 533 at B = 1 / 8 / 32 → **0.75× / 0.75× / 0.36×** (first run 0.75 / 0.74 / 0.35). Still `hw_thermal_slowdown` at 86 °C. The README's old 0.54× at B=32 came from a different session.
  - `30m-serving`: paged/16 at 32 slots **998 tok/s = 1.62×** continuous at 8 slots (615) in the same 64 MiB, 25 peak active, 42 preemptions, KV in use 52% → 97%. 84 °C with `sw_thermal_slowdown`.
- **Headline numbers** (now in the README and RESULTS.md):
  - TinyLlama B=1 P=128 decode: HF 44, naive 31, KV 48, **KV + CUDA graphs 127 tok/s = 66% of the 192 tok/s ceiling, 2.9× HF** (11-04: 128, 66.8%, 2.8×).
  - kv-graph grid (P = 32 / 128 / 512): B=1 130 / 127 / 112; B=8 875 / 814 / 629; B=32 2,332 / 1,984 / 1,214 tok/s.
  - Serving (TinyLlama, 64 requests): static 189, continuous 261 (**1.38×**), cont+graph 581 tok/s at 8 slots.
  - Profile: TinyLlama eager GPU idle 76%, graphs 3.4%, **3.60×**.
  - Logits vs HF: prefill (151 positions) and decode (264) `torch.equal`; greedy 5 / 5.
- **Heat:** 84–87 °C peaks in the long TinyLlama benchmarks; the cool-down gate controls only the start temperature. The OOM warnings in `tl-kv` / `tl-kv-graph` come from the untimed kv-vs-naive correctness check, not from timed rows.
- **Throttle rule narrowed** (D-029 amendment): the first rule flagged 5 / 14, three of them only for a low clock with no thermal reason (30M KV decode idles at 1,095 MHz, 65 °C). Now: a hardware slowdown, or a low busy median together with `sw_thermal_slowdown`. 2 flagged.
- **Docs:** README "221 tests" → 293, `bench_all.py` in Reproduce, roadmap item removed. RESULTS.md header lists the reruns. PLAN 6.8 ticked.
- **Not done:** `stash@{0}` (first render, before the reruns) kept until you say drop it. Not pushed.
- **Next:** push (one CI run), then 6.2 plots (6.5a still lacks them), then 6.1 / 6.3.

### 2026-11-06 — 6.8 build: one-command tables (D-029); GPU run not done yet
- **Overwrite guard:** `provenance.claim_output()` makes the five bench scripts (`bench_generate`, `bench_serving`, `bench_quant`, `tinyllama_vs_hf`, `profile_decode`) refuse an existing output unless given `--overwrite`, checked at startup before any GPU work. Default names now include the mode (`serving_graph_paged.json`, `kv_b8-32.json`, …). `bench_generate` also records warmup / repeats in provenance.
- **`src/tinyllm/results.py`** (pure rendering, no GPU): the list of 14 benchmarks, loading with a clear "missing file → `--only NAME`" message, heat lines (busy SM clock median / min, peak temperature, throttle reasons), and the tables: TinyLlama B=1 vs the ceiling, decode grids, INT8, serving, logits vs HF, decode profile. Renders `docs/RESULTS.md` and the three README blocks.
- **`scripts/bench_all.py`:** `--run --power-mode X` runs everything into `runs/bench/v1/` (refuses a dirty tree or a non-empty dir), waits for ≤ 60 °C before each benchmark, copies output to `run.log`, writes `manifest.json`, stops at the first failure. `--only a,b` reruns some; the default only renders. `--list` shows the commands (~90 min plus cool-downs).
- **Docs:** README results wrapped in `<!-- bench:highlights | tinyllama | serving -->` markers, "Full tables" now points to RESULTS.md. INFERENCE.md: a pointer to RESULTS.md, and the 4.9d continuous table's source (`sweep.log`). D-029. PLAN 6.8 wording fixed (throttle rule uses the busy-clock median).
- **Checks:** 8 CPU tests in `tests/test_results.py` (render from made-up JSON, marker replacement leaves the rest byte-identical and is idempotent, bad markers refused, missing file names the rerun, heat warnings, overwrite guard, per-mode default names, unique outputs). Full suite **292 passed**. With no results, the render exits 1 and lists all 14 missing files with their `--only` commands. On the real README with fake data, text outside the markers is unchanged.
- Committed `4d72033`.
- **First GPU run crashed (10:51 UTC):** `30m-naive` (bf16, never benchmarked before; 4.1 was fp32) failed at B=32 P=128 with `CUBLAS_STATUS_INTERNAL_ERROR` in the lm_head matmul, after 7 good rows. Cause: the naive engine's full forward grows by one position per step, so no cached block fits the next request. The caching allocator keeps reserving new blocks until the GPU is full: **7,991 MiB in use for a 475 MiB peak**, `free: 0` on a 192 MB request. PyTorch catches its own OOM and retries after emptying its cache (that is 4.1's `free: 0` warning on the same row), but cuBLAS's internal allocation does not.
  - Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the two naive benchmarks only (the other engines use fixed-size caches; graph capture is untouched). Same row, isolated, two runs each: **695 MiB**, no OOM, 940 / 921 tok/s vs 963 / 935 without it. `bench_all` records the env per benchmark in the manifest and the log. +1 test.
  - The failed attempt is kept in `runs/bench/v1-failed-1/`.
  - The GPU reached 84 °C during that one row, from a < 60 °C start: the cool-down gate controls only the start.
- **Next:** commit the fix, rerun into a clean `runs/bench/v1/`, then fill this entry's numbers and tick 6.8.

### 2026-11-06 — 4.8b: our 30M model exported to HF Llama format
- Pushed `e336501` (4.8a): CI green.
- Most of the model side existed from 4.9c (`save_as_hf`, HF loads our export with matching logits on random small models). New:
  - **Bug fixed:** `save_as_hf` wrote no token ids, so HF used LlamaConfig's defaults (bos 1, eos 2) and its `generate()` would stop on the wrong token. It now writes `eos_token_id` / `bos_token_id` (null when absent) to `config.json` and a `generation_config.json` (pad = EOS). It also refuses layer types other than mha / gqa.
  - `save_tokenizer_as_hf`: our tokenizer was trained with `tokenizers` (D-007), so the file already is a `tokenizer.json`. The added `tokenizer_config.json` sets `PreTrainedTokenizerFast` and EOS `<|endoftext|>` (id 0), with no BOS (like our server) and no space clean-up on decode.
  - `scripts/export_hf.py`: `checkpoints/main/best.pt` → `runs/export/tinystories-30m/` (git-ignored; 119.6 MB fp32 safetensors + tokenizer). Then it verifies on CPU in fp32 against HF Transformers (reference only, D-015) and writes `runs/export/tinystories-30m-verify.json` with provenance.
  - 3 CPU tests (they run in CI): the exported tokenizer gives our ids and text back (leading spaces, Unicode, EOS in text); HF `generate()` stops on our eos id; non-Llama layers are refused. Full suite **284 passed**.
- **Verification** (step 7599, val loss 1.282; 30M params, tied embeddings):

  | Check | Result |
  |---|---|
  | Logits, full forward (16 validation stories, 2,741 positions) | max abs diff 1.4e-5, top-1 agreement 100%, KL ≈ 0 |
  | Logits, prefill 8 + cached decode | max abs diff 1.0e-5, top-1 agreement 100% |
  | Greedy generation, 8 unseen prompts, ≤ 200 new tokens | 8/8 identical to HF `generate()`; 7 stopped at EOS |
  | Tokenizer, whole validation split (21,981 stories, 4.66 M tokens) | 0 id mismatches, 0 decode mismatches |
  | Round trip through our `load_from_dir` | identical weights |

  - So the export is a standard Llama: anyone can use it with `AutoModelForCausalLM` / `AutoTokenizer`, without our code.
- Not done: GGUF (stretch; llama.cpp's converter identifies BPE tokenizers by a fingerprint and won't know ours). Uploading to the Hub with a model card is 6.5b. Parked: let our server load the exported folder (PARKING_LOT 2026-11-06).
- **Next:** 6.8, one-command tables.

### 2026-11-06 — 4.8a: TinyLlama on ours vs vLLM vs llama.cpp (closes 4.13)
- `scripts/load_test.py --sequential N --weight-bytes B`: one request at a time; decode tok/s excluding the first token, as a % of the bandwidth ceiling (423 GB/s ÷ weight bytes). 2 tests (summary logic; a live sequential run against the tiny server). Full suite **281 passed**.
- **API check** (all three servers, same request): exactly `max_tokens` tokens, 14 prompt tokens, usage in the stream, finish `length`, `[DONE]`. vLLM and llama.cpp send usage in a last chunk with no text; the client already handled it.
- **Setup:** one server at a time, NitroSense **Performance** mode, fans Auto. TinyLlama @ `fe8a4ea`, bf16 unless noted. Prefix caching off (vLLM `--no-enable-prefix-caching`, llama.cpp `--no-cache-prompt`; confirmed in the server logs). llama.cpp: 8 slots × 2048 context, KV not unified. B=1: 20 × 256 tokens. Sweep: 128 tokens, 60 s per rate, local SLOs. 0 failures in every run; GPU back to 0 MiB after each server.

  | Setup | Seats | B=1 decode | % of ceiling | B=1 TTFT p50 | Rated load | Saturation | ITL p50 / p99 at 4 req/s | Past saturation |
  |---|---|---|---|---|---|---|---|---|
  | ours, 8 slots, graphs | 8 | 111.7 tok/s | 58.1% | 41 ms | 4 req/s | ~5.5 req/s, ~710 tok/s | 9 / 41 ms | 429s, TTFT p95 ~5.9 s |
  | ours, 32 slots, graphs | 32 | — | — | — | 8 | ≥ 12.2 req/s, 1,555 tok/s at 16 | 13 / 46 ms | 429s, TTFT p95 2.6–2.9 s |
  | vLLM 0.33, 8 seqs | 8 | 160.7 | 83.5% | 16 ms | 6 | ~8.5 req/s, ~1,090 tok/s | 7 / 15 ms | no 429s, TTFT p95 48 s at 16 |
  | vLLM 0.33, default (256) | 256 | 160.9 | 83.7% | 16 ms | 32 | ~34 req/s, ~4,370 tok/s | 7 / 9 ms | TTFT p95 21 s at 48 |
  | llama.cpp bf16, 8 slots | 8 | 151.0 | 78.5% | 12 ms | 4 | peak 5.4 req/s at 6, falls to 3.5 at 16 | 10 / 30 ms | no 429s, TTFT p95 203 s at 16 |
  | llama.cpp Q8_0, 8 slots | 8 | 233.9 | 64.7% of 361.6 | 11 ms | 4 | peak 6.2 req/s, falls to 3.8 at 16 | 9 / 29 ms | no 429s, TTFT p95 182 s at 16 |

  - **B=1:** vLLM is 1.44× us; llama.cpp bf16 1.35×. That's kernel speed: FlashAttention and fused ops vs our SDPA over padded slots.
  - **Under load, seats decide.** Little's law: 8 ÷ ~1.4 s per request ≈ 5.7 req/s for us, 8 ÷ ~0.9 s ≈ 8.9 for vLLM-8. 32 slots doubled our rated load (4 → 8 req/s), with ITL p50 rising 9 → 15 ms. vLLM's default (paged KV pool, 256 seqs) had only ~16 requests in flight at 16 req/s, and saturates at ~34 req/s.
  - At 8 seats we tie llama.cpp bf16 under load (rated 4 req/s each, ~690–710 tok/s) despite its faster B=1.
  - **Overload:** we shed (429s, bounded TTFT). vLLM queues without limit. llama.cpp's throughput drops by ~35% past its peak; cause not investigated (not our client: it handled 947 streams against vLLM).
  - **Throttling:** every run was thermally throttled (84–87 °C; `sw_power_cap`, `sw_thermal_slowdown`, some `hw_slowdown`). SM clock medians 1,515–1,770 MHz in the sweeps. At B=1 ours ran at 1,875 MHz / 76 °C, the others at 1,530–1,695 MHz, so our B=1 number is if anything flattered. Differences within ~15% are ties; the 1.3–1.5× gaps are not.
  - All records carry git `ccdc521+dirty` (this uncommitted `--sequential` mode). Raw JSON + server logs in `runs/bench/load/4.8a/` (git-ignored).
- Written up in docs/SERVING.md, "Load testing" → "Results". Prefix caching explained in SERVING.md (not implemented; A for v1.1, B in Phase 7). Parked: the leading space dropped from served text (PARKING_LOT 2026-11-06).
- **4.13 closed:** its done-when (greedy outputs = `generate()`, load test against our server and vLLM) is met.
- **Next:** 4.8b (export the 30M model to HF Llama format), then 6.8.

### 2026-11-06 — 4.13 step 3: SLOs + load generator; first load test on TinyLlama
- Pushed `8a97d70` + `f33e881`: CI green (49 s; GPU / TinyLlama tests skip there, the 22 server tests run).
- docs/SERVING.md, "SLOs" (written before measuring; changing a target needs a DECISIONS entry):
  - TTFT p95 ≤ 1 s local / ≤ 2 s Space
  - availability ≥ 99%
  - ≤ 1% 429s at or below rated load
  - error budgets; the burn-rate rule for the 6.5b status page
- `scripts/load_test.py`:
  - Poisson arrivals per rate (seeded); streamed, `ignore_eos`, fixed `max_tokens`, so the same work for our server and vLLM.
  - Client-side TTFT / ITL / latency p50 / p95 / p99, tok/s, 429s, failures, SLO checks per rate, and the **rated load** (the highest rate where it and every lower rate meet all SLOs).
  - JSON with provenance in `runs/bench/load/`. `--power-mode` is required.
  - 3 tests (Poisson rate, summary / SLO / rated-load logic, a live run against the tiny server that must shed at 100 req/s). Full suite **279 passed**.
- **Load test, TinyLlama 1.1B** (bf16, `--slots 8 --cuda-graph`, max queue 32; NitroSense **Performance** mode, fans Auto, GPU otherwise idle):
  - Setup: 8 story-opening prompts, 128 tokens each, 60 s per rate.

    | rate (req/s) | sent | ok | 429 | fail | tok/s | TTFT p50 / p95 / p99 | ITL p50 / p99 | latency p99 | SLO |
    |---|---|---|---|---|---|---|---|---|---|
    | 0.5 | 23 | 23 | 0 | 0 | 49 | 83 / 162 / 249 ms | 9 / 31 ms | 1.55 s | ok |
    | 1 | 56 | 56 | 0 | 0 | 119 | 48 / 85 / 100 ms | 9 / 23 ms | 1.41 s | ok |
    | 2 | 125 | 125 | 0 | 0 | 264 | 50 / 97 / 124 ms | 9 / 42 ms | 1.53 s | ok |
    | 3 | 173 | 173 | 0 | 0 | 366 | 51 / 222 / 604 ms | 9 / 44 ms | 1.98 s | ok |
    | 4 | 242 | 242 | 0 | 0 | 506 | 50 / 795 / 1050 ms | 9 / 42 ms | 2.47 s | ok |
    | 6 | 381 | 348 | 33 | 0 | 689 | 4.50 / 5.81 / 5.90 s | 10 / 43 ms | 7.31 s | miss: TTFT, 429 |
    | 8 | 454 | 362 | 92 | 0 | 693 | 5.25 / 5.83 / 5.90 s | 10 / 45 ms | 7.33 s | miss: TTFT, 429 |

  - **Rated load 4 req/s** (~506 tok/s). Saturation ~5.4 req/s, ~690 tok/s. That's above the 4.11b offline 581 tok/s at 8 slots, because here every request is the same length, so there's no long-tail drain.
  - Past capacity the server behaves as designed: the queue fills, the rest get 429, **0 failures**, and TTFT is capped at ~5.9 s = 32 queued ÷ 5.4 req/s. The server's `/stats` agrees (1,458 requests, error rate 0, 429 rate 8.6%, queue wait p95 5.7 s).
  - ITL p50 stays at 9–10 ms (~105 tok/s per stream) at every load. ITL p99 of 30–45 ms comes from prefills of newly admitted requests stalling the decode step. The prompts are ~15 tokens, so the stall is one eager prefill forward, not prompt length: capturing the prefill as a CUDA graph would fix it, chunked prefill would not (PARKING_LOT 2026-11-06).
  - At 0.5 req/s TTFT p50 is *higher* (83 vs 48 ms). Likely the GPU dropping clocks between sparse requests; provenance has the clock samples.
  - Server: 1,554 JSON log lines (1,458 request lines, no text); VRAM 2.5 GB, RSS 1.5 GB; clean drain on SIGTERM.
- Observation for later: a 32-place queue lets admitted requests wait ~6 s, far past the 1 s TTFT SLO. A queue sized from the SLO (≈ TTFT target × throughput ≈ 5) would shed more but keep admitted requests within the SLO. Parked (PARKING_LOT 2026-11-06).
- Measured the same day with `--max-queue 5` (same setup, Performance mode; `runs/bench/load/tinyllama-8slots-graph-q5.json`):

  | Rate (req/s) | Queue | OK | 429 | TTFT p95 | Latency p99 | tok/s |
  |---|---|---|---|---|---|---|
  | 6 | 32 | 348 | 33 | 5.81 s | 7.31 s | 689 |
  | 6 | 5 | 323 | 37 | **0.92 s** | **2.52 s** | 670 |
  | 8 | 32 | 362 | 92 | 5.83 s | 7.33 s | 693 |
  | 8 | 5 | 334 | 156 | **1.13 s** | **2.69 s** | 693 |

  - Throughput is unchanged (saturation ~5.4 req/s); the short queue only moves waiting time out of TTFT and into 429s, which a client can retry. Past capacity, TTFT p95 drops 5–6× and latency p99 ~2.8×.
  - At 4 req/s neither queue sheds and both meet the SLOs (rated load 4 req/s either way). The TTFT p95 there (0.45 vs 0.80 s) differs with the arrival sample, so it isn't a fair comparison.
  - At 8 req/s TTFT p95 is still 1.13 s: about 5 / 5.4 ≈ 0.93 s of queue wait plus prefill and step jitter. A queue of 4 would fit the 1 s target. The server default stays 32 for now. The Space default (6.5b) should come from this rule: queue ≈ TTFT target × saturation rate, minus some headroom.
- Next: 4.8a, the same `load_test.py` against vLLM (and llama.cpp) on TinyLlama.

### 2026-11-06 — 4.13 step 2, parts 3 + 4: the HTTP server and failure injection
- `src/tinyllm/server.py` (FastAPI + uvicorn, D-025 / D-028; design in docs/SERVING.md):
  - `EngineWorker`: one thread owns the `ContinuousEngine`; handlers talk to it through an inbox; tokens come back over a per-request `asyncio.Queue` via `call_soon_threadsafe`.
  - `POST /v1/completions`: text or token ids, `max_tokens`, `temperature`, `top_p`, `top_k`, `ignore_eos`, `return_token_ids`; SSE streaming ending with `[DONE]` (incremental detokenizer: the streamed text equals the non-streamed text); `finish_reason` stop / length; `usage`; OpenAI error shape.
  - Hardening: a bounded queue (slots + `--max-queue`) → 429 + `Retry-After`; a per-client token-bucket rate limit → 429; a per-request timeout → 504 with the slot freed; client disconnect → abort (499 in metrics); graceful drain on shutdown; an engine error → 500, `/health` + `/ready` 503, new requests 503; OOM → 503 for in-flight requests, then recovery.
  - Observability: a JSON log line per request (no text; salted client hash), `X-Request-ID` (the client's if sane), `X-Model-Version` (checkpoint hash + git commit), engine snapshots every 5 s, `/stats`, `/metrics`, per-step debug timings, slow-request timelines, `/debug/state` (localhost only).
  - `Router` with one variant (6.9 adds release rules here). CLI: `python -m tinyllm.server --model tinyllama | --ckpt PATH`.
  - `Faults` (test only): a slow step, an exception at step N, an OOM at step N.
- `engine.py`: `load()` (queued, busy) and `kv_used_frac()`.
- Deps: fastapi 0.162.0, uvicorn 0.54.0; dev: httpx 0.28.1.
- `tests/test_server.py`, 22 tests against real uvicorn on 127.0.0.1, port 0 (httpx's ASGI transport buffers streams):
  - greedy == `generate()` for id and text prompts, and with 7 concurrent requests on 3 slots
  - stream == non-stream; EOS → `stop`
  - 9 kinds of 400
  - 429 on a full queue; the rate limit; 504 (non-stream and stream)
  - disconnect (non-stream and stream) → slot freed
  - shutdown mid-request → 200
  - metrics, stats and log fields
  - the injected engine error and the injected OOM
- Bug found by the tests: the non-stream disconnect check only ran when no token had arrived for 0.25 s, so a busy request never noticed. It now checks on a timer.
- Three reruns, all green (~11 s each). Full suite **276 passed** (116.9 s).
- CLI smoke on `checkpoints/main/best.pt` (GPU, 4 slots): 200 with headers, the stream text equals the non-stream text, `/stats` filled, JSON logs, clean shutdown.
- Next: 4.13 step 3, SLOs first, then `scripts/load_test.py`.

### 2026-11-06 — 4.13 step 2, part 1: per-request sampling
- `engine.py`:
  - New frozen `Sampling(temperature, top_k, top_p)`, which checks its own values.
  - New `Request.sampling` (None = the engine's settings).
  - `ContinuousEngine._sample` makes one `sample_next` call per distinct setting. When every row has the same settings, the result and RNG use are exactly as before. Idle decode rows use the engine default.
  - `run_static` rejects per-request settings instead of ignoring them.
- New tests:
  - greedy and `top_k=1` rows mixed with hot sampled rows (staggered arrivals) equal solo `generate()`
  - hot rows do differ, and the same seed gives the same tokens
  - explicit settings equal to the engine's sample exactly as without them
  - bad settings are rejected
- Full suite **247 passed**.
- Part 2, `obs.py` (stdlib only):
  - `RequestRecord`: one JSON log line per request, never text; the client is a salted hash
  - `EngineSnapshot`
  - `Metrics`, thread-safe: rolling 5 min / 1 h windows for `/stats`; counters and TTFT / latency histograms by variant for `/metrics` (Prometheus text format)
  - `JsonFormatter` and `setup_logging`
  - 7 tests, all passing.
- Next: part 3, `server.py`.

### 2026-11-06 — Serving design: production layer, observability, release strategies (D-028)
- Design discussion before 4.13 step 2. Recorded in **D-028**, **[docs/SERVING.md](docs/SERVING.md)**, PLAN 4.13 / 6.5b / 6.9 and PARKING_LOT.
- What is deployed:
  - TinyLlama 1.1B on the local server, for load tests and 4.8a
  - our 30M model on the public CPU Space, framed as story continuation (TinyStories base model, no SFT / RLHF)
- 4.13 step 2 now also includes per-request sampling, hardening (429 queue limit, rate limit, timeouts, disconnect → abort, drain), an observability core (JSON request log, request ids, `/stats`, `/metrics`, debug mode) and a one-variant router with `X-Model-Version`. 4.13 is now ~11–13 h.
- 6.5b adds a status page, an SLO, private log retention, a usage report, a privacy notice and the scaling write-up (+2–3 h).
- New 6.9 (v1.1, ~11-19 → 11-21): blue-green, shadow, canary with auto-rollback, A/B.
- v1 date unchanged (~11-18).
- Next: 4.13 step 2.

### 2026-11-06 — 4.13 step 1: `ContinuousEngine`
- `engine.py`: `run_continuous`'s loop is now `ContinuousEngine` with `add_request()` (any time; returns an id; rejects requests that can never fit), `step()` (admit + one decode step; returns the new tokens as `Event(rid, token, done)`), `abort()` (frees the slot or drops the queued request) and `results` (popped by the caller). `run_continuous` is a thin wrapper, so every 4.4 / 4.5 / 4.11b test checks the refactor unchanged.
- TTFT and latency now count from each request's arrival. With all requests added at t=0 they equal the old numbers.
- Sampling settings stay per engine. Per-request sampling is not needed for the greedy done-when; decide in step 2 if the server needs it.
- New tests: late arrivals (slots and paged, with and without EOS) equal solo `generate()`, the event stream equals the result, abort of a running and a queued request, rejection when added, graphs equal eager under staggered arrivals (CUDA). Full suite **239 passed**.
- Next: step 2, the server.

### 2026-11-06 — 4.11b CUDA graphs in the continuous engine
- `SlotGraphDecoder` (`cuda_graph.py`): one graph per attention-window bucket (128, 256, … max_len) in a shared pool. `SlotKVCache` gets a fixed `window` (the bucket) and `advance_host()`, the host half of `end_step` (a replay runs only the device half). Prefill and sampling stay eager.
- `run_continuous(cuda_graph=True)` builds a decoder per call; passing a `SlotGraphDecoder` reuses its cache and graphs (the benchmark captures once, outside the timed runs). Paged KV raises "kv='slots' only".
- Tests: replay is bit-identical to the windowed eager step; the slot graph equals the eager slot step (gap 0); `run_continuous` with graphs equals eager at 1 / 3 / 8 slots, with and without EOS, with joins / leaves and bucket switches mid-run; decoder reuse doesn't recapture; TinyLlama teacher-forced, 0 flips in 240 slot-steps. **Graphs equal eager exactly here**, unlike 4.11's `generate()`: the eager slot step already uses a mask, so both use the same SDPA kernel.
- Full suite: one run had `test_model.py::test_bf16_autocast_cuda` fail (an unseeded model, checked within 0.1 of log V; passes alone). The new CPU-model test now builds its model inside `torch.random.fork_rng()` so it leaves the global RNG unchanged. Rerun: **232 passed** (157 s).
- `bench_serving.py --model tinyllama --cuda-graph` (64 requests, 9,004 tokens, median of 3, NitroSense **Performance** mode):

  | Slots | static | continuous (eager) | cont + graph | graph vs eager | latency mean / p95 (s), graph | capture |
  |---|---|---|---|---|---|---|
  | 8 | 181 | 245 | **581** | 2.37× | 7.4 / 13.5 | 0.27 s |
  | 32 | 631 | 679 | **956** | 1.41× | 4.8 / 8.2 | 0.25 s |

  - Same tokens as eager at both sizes: identical divergence from solo `generate()` (22/64 and 32/64 identical).
  - The graph rows run the GPU at 93–97% utilisation (eager: 44–65%), hot enough to throttle (`sw_thermal_slowdown`, 84 °C, SM 1575–1725 vs 1785–1875 MHz for eager). The gain is measured at a lower clock.
  - The eager rows are ~9% below the 2026-11-04 run (245 vs 272, 181 vs 198 at 8 slots). Eager is host-bound, so CPU speed and load move it.
  - A first run was discarded because the laptop switched from Default to Performance mode mid-run (memory clock 7001 → 7201 MHz). In Default mode, eager at 8 slots gave 209 tok/s. Benchmarks before 2026-11-06 ran in Performance mode; from now on each entry names the mode.
- Next: push the local commits, then 4.13.

### 2026-11-05 — Planned 4.11b and 4.13 (D-027)
- 4.13 plan made: an engine object with `add_request()` / `step()`, a FastAPI server with `/v1/completions` (streaming, `ignore_eos`), a Poisson load generator (httpx), and CPU tests. Estimate ~7–9 h.
- Found that the continuous engine has no CUDA graphs: eager at ~22 ms/step, so served latency would mostly measure kernel-launch overhead, and vLLM uses graphs.
- D-027: new step 4.11b (graphs in the continuous engine, slot cache first, ~1 day) **before** 4.13, so 4.13's first load test shows graph-speed numbers.
- Hosting: the server runs locally in WSL for benchmarks. The public demo stays 6.5b (HF Space).
- Next: 4.11b on 2026-11-06.

### 2026-11-05 — 4.8-spike: vLLM + llama.cpp install
- Everything is in `~/spike/` outside the repo, each in its own venv. Model: TinyLlama @ `fe8a4ea` from the HF cache.
- **System change:** `cuda-toolkit-13-0` (nvcc 13.0.88) from NVIDIA's wsl-ubuntu apt repo, installed into `/usr/local/cuda-13.0`, plus cmake 3.28. No driver packages; libcuda is still `/usr/lib/wsl/lib`. Not on PATH, and `.bashrc` is untouched.
- **vLLM 0.33.0:**
  - Its venv is 8.9 GB with its own torch 2.14.0+cu130; the install took 21 min.
  - Offline greedy decoding works, ~153 tok/s at B=1. That's rough: one run with prefill included, not comparable yet.
  - `vllm serve` is ready in 77 s and answers `/v1/completions`. The GPU is back to 0 MiB after it stops.
  - **Issue:** the sampler uses FlashInfer, which JIT-compiles on first use and fails with `ninja` not found, even though vLLM logs "FlashInfer kernels are disabled". Workaround: `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy doesn't need it). For 4.8a, also try `vllm download-kernels` (prebuilt) and record which sampler ran.
- **llama.cpp `61d4472`:**
  - CUDA build for sm_86 (llama-cli/server/bench/quantize).
  - **Issue:** CMake found the *Windows* CUDA 12.4 through the `/mnt/c` entries WSL adds to PATH. Fix for the build only: drop `/mnt/` from PATH and set `CUDAToolkit_ROOT=/usr/local/cuda-13.0`.
  - Converter venv: 0.9 GB, pins torch 2.11 CPU and transformers 4.57.6.
  - GGUF files: bf16 2,100 MiB, Q8_0 1,116 MiB.
  - `llama-bench`, pp128 / tg256, all layers on GPU, 3 reps:

    | GGUF | tg256 tok/s | pp128 tok/s |
    |---|---|---|
    | bf16 | 135 ± 6 | ~3,600 (± 2,100, noisy) |
    | Q8_0 | 238 ± 7 | ~8,500 (± 3,600) |

  - `llama-server` is ready in 4 s and answers `/v1/completions` (126 tok/s for that request).
  - Its greedy text is word for word the same as vLLM's.
- **First look, to verify in 4.8a under the same protocol:**
  - At bf16, B=1, vLLM (~153) and llama.cpp (135) are close to our 128.
  - llama.cpp Q8_0 at 238 tok/s is the bar for the fused INT8 kernel. Our INT8 path is 0.80× bf16 today.
- **Project env unchanged** (before vs after snapshot):
  - Same `pip list`; same hashes of `uv.lock`, `pyproject.toml`, `.bashrc` and `.profile`; same torch info.
  - **221 passed** after the spike as before.
  - check_env: bf16 matmul 39.3 vs 40.0 TFLOPS, copy/read 409/419 vs 409/426 GB/s (noise).
  - Torch still loads its own pip CUDA libraries (`nvidia/cu13/lib`), not `/usr/local/cuda`.
  - First test run after the spike: 136 s vs 84 s before; the rerun took 97 s. Likely a cold file cache after the 3 GB of GGUF files (7 GB RAM), not an env change.
- Cleanup if wanted: `rm -rf ~/spike ~/.cache/vllm`, then `apt remove cuda-toolkit-13-0 cuda-keyring` + `apt autoremove`.
- Next: 4.13 (D-025).

### 2026-11-05 — Repo public → v1-preview
- github.com/cooldip00/tiny-llm is public:
  - The page loads without a login, and the CI badge shows "passing" to anonymous visitors.
  - About box set: a description and 10 topics.
- v1-preview was planned for about 2026-11-11, so this is 6 days early.
- 6.5a is still open until the README plots are in (v1, PARKING_LOT "README polish").
- Next: the vLLM + llama.cpp install spike, then 4.13 (D-025).

### 2026-11-05 — First README (6.1 draft) + MIT license
- `README.md` is results first, for the 30-second reader (6.5a):
  - CI badge and 4 highlight bullets.
  - Results tables: TinyLlama decode against the ceiling, serving, training.
  - Two Mermaid diagrams: the model block and the serving request path.
  - Repo layout, reproduce commands, roadmap.
- Every number is checked against INFERENCE.md and this log.
- `LICENSE`: MIT. Without a license, a public repo is "all rights reserved".
- Left for v1, in PARKING_LOT "README polish": plots, a demo GIF, Space and blog links, a social preview image, design-decision and plain-language sections.
- Gap: `runs/` is git-ignored, so the raw benchmark JSONs aren't in the repo. Decide in 6.8 whether to commit them.
- 6.7 is closed by the badge.

### 2026-11-05 — Step 6.7 CI
- `.github/workflows/ci.yml`: runs on push, pull request and manual dispatch, on ubuntu-24.04 with uv and Python 3.12.
  - It installs CPU torch 2.14.1 before the project, so the CUDA 13.0 wheels from pyproject (D-017) are never pulled.
  - The job fails if any `nvidia-*` package is installed.
- Expected skips (22):
  - 18 need CUDA.
  - 2 need `data/*.bin`.
  - 2 need TinyLlama in the HF cache.
- Results:
  - Local simulation (fresh clone, empty HF cache): 199 passed, 22 skipped, no nvidia packages.
  - GitHub: first run 43 s total (pytest 14.7 s). After the bump to `actions/checkout@v7` and `astral-sh/setup-uv@v10.3.3` (clears the Node 20 warning), 199 passed, 22 skipped in 20.4 s.
- `gh` CLI is set up locally with a read-only fine-grained token (tiny-llm only; Actions and Contents read) to check runs.
- Next: the first README, including the badge, closes 6.7. Then make the repo public (v1-preview).

### 2026-11-04 — Step 4.9d TinyLlama benchmarks against the bandwidth ceiling
- `bench_generate.py`, `bench_serving.py`, `bench_quant.py` take `--model tinyllama`, which writes to `runs/bench/tinyllama/*.json`.
  - Prompts: validation stories re-tokenized with TinyLlama's tokenizer (`data/validation_tinyllama.bin`, git-ignored, built on first use).
  - Each row reports the ceiling, the % of it, and peak reserved memory.
  - `--engine hf` times HF `generate` (bf16, SDPA) as the baseline, with a greedy and teacher-forced comparison.
  - `bench_quant` on TinyLlama compares bf16 (the baseline) with int8. An fp32 copy doesn't fit in 8 GB.
- Ceiling 192 tok/s per sequence (423 GB/s ÷ 2,098 MiB). Excluding the 125 MiB input embedding gives ~204.
- Decode at B=1, P=128: HF 45 (23.3%), naive 33, eager KV 45 (23.4%), **KV + CUDA graphs 128 tok/s (66.8%)**. At P=32 graphs reach 134 (69.5%).
  - Eager, ours and HF, is host-bound at ~20–22 ms/step whatever B or P.
  - Graphs: B=8 851 (55%), B=32 2,083 (34%; the % counts weight reads only). They slow at long context: B=32 P=512 is 1,254 tok/s.
- Continuous vs static: 1.37× / 1.32× / 1.10× at 8 / 16 / 32 slots. Slots reserve 2,048 positions each, so only 8–10% of the reserved KV holds tokens.
- INT8: val loss Δ +0.0010, argmax 98.49% vs bf16. Speed 0.80× / 0.74× / 0.54× bf16 at B=1 / 8 / 32: the cast materialises a bf16 weight every call. A fused kernel is 4.10b.
- Near-ties only:
  - HF vs ours: 6 teacher-forced flips, all at margin ≤ 0.125.
  - Graphs vs eager: 21 flips, ≤ 0.125.
  - KV vs naive: max diff 1.53, consistent with 4.9c.
- **WSL finding:**
  - When the caching allocator's reserved memory passes physical VRAM, WSL spills to host memory instead of raising OOM. Eager B=32 P=512 then took 10.6 s to its first token (it should take 1.2 s).
  - `bench()` now calls `empty_cache()` before each row. The kv-graph bench keeps one live `GraphDecoder`; stale ones caused a real OOM.
  - Eager B=32 still varies ±20% run to run (memory-marginal on 8 GB).
- Docs: TinyLlama results table filled (vLLM / llama.cpp rows wait for 4.8a), plus the INFERENCE.md 4.9d section. A correction note in 4.9b: the GPU is the 3070 Ti Laptop, not a 4060.
- Next: 6.7 CI.

### 2026-11-04 — Step 4.11 CUDA graphs on the decode step
- `src/tinyllm/cuda_graph.py`:
  - `StaticKVCache` uses the same cache protocol, so the model code doesn't change.
    - The position is a device tensor. K/V are written with `index_copy_`, and RoPE is read with `index_select` at that position.
    - Every step attends over a fixed window of W keys with a mask, covering left padding too.
  - `GraphDecoder` keeps prefill and sampling eager and captures one graph per length bucket (128, 256, …, max), all in a shared memory pool.
  - Turned on with `generate(cuda_graph=True | decoder)`. Off by default.
- Correctness:
  - Replay is bit-identical to the same static step run eagerly.
  - Against variable-shape eager it is bf16-close. On the 30M model: argmax agreement 99.41%, and all 12 flips at eager margin ≤ 0.125 (median 1.75). On TinyLlama: 1 flip in 3 × 64 tokens, at margin 0.125.
  - The cause is SDPA choosing flash for the no-mask eager step and mem-efficient for the masked window.
  - `tests/test_cuda_graph.py`: 9 tests. Suite: **221 passed**.
- Speed (bench_generate, 30M, bf16, median of 5): **8.00 → 0.99 ms/step at B=1 (8.1×)**, 8.35 → 1.25 at B=8, 8.64 → 1.75 at B=32 (4.9–8.1× overall). TTFT is unchanged.
- Profile (`scripts/profile_decode.py`, traces in `runs/profile/`):
  - Eager was host-bound. Issuing a step took 8–9 ms of CPU whatever the batch. The GPU was idle 58–92% of the time and downclocked to 800–1,410 MHz.
  - Graph replay keeps 1,875 MHz with the GPU idle ~10%. A launch still costs 0.35–1.4 ms of host time on WSL.
  - The next limit is ~430 small kernels per step, which needs fusion (4.10 / torch.compile).
- TinyLlama B=1: 38–48 → **124–133 tok/s, 65–69% of the 192 tok/s ceiling** (GPU idle 3%).
- `bench_generate.py`: `--engine kv-graph` and `--dtype bf16`. Wrote runs/bench/kv_bf16.json and kv-graph_bf16.json.
- Next: 4.9d.

### 2026-11-04 — Step 4.9c logits match HF Transformers
- `transformers` 5.22.1 added to the `dev` group only. `src/tinyllm` never imports it (D-015).
- `src/tinyllm/logit_compare.py`:
  - `ours_stepwise` / `hf_stepwise`: prefill, then one token per step through each model's own KV cache.
  - `logit_stats`: max/mean abs, KL, top-1. `first_divergence`. Reusable for 4.8b.
- `scripts/tinyllama_vs_hf.py` → `runs/bench/tinyllama_vs_hf.json`. 5 prompts × 64 teacher-forced decode tokens, against HF fp32 eager.
  - **Ours bf16 is bit-identical to HF bf16** at all 415 positions (prefill + cached decode).
  - Error vs fp32, the same for both: KL 3.3e-4 (prefill) / 2.6e-4 (decode); max abs 0.76 / 0.56.
  - Top-1 100% / 98.86%. The 3 flips are near-ties (fp32 margin 0.002–0.019 vs a median of 1.7), at the same positions for HF.
  - Greedy text: ours == HF bf16 on 5/5 prompts.
- Done-when adjusted from the plan: "top-1 ≥ 99%" fails for HF's own bf16 too. It's replaced by "no more flips than HF + 1, each at a near-tie (margin < 0.1)", plus KL and max-abs ≤ 1.5× HF's.
- **Bug fixed:** transformers 5 configs put θ in `rope_parameters`, and `config_from_hf` would have silently used 10000. It now reads both layouts and rejects scaled `rope_type`s. TinyLlama's own config is the old layout, so 4.9b's results stand.
- Tokenizer: we follow the shipped `tokenizer.json`. transformers 5 `AutoTokenizer` applies `legacy: false` on top. Two documented differences (an extra `▁` after special tokens, one leading space), plus no BOS in its chat template. The chat string itself is identical. Details in INFERENCE.md 4.9c.
- Tests: `tests/test_hf_reference.py`, 8 tests:
  - 6 CPU tests (random tiny Llamas in fp32 vs HF eager, both conversion directions, cached decode, greedy generate). These run in CI.
  - 2 GPU TinyLlama tests: tokenizer differences, and the bf16-vs-fp32 done-when.
  - `test_hf_llama.py` +2 (`rope_parameters` layout and scaled rope_type).
- Next: 4.11 decode-step overhead.

### 2026-11-04 — Step 4.9b load TinyLlama
- D-026: `TinyLlama-1.1B-Chat-v1.0`, pinned at `fe8a4ea`.
  - Single bf16 file (2,098 MiB), in the HF cache only.
  - Our own safetensors reader/writer; no new dependency.
- `src/tinyllm/safetensors_io.py` (~90 lines):
  - `read_header`; `load_file` memory-maps the file, checks byte counts, and copies each tensor once to its device/dtype.
  - `save_file` sorts names and pads the header to 8 bytes.
- `src/tinyllm/hf_llama.py`:
  - `config_from_hf` raises on features we don't implement.
  - Name mapping is just the `model.` prefix.
  - `load_from_dir` builds on meta, then a strict name/shape check, then `assign=True`, then rebuilds RoPE on the real device. Tied and untied heads both work.
  - `save_as_hf` (for tests and 4.8b), `HFTokenizer` (`tokenizers` + Zephyr chat prompt).
- `scripts/tinyllama_smoke.py`:
  - Load 3.8 s; **1,100,048,384 params = `estimate_params`**; all bf16; **2.05 GiB** VRAM.
  - Greedy chat is coherent ("The capital of France is Paris." + EOS).
  - About 40 tok/s eager at B=1 (not a benchmark).
- Tokenizer chat template checked by hand against our `chat_prompt` (same string under HF's `trim_blocks`). 4.9c checks it token by token.
- Tests: `tests/test_hf_llama.py`, 16 tests:
  - safetensors round trip over all dtypes; a hand-built file; bad byte count
  - TinyLlama config → 1.1B exactly; 6 unsupported-config cases
  - HF-dir round trip (tied and untied) gives identical logits and generate output
  - missing/extra weights raise; bf16 load rebuilds RoPE
  - TinyLlama header == our state dict; tokenizer ids (the last two need the cached download)
- Suite: **202 passed**.
- `DECISIONS.md` had D-023 twice: an accidental duplicate pasted after D-025.
  - The two copies differed only in how they cite the training setup ("D-019's 'Other' line" vs "the training 'Other' choices above"). The first is the precise reference.
  - With the user's OK, I merged them into the first copy and removed the second. This is a one-off exception to the D-021 additions-only rule, and no content was lost.
- Next: 4.9c logits match HF.

### 2026-11-04 — Step 4.9a `gqa` attention
- `src/tinyllm/attention/gqa.py`, registered as `"gqa"`. `k_proj` / `v_proj` map d → n_kv_heads·hd, so every cache stores only the KV heads (the caches and engines already sized by `n_kv_heads`).
  - Head order matches HF's `repeat_kv`: query head i reads KV head i // g. That's what TinyLlama's weights expect.
  - With `n_kv_heads == n_heads` it is exactly `mha` (same names and shapes; an `mha` state dict loads into it).
- `config.py`: default `layer_types` is `"gqa"` when `n_kv_heads < n_heads`, and `get_config(..., n_kv_heads=…)` re-derives it. Existing configs and checkpoints are unchanged.
- **Kernel choice, measured** (torch 2.14, TinyLlama heads 32 / 4, hd 64, bf16): SDPA `enable_gqa=True` takes **0.07–0.74×** the time of `repeat_kv` + SDPA, masked cases included, with the same outputs. Copying K/V every decode step is the expensive part (B=32, 512 positions: 1.09 ms vs 0.08 ms). I had expected masks to force the slow path; they don't. `repeat_kv` stays as the test reference.
- Tests: `tests/test_gqa.py`, 14 tests:
  - matches a per-head reference loop; equals `mha` exactly at equal heads; HF head order
  - causal; chunked KV cache == full pass; bf16 CUDA vs reference
  - cache bytes (1/4 of mha at 2 KV heads); params == `estimate_params`; a training step backprops
  - model level: `generate` == `sample`; slots and paged engines == solo; `quantize_model` works
- Suite: **186 passed**.
- Next: 4.9b load TinyLlama.

### 2026-11-04 — Step 4.12 benchmark provenance
- `src/tinyllm/provenance.py`: `Provenance(**extra).start()` / `.finish()` returns a dict of static info (git commit + dirty, python / torch / CUDA / cuDNN, GPU + driver, the weight dtype), plus `before` / `during` / `after` samples.
  - `during` comes from one `nvidia-smi -lms 500` process beside the run (min / median / max SM + memory clock, temperature, power, utilisation; busy-only SM clock; NVML throttle reasons seen; WSL load).
  - Windows CPU % (via `powershell.exe`, ~2 s) is read only before and after. Missing tools give None, never an error.
- Wired into `bench_generate.py`, `bench_serving.py` (also `gpu_during` per row) and `bench_quant.py`, as a `"provenance"` key in each JSON. All three smoke-tested.
- **Finding:** during B=1 KV decode the GPU runs at **~860 MHz SM / ~27 W** with only `gpu_idle` set, versus ~1,875 MHz boost and ~1,350 MHz under a heavy matmul at the 135 W cap. The step is so launch-bound that the GPU downclocks between kernels. More evidence for 4.11.
- Suite: **172 passed**. `tests/test_provenance.py`: 6 tests (parsing, N/A handling, throttle decoding, summaries incl. busy-only clock, git info, no-tools fallback).
- Next: 4.9a `gqa`.

### 2026-11-04 — Step 4.6 INT8 weight-only quantization
- `src/tinyllm/quant.py`: per-output-channel symmetric absmax INT8 (`quantize_per_channel`, `Int8Linear`, `quantize_model`, `weight_bytes`).
  - Forward: `F.linear(x, q.to(bf16)) * scale`. The scale factors out of the matmul exactly.
  - All 57 Linears are quantized, including `lm_head`. With tied embeddings, `lm_head` gets its own int8 copy and the embedding stays float.
- `scripts/bench_quant.py` compares three variants of `best.pt`: fp32 + autocast (the engine so far), bf16-stored weights, and int8.
  - **Quality:** val loss 1.2825 → 1.2826 (**Δ +0.0001**, same as bf16). Argmax agreement with plain fp32: 99.08%, vs 99.34% for bf16. Every flip is a near-tie (margin ≤ 0.10 nats; median 1.70).
  - **Memory:** matmul weights 57 → 28.7 MiB (−50%).
  - **Speed:** INT8 is **slower**: 0.78× / 0.85× / 0.92× fp32 at B=1 / 8 / 32 (82 vs 106 tok/s at B=1). The cast + matmul + scale is three ops per Linear in PyTorch, and decode here is overhead-bound (weight reads are ~2% of a step). As expected in the plan; the fused GEMV (4.10b) and TinyLlama (4.9d) are where INT8 can pay off.
  - **Side finding:** bf16-stored weights are **1.20–1.31×** fp32 + autocast at the same quality, measured for the PARKING_LOT B=1 item. Not adopted yet; needs a DECISIONS entry.
- Suite: **166 passed** (16 new in `tests/test_quant.py`).
- Also fixed the INFERENCE results table: the 4.4 footnote and the 4.5 row had split it in two.
- Next: 4.9a `gqa`.
- D-023 accepted (after the 4.6 commit): from 4.9 on, inference stores weights in bf16; training is unchanged. Earlier numbers stay labelled fp32 + autocast.
- D-024 accepted:
  - new step 4.11 (CUDA graphs on the decode step, profiled), placed between 4.9c and 4.9d;
  - 4.7 becomes prompt-lookup decoding;
  - 4.8 compares on TinyLlama first, with the 30M export as 4.8b.
  - Dates unchanged. New order: 4.9a → 4.9b → 4.9c → 4.11 → 4.9d → README + public.
- D-025 accepted:
  - 4.12 benchmark provenance (first, before 4.9);
  - 4.13 OpenAI-compatible server + load generator (v1, before 4.8a);
  - 6.7 CI (before going public);
  - 6.8 one-command results tables (v1).
  - Revised timeline table in PLAN.md; dates unchanged.
- Fixed the parking-lot test `test_initial_loss_near_uniform`: seeded, and its target corrected to ln V + σ²/2 = 9.088 (it was ln V; measured 9.088 ± 0.026 over 20 seeds). Suite: 166 passed.

### 2026-11-04 — Step 4.5 paged KV cache
- `src/tinyllm/paged_kv.py`:
  - `BlockAllocator`: free list; double free raises.
  - `PagedKVCache`: per-layer block pools and a shared block table. Blocks are allocated on demand; pad tokens are not stored (the pad count becomes a RoPE offset). Reads gather the blocks into a contiguous view, then call SDPA.
  - Preemption of the newest sequence when blocks run out, with recompute (as in vLLM).
- `run_continuous(kv="paged", block_size, kv_blocks)`. `SlotKVCache` gets the same `can_admit` / `reserve_step` / `allocated_tokens` hooks.
- Benchmark (5 repeats, median), with the same 64 MiB of KV as 8 slots:
  - paged runs **25** sequences at once: **1.46×** the throughput at block 16 (924 vs 633 tok/s) and 1.54× at block 8.
  - Pool used 41% → 76%; claimed KV holding tokens 52% → 97%.
  - Gather cost at equal concurrency: −7%.
  - 42 preemptions; 64 slots don't help (budget-bound).
- Bug found by the benchmark: a preempted request (longer context), padded together with a long-remaining fresh one, overflowed the 512-position RoPE table. Admission now checks the padded batch; regression test added.
- Benchmark variance: absolute tok/s moved 15–20% between sessions, and two paged rows had a large spread (host interference, CPU-bound steps). Now `--repeats` with a median. The 4.4 rerun gives 1.37× at 8 slots and only 1.05× at 32, noted in INFERENCE.md.
- Suite: **150 passed**. PARKING_LOT: paged-attention kernel, active-row compaction, admission watermark, prefix caching.
- Next: 4.6 INT8.

### 2026-11-04 — Step 4.4 static and continuous batching
- D-022 accepted: 4.4 + 4.5 before 4.6, dates unchanged.
- Cache protocol refactor: `begin_step` / `layer(i)` / `end_step`, with the mask logic in `kv_cache.py`. `KVCache(pad_lens=)` adds left padding with a padded causal mask. `SlotKVCache` adds per-slot lengths, `RotaryEmbedding.at(positions)` and scatter writes, with no per-step GPU sync for bookkeeping.
- `src/tinyllm/engine.py`: `run_static` (left-pad, decode to the longest in the batch) and `run_continuous` (admit into free slots with a batched padded prefill, then one decode step over all slots; release on length or EOS). Both report throughput, latency, TTFT, row utilisation and KV reserved vs used.
- `scripts/bench_serving.py`, 64 mixed requests: continuous **1.34×** static at 8 (525 vs 391 tok/s, p95 latency 22.1 → 14.8 s, row util 59% → 86%) and **1.12×** at 32 (1,434 vs 1,281 tok/s; 2 waves only, so the drain tail dominates). Only 31–41% of reserved KV holds real tokens, which is 4.5's target.
- Correctness:
  - fp32: both engines are identical to solo `generate()` (`tests/test_engine.py`, 12 tests).
  - bf16: 39–43 of 64 requests are identical. Every divergence is a near-tie: the top-2 gap is ≤ 0.076, against a median of 1.84.
- Suite: **136 passed**.
- Next: 4.5 paged KV.

### 2026-11-04 — Step 4.3 KV cache
- Added `src/tinyllm/kv_cache.py` (preallocated `[B, n_kv_heads, max_len, head_dim]` per layer, allocated on first use with the layer's dtype), `Transformer.forward(..., cache=)` (RoPE offset = `cache.pos`), MHA cache path with correct masks for prefill / single-token decode / multi-token on a non-empty cache, and `generate.generate()` (prefill once, then one token per step, EOS per row). `bench_generate.py --engine kv`.
- Speedup vs 4.1 naive: B=1 P=128 **0.93×** (97 tok/s); B=8 P=128 1.12×; B=32 P=32 **2.3×** (2,976 tok/s); B=32 P=128 **3.1×** (2,767 tok/s). P=512 rows n/a (a cache can't slide past max_seq_len).
- Every row is now ~10–11.5 ms/step: overhead-bound, as 4.1 predicted.
- Correctness: fp32 greedy identical to naive (tests). bf16: 5/8 rows identical, 3 diverge at tokens 51/78/239; teacher-forced max |logit diff| 0.125 on scale 24.1, argmax agreement 99.76% → bf16 near-ties, within the reference-answer rule.
- Bug found and fixed: a cache ↔ layer reference cycle kept finished caches alive until gc (B=32 peak 2,588 MiB → 466 MiB after the fix). Test added.
- Replaced the placeholder `test_cache_not_implemented_yet` with `test_cached_chunks_match_full_pass`. Suite: **124 passed**.
- Next: 4.6 INT8.

### 2026-11-04 — Step 4.2 sampling (temperature, top-k, top-p)
- Moved the token choice into a pure `generate.sample_next(logits, temperature, top_k, top_p, generator)` (B, V) → (B, 1), so the KV-cache and batching engines (4.3+) reuse it. `sample()` gained `top_p` and calls it; greedy behaviour is unchanged (4.1 gate still OK).
- Order follows HF: temperature → top-k → top-p. Top-p keeps the smallest set of most likely tokens with mass ≥ p, always including the top token. Invalid `top_k < 1` or `top_p` outside (0, 1] raise.
- Tests: new `tests/test_generate.py`, 13 tests (known-distribution nucleus sets and renormalised frequency, threshold crossing, top-k then top-p, temperature before top-p, per-row filtering, tiny top-p = greedy end to end, invalid args). Suite: **112 passed**.
- Found: `test_initial_loss_near_uniform` is unseeded and order-dependent (fails alone). The new test uses `fork_rng` so it doesn't shift the global RNG. Logged in PARKING_LOT, not fixed here.
- Next: 4.3 KV cache.

### 2026-11-04 — Step 4.1 naive generation baseline
- Added `scripts/bench_generate.py`: naive greedy generation (full forward per token, sliding window at 512), INFERENCE.md protocol (prompts 32/128/512, 256 new tokens, batch 1/8/32, 3 warmups, median of 5). Prompts are fixed validation windows. Correctness gate: the timed loop equals `sample(temperature=0)` (OK).
- Key numbers (`checkpoints/main/best.pt`, full grid in docs/INFERENCE.md):
  - B=1 P=128: **104 tok/s decode**, TTFT 9.1 ms, peak 199 MiB.
  - B=8 P=32: 800 tok/s. B=32 P=32: 1,273 tok/s. B=32 P=512: 453 tok/s (70.7 ms/step).
- Finding: batch 1 is overhead-bound (~10 ms/step at any context length), ~1.5% of the ~7,000 tok/s bandwidth ceiling for 60 MB of bf16 weights at 423 GB/s. Expect 4.3's KV cache to help mostly at long context and large batch, not at B=1.
- One transient allocator OOM warning in the B=32 P=128 row; a clean rerun matched (873 vs 891 tok/s).
- Pushed the repo to GitHub (private) before starting.
- Next: 4.2, add top-p to `sample`, with tests.

### 2026-11-04 — Scope decision for v1 (D-021)
- No code changes. Planning only.
- Decided: two milestones following the step dependencies. **v1-preview** (~2026-11-11): 4.1–4.3, 4.6, 4.9a–d, first README, public repo. **v1** (~2026-11-18): 4.4 + 4.5, 4.8, rest of Phase 6. A single one-week deadline for everything was judged too tight.
- 4.4 + 4.5 are done together, and 4.5 (paged KV) is must-do because it links to prior allocator experience. 4.8 (vLLM / llama.cpp comparison) is must-do for credibility.
- Deferred, not dropped: Phase 5 ablation, and new Phase 7 future research (Weight-Space Programming replication, experiments from the research report).
- Docs updated (additions only): DECISIONS D-021; PLAN (`[>]` status key, Phase 4 priority, Phase 5 deferral note, 6.5 publish, 6.6 self-check, Phase 7, revised timeline); PARKING_LOT (D-021 section); RESEARCH ("Future directions"); PRIOR_ART (423 GB/s update, positioning update, PA-2); CLAUDE.md (goal 3 deferral note).
- Next: 4.1. Read docs/INFERENCE.md, then benchmark naive generation with `checkpoints/main/best.pt`.

### 2026-11-03 — Step 3.7 main run · Phase 3 complete
- Run: `--preset main --max_steps 7600 --run_name main-3.7`, D-019 defaults, compiled (D-020). Logged online: [run `ugd124lz`](https://forge.coreweave.com/wandb/cooldip00-personal/tinyllm/runs/ugd124lz). Log in `checkpoints/main/train.log`.
- 996M tokens (~2.1 passes over the 465M-token train set) in **2 h 27 min** (14:40 → 17:07 UTC), including 39 evals and samples.
- Val loss:

  | Step | 0 | 200 | 400 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 7599 |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | Val | 9.009 | 2.431 | 1.917 | 1.617 | 1.484 | 1.417 | 1.369 | 1.333 | 1.304 | 1.287 | **1.2825** |

  - **Plateaued:** −0.0026 over the last 400 steps, −0.0006 over the last 200, as the LR reached its floor.
  - Final train loss 1.280 ≈ val 1.2825, so there's no overfitting despite ~2 epochs.
  - Main vs smoke at the same step 1999: 1.484 vs 1.542. The bigger model is ahead from early on, and its full run ends 0.26 lower than smoke.
  - This loss isn't directly comparable to llama2.c's numbers, because loss per token depends on the tokenizer (ours is BPE 8192, theirs Llama 32K).
- Throughput: **115.3K tok/s mean** (111.6–128.3K), **56.1% MFU**, ~23.6 TFLOPS over the whole run. Heat cost only ~3% (vs ~8% eager).
- Samples ("Once upon a time"):
  - Step 0: random tokens.
  - By step 1000: coherent multi-paragraph stories with dialogue.
  - Final, step 7599: "…a little girl named Lily. She was very curious… she found a shiny rock that looked like it was magic… she saw a butterfly that was also very shiny… found a beautiful park with a big fountain…" The story is fluent and on-topic, with a few loose details ("rock" later becomes "crystal").
- Checkpoints: `ckpt.pt` 359 MB, `best.pt` 120 MB (git-ignored). `best.pt` is the model for Phase 4.
- **Phase 3 "Done when" met:** val plateaued, coherent stories, final val loss / tok/s / MFU recorded.
- Next: Phase 4.1 naive generation baseline.

### 2026-11-03 — `torch.compile` before the main run (D-020)
- `TrainConfig.compile` (default on, `--no-compile` to disable) compiles only the training step. Eval, samples and checkpoints use the eager model, so there are no recompiles and no `_orig_mod.` keys.
- First attempt failed: there was no C compiler in WSL, and Triton needs one. You installed `build-essential` (gcc 13.3, SETUP Step 16).
- Benchmark (main preset, 120 steps, eager → compiled → eager): **72.5K → 114.1K → 70.3K tok/s**. Compiled is **+57%**, MFU 35% → 55.5%. Startup costs ~20 s more.
- Loss check: the two eager runs with the same seed differ by 0.53 at step 119 (3.82 vs 4.35), because GPU runs aren't bit-reproducible and they pass the early plateau at different times. The compiled run (4.10) falls between them.
- Tests 99/99 pass, including a new CUDA test (compiled ≈ eager losses, plain checkpoint keys).
- Main-run estimate: ~2.7 h for 1B tokens, down from ~4.2 h.
- Next: 3.7 main run.

### 2026-11-03 — Step 3.6 smoke run
- W&B login done by the user. Host `api.forge.coreweave.com`, user `cooldip00` (SETUP Step 15).
- Run: `--preset smoke --run_name smoke-3.6`, default D-019 hyperparameters, 2,000 steps = 262M tokens. Online to W&B: [run `cel67ssj`](https://forge.coreweave.com/wandb/cooldip00-personal/tinyllm/runs/cel67ssj). Log in `checkpoints/smoke/train.log`.
- Wall clock **35 min** (13:40 → 14:15 UTC), including 11 evals and samples.
- Val loss:

  | Step | 0 | 200 | 400 | 800 | 1200 | 1600 | 1999 |
  |---|---|---|---|---|---|---|---|
  | Val | 9.031 | 2.646 | 2.032 | 1.753 | 1.641 | 1.573 | **1.542** |

  It was still falling slowly at the end (−0.012 over the last 200 steps), so the bigger model and more tokens in 3.7 should help.
- Throughput: steady at **124K tok/s** (122.6–136.3K), 28.6% MFU. That's ~8% below the 1-minute 3.5 numbers because of thermals. Main estimate becomes ~67K tok/s sustained.
- Samples ("Once upon a time"):
  - Step 0: random tokens.
  - By step 800: coherent short stories.
  - Final, step 1999: "…she found a shiny coin on the ground… she accidentally dropped the coin in a puddle… a kind lady saw what had happened and helped her… She learned that it's okay to cry, but it's important to be honest and kind." The grammar holds and the story has an arc, with occasional logic slips.
- Checkpoints: `ckpt.pt` 165 MB, `best.pt` 55 MB (git-ignored).
- Next: 3.7 main run.

### 2026-11-03 — Step 3.5 throughput and MFU
- `ModelConfig.flops_per_token(seq_len)` = 6·N + 12·L·d·T, the PaLM convention.
  - N is the matmul weights only (layers + LM head); embedding lookup and norms are excluded.
  - The attention term counts the full T×T matrix, so the numbers compare with published MFU.
  - Values at T = 512: main 204.5M FLOPs/token, smoke 96.7M.
- `train.py` logs `train/tflops` and `train/mfu` (vs the 42.0 TFLOPS bf16 peak, `--peak_tflops`) next to tok/s, in the terminal and to W&B.
- GPU check (60 steps per preset, plugged in; step 0 is slow from warmup and is ignored):

  | Preset | tok/s | TFLOPS | MFU |
  |---|---|---|---|
  | smoke | 134–137K | 13.0–13.3 | **31%** |
  | main | 72–75K | 14.7–15.4 | **35–37%** |

  - Throughput drifts down ~4% over a minute as the GPU heats up.
  - The earlier "~32%" for main used 6·N only. With the attention term it's ~36%.
  - Main at ~72K tok/s: 1B tokens ≈ 3.9 h.
- Tests (98/98 pass): hand-computed FLOPs/token for both presets; the attention term scales with T; GQA lowers it; logged MFU = tok/s · FLOPs/token / peak.
- Parked: `torch.compile` for more throughput (PARKING_LOT).
- Next: 3.6 smoke run, which needs `wandb login` first.

### 2026-11-03 — Step 3.4 W&B logging
- `uv add wandb` → wandb 0.31.0.
- `src/tinyllm/logger.py`: one small interface (`log` / `log_sample` / `finish`) per D-013.
  - `Logger` is the no-op (`--logger none`).
  - `WandbLogger` takes `--wandb_mode online|offline|disabled`. wandb is imported lazily, so the no-op path never needs it.
- `train.py` logs:
  - `train/loss`, `train/lr`, `train/grad_norm`, `train/tok_per_s`, `train/tokens` every `log_every` steps.
  - `val/loss` and `val/best` at each eval, plus a `samples` table (step, text).
  - Only scalars and text go to W&B, never checkpoints (Free plan is 5 GB/month).
- Resume: the W&B run ID is saved in `ckpt.pt`, so `--resume` continues the same run and the charts stay continuous. The logger is closed in a `finally` block, so a crash still closes the run.
- New CLI flags: `--logger`, `--wandb_project` (default `tinyllm`), `--wandb_mode`, `--run_name`.
- GPU check (smoke, offline, 30 steps, stopped at 20 and resumed): both parts used the same run ID `evdo3w4w`, and `ckpt.pt` stores it. Val 8.58 → 5.73 → 5.56 → (resume) **5.34**.
- Tests (97/97 pass), using a recording logger:
  - Exact steps for train and val metrics, the metric key set, `train/tokens`.
  - The run ID lands in the checkpoint.
  - `finish()` runs even when training crashes.
  - W&B `disabled` mode works with no network; an unknown backend raises.
- Next: 3.5 MFU logging.

### 2026-11-03 — Step 3.3 eval, checkpoints, samples
- `train.py` additions:
  - `evaluate()`: mean val loss over 20 fixed batches (327,680 tokens). The seed is fixed, so every eval uses the same windows and losses are comparable.
  - `save_checkpoint` / `load_checkpoint`: temp file then rename (crash-safe), loaded with `weights_only=True`.
    - `ckpt.pt` holds model + optimizer + rng state + step + best val + both configs, for resuming.
    - `best.pt` holds weights + model config only, for sampling and export.
  - Every `eval_every` steps (default 200) and at the last step: val loss, then save `ckpt.pt` (and `best.pt` if val improved), then a sample from the prompt "Once upon a time".
  - `--resume` continues from `<out_dir>/ckpt.pt`. `--stop_at N` stops early while keeping the full-length LR schedule, so the ~3.7 h main run can be split across sessions. A crash loses at most `eval_every` steps (~6 min for main).
  - Default run dir: `checkpoints/<preset>` (git-ignored).
- `src/tinyllm/generate.py`: `sample()` with temperature (0 = greedy), top-k, EOS stop and sliding context, no KV cache. It's only for training samples; the real engine comes in Phase 4.
- GPU check (smoke, 60 steps, stopped at 40 and resumed): val 8.64 → 4.67 (step 30) → 4.38 (step 39) → **4.13** (step 59).
  - The samples move from random tokens to TinyStories-like text ("Then, Lily was a good to, the best for…").
  - Checkpoint sizes: smoke `ckpt.pt` 165 MB, `best.pt` 55 MB. Main will be ~360 MB / ~120 MB.
- Tests (93/93 pass):
  - **Resume is bit-identical:** 20 straight steps equal 10 + stop + resume + 10 (CPU, exact equality).
  - Eval schedule; val loss drops; checkpoint contents; `best.pt` loads into a fresh model.
  - Sampling: greedy is deterministic, top-k=1 equals greedy, stops at EOS, slides past max_seq_len, restores train mode.

### 2026-11-03 — Step 3.2 training loop
- `src/tinyllm/train.py`:
  - `TrainConfig` dataclass; every field is also a CLI flag (`python -m tinyllm.train --preset main --max_steps 100`).
  - `lr_at`: linear warmup, then cosine to `min_lr`.
  - `make_optimizer`: fused AdamW, with decay on 2-D weights only.
  - `train_step`: gradient accumulation (loss ÷ n micro-batches), bf16 autocast on CUDA, clip at 1.0, and returns device tensors so there's no GPU sync per step.
  - `train`: the step loop, logging loss / lr / grad norm / tok/s.
- Hyperparameters recorded as **D-019**: 131,072 tokens/step (32×512 × 8), peak LR 1e-3, warmup 200, cosine to 1e-4, wd 0.1, betas (0.9, 0.95).
  - My first default had 4 accumulation steps, which is only 65,536 tokens/step. The test caught it, and it's fixed to 8.
- Short GPU runs (warmup 10):

| Preset | Steps | Loss | Throughput | ≈ TFLOPS (6·N·tok/s) | 1B tokens would take |
|---|---|---|---|---|---|
| smoke | 40 | 9.07 → 4.59 | ~131K tok/s | 10.8 | ~2.1 h |
| main | 30 | 9.09 → 5.60 | ~74K tok/s | 13.3 (~32% of peak) | **~3.7 h** |

- `tests/test_train.py`:
  - LR schedule values and monotonic decay; optimizer groups (norms not decayed, every param exactly once).
  - **Grad accumulation 2 × B equals one 2B batch** (SGD, params equal to 1e-6).
  - Clipping: the reported norm is pre-clip, and the step size equals the clip value.
  - The CPU loop learns a periodic corpus; seq_len above max raises; CUDA fused + bf16 path.
- 87/87 tests pass.

### 2026-11-03 — Step 3.1 data loader
- `tinyllm.data.get_batch(tokens, batch_size, seq_len, rng, device)`:
  - Returns random windows `x = tokens[s:s+T]`, `y = tokens[s+1:s+T+1]` as int64. Sampling is with replacement, and windows cross story boundaries (EOS marks where a story ends).
  - One vectorised fancy-index gather from the memmap. On CUDA the batch goes through pinned memory with `non_blocking=True`.
  - The seeded `np.random.Generator` makes runs reproducible. Saving `rng.bit_generator.state` in a checkpoint makes resume exact (tested).
- Throughput, train memmap to GPU: 16×512 at 6.0 ms (1.4M tok/s); 32×512 at 4.6 ms (**3.6M tok/s**).
  - Training is expected at ~70K tok/s, so the loader costs ~2% of step time and needs no prefetch thread.
- `tests/test_batch.py`: shapes/dtype, contiguous windows with y shifted by one, same seed gives the same batch, rng-state resume, both ends of the array reachable with no out-of-range starts, too-short input raises, CUDA device, real `train.bin`.
- 79/79 tests pass.

### 2026-11-03 — Step 2.6 one-batch overfit done, so Phase 2 is complete
- `scripts/overfit_batch.py`: one fixed batch of 8 × 256 random windows from `train.bin`, AdamW lr 1e-3 (no weight decay), bf16 autocast, 300 steps. Exits non-zero if the final loss is ≥ 0.05.

| Preset | Params | Start loss | Final loss | < 1.0 at step | < 0.05 at step | Time | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| smoke | 13.8M | 9.067 | **0.0033** | 118 | 153 | 9.7 s | 727 MiB |
| main | 29.9M | ~9.0 | **0.0034** | 184 | 209 | 12.3 s | 1213 MiB |

- Start loss ≈ ln(8192) = 9.011, as expected from the init.
- Phase 2 "Done when" met: all 72 tests pass and the overfit succeeds.

### 2026-11-03 — Step 2.5 whole-model tests
- Added to `tests/test_model.py`:
  - Shapes at T = 1, 7 and 512 (max).
  - Causality through all 6 layers: changing the token at position 20 leaves logits 0–19 identical.
  - `num_params()` == `estimate_params()` for smoke, main, untied, and 3 layers. Exact sizes: smoke 13,767,552, main 29,893,120.
  - Initial loss within 0.1 of ln(8192) = 9.01, so the init is near-uniform as intended.
  - Every parameter gets a non-zero gradient (nothing detached or unused).
  - bf16 autocast on CUDA: bf16 logits, fp32 finite loss ≈ 9.01.
- 72/72 tests pass.

### 2026-11-03 — Step 2.4 block + full model
- `src/tinyllm/model.py`:
  - `Block`: pre-norm, `x + self_attn(input_layernorm(x))`, then `x + mlp(post_attention_layernorm(x))`. The attention class comes from the registry via `cfg.layer_types`.
  - `Transformer`: `embed_tokens` → layers → `norm` → `lm_head`, with `lm_head` tied to `embed_tokens` by default (saves V·d = 4.2M params in main).
    - `forward(idx, targets=None)` returns `(logits, loss)`; the loss is cross-entropy computed in fp32.
    - `num_params()` counts the tied matrix once.
  - Init: Linear/Embedding weights N(0, 0.02); `o_proj` and `down_proj` N(0, 0.02/√(2·n_layers)) (smoke: 0.0058); RMSNorm weights 1.
- The attention import inside `Block.__init__` is lazy because `attention/mha.py` imports `apply_rope` from `model.py` (avoids a circular import).
- `tests/test_model.py`: block and model shapes, tied/untied head, HF Llama names, RoPE tables not in the state_dict, init std per weight group, too-long input rejected.
- 60/60 tests pass.

### 2026-11-03 — Step 2.3 attention interface + `mha`
- `src/tinyllm/attention/base.py`: `Attention` contract (`forward(x, rope, cache=None)`, `cache_bytes_per_token()`), `ATTENTION` registry, `@register` decorator (duplicates raise), `build_attention(cfg, layer_idx)` (unknown names raise `KeyError` listing the registered ones).
- `src/tinyllm/attention/mha.py`: `MHA` with HF names `q/k/v/o_proj`, no biases, RoPE on q and k, `F.scaled_dot_product_attention(is_causal=True)`.
  - Rejects `n_kv_heads != n_heads` (that is `gqa`, Step 4.9a).
  - `cache` raises `NotImplementedError` until the KV cache in Phase 4.
  - Cache cost: main = 2048 bytes per token per layer (bf16), 16 KB per token for 8 layers.
- `tests/test_attention.py`: matches a naive explicit-mask softmax oracle (atol 1e-5); changing tokens 10–15 leaves outputs 0–9 unchanged; registry, HF names, GQA rejection; bf16 autocast on CUDA within 3e-2 of fp32.
- 52/52 tests pass (CUDA test ran, nothing skipped).

### 2026-11-03 — Step 2.2 components
- `src/tinyllm/model.py`:
  - `RMSNorm`: statistic computed in fp32, output in the input dtype.
  - `RotaryEmbedding` + `apply_rope`: rotate_half (HF Llama) layout, precomputed non-persistent cos/sin, `offset` for KV-cached decode.
  - `SwiGLU`: HF names `gate_proj` / `up_proj` / `down_proj`, no biases.
- `tests/test_components.py`, checked against independent references:
  - RMSNorm matches `F.rms_norm`.
  - RoPE matches a complex-number reference, is identity at position 0, preserves norms, and gives the same q·k score for the same relative distance (pos 5/2 vs 105/102).
  - SwiGLU: formula, param count 3·d·ffn, HF names.
- 41/41 tests pass.

### 2026-11-03 — Step 2.1 ModelConfig
- `src/tinyllm/config.py`: frozen `ModelConfig` dataclass.
  - Fields: sizes, `n_kv_heads` (None = MHA), per-layer `layer_types` (None = all `mha`), norm eps, RoPE theta, init std, tied embeddings.
  - Validation: heads divide d_model; KV heads divide heads; even head_dim for RoPE; layer_types length matches n_layers.
  - Also: `estimate_params()`, dict round trip for checkpoints, and `get_config(name, **overrides)`.
- Exact param counts: smoke **13,767,552**, main **29,893,120**. They match the ARCHITECTURE.md estimates.
- `tests/test_config.py`: 29/29 tests pass in total.

### 2026-11-03 — Steps 1.3 + 1.4 done, so Phase 1 is complete
- `tinyllm.data.write_tokens` / `load_tokens`: stream-encode, story + EOS, uint16 written via `.tmp` then renamed; memmap loader.
- `scripts/tokenize_data.py` (wall 879 s, peak RAM 2.7 GB):

| Split | Stories | Tokens (incl. EOS) | File | Avg / median / p99 / max tokens per story | Stories > 512 ctx |
|---|---|---|---|---|---|
| train | 2,118,859 | **465,373,792** | 930.7 MB | 219 / 187 / 641 / 1576 | 3.05% |
| val | 21,981 | **4,677,682** | 9.4 MB | 212 / 185 / 622 / 1079 | 2.60% |

- Checks passed: one EOS per story, the file ends with EOS, max id < vocab, and the first, middle and last story decode exactly from the memmap.
- Tests: 17/17 pass (adds `tests/test_tokens.py`).
- Implications:
  - The 1B-token main run is about **2.15 epochs** of train.
  - Ctx 512 covers about 97% of stories whole. Training samples random 512-token windows across story boundaries (EOS marks the boundary).
- Possible speed-up (not needed now): encoding was single-core, ~15 min. Parallelize only if we ever re-tokenize often.

### 2026-11-03 — Step 1.2 tokenizer trained
- `src/tinyllm/tokenizer.py`: byte-level BPE with a GPT-2-style pre-tokenizer, `add_prefix_space=False` (exact roundtrip), `min_frequency=2`, all 256 bytes in the base alphabet, `<|endoftext|>` = id 0.
- `scripts/train_tokenizer.py` trained on 2,118,859 cleaned train stories in **443 s**, peak RAM **2.7 GB**.
- Saved to `tokenizer/tinystories-bpe8192.json` (554 KB, committed).
- Validation: **4,655,701 tokens**, **4.12 chars/token**, **212 tokens/story**, 0 roundtrip mismatches.
- Train-token estimate: 1.898B chars ÷ 4.12 ≈ **461M tokens**. The exact count comes in 1.3.
- The last merges learned are whole words (`Ġscooters`, `Ġknives`), so the vocabulary is saturated with real words, not junk.
- `tests/test_tokenizer.py`: vocab, EOS id, uint16 fit, exact roundtrip (including non-ASCII via bytes), common words are single tokens. 16/16 tests pass in total.

### 2026-11-03 — Step 1.1b text cleanup (D-018)
- `scripts/inspect_chars.py` showed that the non-ASCII text is mojibake. Of 130,115 non-ASCII train stories, 129,940 are affected. Top sequences: `â€œ` 270,583; `â€™` 195,201; bare `â€` ~270K (lost closing quote); dashes 15K.
- Added `src/tinyllm/data.py` with `clean_text` and `load_clean`, and made `tinyllm` an installed package (hatchling build backend).
- `tests/test_data.py`: 8 passed.
- `scripts/check_clean.py`: train 2,118,859 kept (860 dropped), val 21,981 kept (9 dropped), 0 non-ASCII left, 1,898,184,006 train characters.

### 2026-11-03 — Phase 1.1 done
- Added `datasets` 5.1.0 and `tokenizers` 0.23.3.
- `scripts/download_data.py` downloaded `roneneldan/TinyStories` into the HF cache (2.8 GB on disk).
- Train: 2,119,719 stories, 1.90B characters, average 896 characters, 230 empty stories.
- Val: 21,990 stories, 19.2M characters, none empty.
- 6.1% of train stories contain non-ASCII characters. Check what they are before training the tokenizer.
- Rough estimate: ~4 characters per token gives ~470M train tokens, so the planned 1B-token main run is about 2 epochs.

### 2026-11-03 — Phase 0.5 done, so Phase 0 is complete
- Added `scripts/check_env.py`. Ran it plugged in, in performance mode.
- bf16 GEMM n=8192: **39.7 TFLOPS** (94% of the 42.0 theoretical peak). Sustained 30 s: **38.3 TFLOPS**, 70 → 81 °C, no real throttling.
- Bandwidth: copy 410 GB/s, read 423 GB/s (rated 448).
- Training estimate revised to ~12–19 TFLOPS effective, so the main run takes ~2.6–4 h (was 4–5 h).
- **Next:** Phase 1.

### 2026-11-03 — Phase 0.3, 0.6, 0.7 done; move to `~/llm`
- Docs copied to `~/llm` and checked identical with `diff -r`. D-014 accepted.
- `git init -b main`, `.gitignore` added, local author set. Nothing committed yet.
- uv 0.13.7, Python 3.12.16, `.python-version` pinned to 3.12.
- `pyproject.toml`: torch 2.14.1 from the cu130 index (D-017), numpy, pytest (dev).
- 0.4 verified: `torch 2.14.1+cu130`, CUDA 13.0, `cuda.is_available()=True`, RTX 3070 Ti Laptop, compute capability (8, 6), bf16 supported. Triton 3.8.0 came with torch.

### 2026-11-03 — Phase 0.1–0.2 done
- Free space: C: 38.3 GB, D: 167 GB, so Ubuntu was installed on D:.
- WSL 2.6.1.0. Ubuntu 24.04.5 LTS, user `kuladip`, disk at `D:\WSL\Ubuntu-24.04`.
- `nvidia-smi` in WSL: RTX 3070 Ti Laptop, 8192 MiB, driver 610.91, CUDA 13.3 supported, 140 W power cap.
- RAM visible to WSL: 7.6 GiB (WSL's default of 50% of host RAM).

### 2026-11-03 — W&B account
- Checked W&B pricing on the official page (now CoreWeave Forge): Free = $0, 5 GB/month, public projects allowed.
- Created the account (`cooldip00`, org `cooldip00-personal`). D-013 accepted.

### 2026-11-03 — Plan update: Track B
- Reviewed a LinkedIn post about a C++/CUDA inference engine for TinyLlama 1.1B; adopted its method, not its scope.
- Added D-015 (TinyLlama 1.1B inference track) and D-016 (timeboxed Triton kernels).
- PLAN: new steps 4.9 and 4.10; `gqa` pulled forward into 4.9a; timeline extended.
- INFERENCE.md: bandwidth-ceiling method, reference-answer rule, TinyLlama and Triton results tables.

### 2026-11-03 — Planning
- Discussed feasibility, the inference-at-scale concepts and the research options.
- Chose the framework stack (DECISIONS.md D-001 to D-012).
- Host environment check:
  - GPU: NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB, driver 610.91
  - Python 3.12.6 (Windows), nvcc 12.4
  - WSL2: only the `docker-desktop` distro exists, so Ubuntu still needs installing
- Created the project docs.
- **Next:** Phase 0.1.

---

## Old Current block
_Moved here unchanged on 2026-11-09 when the top of this file was rewritten. History only: nothing here is pending unless it is also in Current._

- **Phase:** 4 — Inference engine (Phase 3 complete)
- **Step (2026-11-07):** toward **v1** (~2026-11-18): 4.10b → 4.10a → 4.10c rerun (D-032) → 6.5b → replicas (D-031) → 6.1 → 6.3 → 6.6 → 6.5c–g. _Earlier: 4.11b → 4.13 → 4.8a → 4.8b → 6.8, all done 2026-11-06._ **v1-preview reached 2026-11-05**, 6 days early: repo public with the README, CI badge passing.
- **Next action:** the vLLM + llama.cpp install spike, which was planned for the 4.9 days but not done (PLAN 4.8a). It's one hour, in separate environments outside the repo, to surface install problems early. Then 4.13. _Spike done 2026-11-05: both engines run TinyLlama and serve `/v1/completions`; project env unchanged (see the log). Next: 4.13._ **Next action (2026-11-06):** 4.11b step 1, `SlotGraphDecoder` in `cuda_graph.py` (PLAN 4.11b). _4.11b done 2026-11-06 (see the log)._ **Next action:** push the local commits (one CI run), then 4.13 step 1, the engine object with `add_request()` / `step()`. _Pushed; 4.13 step 1 done 2026-11-06._ **Next action:** 4.13 step 2, the FastAPI server with `POST /v1/completions` (adds fastapi + uvicorn, D-025). _Step 2 done 2026-11-06 (see the log)._ **Next action:** 4.13 step 3: SLOs in docs/SERVING.md first, then `scripts/load_test.py`. _Step 3 done 2026-11-06 (see the log); the vLLM run of the same load test is part of 4.8a._ **Next action:** 4.8a, TinyLlama on vLLM / llama.cpp with the same `load_test.py`. _4.8a done 2026-11-06, which also closes 4.13 (see the log)._ **Next action:** 4.8b, export the 30M model to HF Llama format, then 6.8. _4.8b done 2026-11-06 (see the log)._ **Next action:** 6.8, one-command tables. _The full plan was agreed 2026-11-06 and is saved under PLAN 6.8, with the steps, the cool-down gate, the recovered history and the lessons. Not started: build it first, then the GPU run._ _6.8 done 2026-11-06 (see the log)._ **Next action:** push the 3 local commits (one CI run), then 6.2 plots for the README (6.5a), then 6.1 / 6.3 toward **v1**. _Pushed (CI green); 6.2 done 2026-11-06 (see the log)._ **Next action:** push 6.2 and check the plots on GitHub in light and dark, then 6.1 (README: serving architecture diagram, stale Status line, demo GIF), then 6.3. _README completeness pass done 2026-11-06 (see the log; Status line fixed)._ **Next action:** push both local commits (one CI run), check the 6 SVGs on GitHub in light and dark, then 6.1 (serving architecture diagram, demo GIF), then 6.3. _Pushed 2026-11-06 (`5a0ce75`, CI green); the 6 SVGs checked on GitHub in dark mode._ **Next action:** 6.1 (serving architecture diagram, demo GIF), then 4.10b Triton INT8 GEMV (half-day timebox, judged with CUDA graphs on; D-016 amendment), then 6.3. _Order changed 2026-11-07 (D-016 second amendment, D-031; see the log). CI green for `e8a20c4`; repo pinned (6.5a done)._ **Next action:** 4.10b (a cool `tl-int8` baseline first) → 4.10a → 6.5b single Docker Space → replicas + router Space (D-031) → 6.1 diagrams → 6.3. _4.10b steps 1–5 done 2026-11-07 (kernel, bench scripts, first GPU session; stop rule passed; see the log)._ **Next action:** finish 4.10b: (1) `bench_quant` keeps only the timed variant on the GPU (int8 + int8-triton share weights, move together) + test; (2) diagnose batch 32 with fans on Max (B=32 only); (3) lm_head-on-last-position only if B=32 still spills or loses; (4) step 6 wiring: `bench_all` entries `tl-int8-gemv`, `tl-int8-graph` + RESULTS/README rendering; (5) one final session, fans on Max: `bench_all --only tl-int8-gemv,tl-int8,tl-int8-graph` (copy `v1/tinyllama/int8.json` aside first). Then 4.10a. _(1)–(2) done 2026-11-07 (`55e8595`; B=32 timed alone: bf16 2106 / int8 1080 / int8-triton 2222 tok/s = 1.05×, int8 TTFT 416 ms vs 5.4–6.8 s before; 84 °C, no hw slowdown; `runs/bench/4.10b/int8_graph_triton_b32_alone.json`, room fan overhead). The spill was the cause; (3) not needed. Known gap: int8 TTFT ~85 ms over bf16 (prefill uses the torch path)._ **Next action:** (4) step 6 wiring (outline first, 3+ files), then (5) the final session. _(4) code done 2026-11-07: `tl-int8-graph` + `tl-int8-gemv` in `results.BENCHES`, RESULTS INT8 section in three parts, README INT8 line from `tl-int8-graph`. Render-only `bench_all` fails until (5) writes the two new JSONs._ **Next action:** (5) final session, fans on Max: copy `v1/tinyllama/int8.json` aside, `bench_all --only tl-int8-gemv,tl-int8,tl-int8-graph`, show the docs diff; then INFERENCE.md 4.10b section + Triton table row, PLAN tick, commit. Then 4.10a. _4.10b done 2026-11-07 (see the log)._ **Next action:** push the local commits (one CI run), then 4.10a fused RMSNorm + residual (~1–1.5 h) → 6.5b → replicas (D-031) → 6.1 → 6.3. _Pushed 2026-11-07 (`db0e002`, CI green). 4.10a done 2026-11-07; Triton norm made the default (D-032); v1 shipping matrix and CPU-only image (D-033); see the log._ **Next action:** (1) GPU test suite on the new default (~3 min; ask first); (2) push (ask first), CI; (3) 4.10c: rerun every GPU table into `runs/bench/v1b` + the 4.8a load sweep, one session, fans on Max (ask first, ~2.5 h + sweep) → 6.5b single Docker Space (CPU image) → replicas (D-031) → 6.1 → 6.3. _GPU suite 425 passed on the new default; pushed 2026-11-07 (`3928889`, CI green). 4.10c done 2026-11-07 (see the log)._ **Next action:** push 4.10c (ask), then 6.5b single Docker Space (CPU image; the smoke test checks `/stats` reports `norm_backend: "torch"` in the container) → replicas (D-031) → 6.1 → 6.3. _Pushed (`6756629`, CI green). 6.5b steps 1–2 built locally 2026-11-07, not published (see the log)._ **Next action:** commit + push (ask; CI's new `docker` job runs), `upload_model.py --yes` (ask; public HF repo), pin `REVISION` in the Dockerfile to the upload commit, test the Hub path in the container; then 6.5b step 3 (demo + status page). _Pushed (`485e951`, CI green incl. the first `docker` job). Model uploaded 2026-11-07 (see the log)._ **Next action:** commit the `REVISION` pin (ask), then 6.5b step 3 (demo + status page). _Pushed (`dbe91b2`, CI green)._ **Next action:** 6.5b step 3, following the agreed plan saved in PLAN 6.5b → "Step 3 plan" (items 1–6; reminders for steps 4–5 there too). _Step 3 done 2026-11-07 (see the log), not committed yet._ **Next action:** commit + push step 3 (ask), then 6.5b step 4 (logs dataset, privacy notice, `analyze_logs.py`; PLAN 6.5b → "Step 4–5 reminders"). _Pushed (`6873a89`, CI green)._ **Next action:** 6.5b step 4, following the agreed plan saved in PLAN 6.5b → "Step 4 plan" (items 1–6; retention 30 days; private dataset `CoolDip00/tiny-llm-space-logs`; ask before creating it in item 5). _Step 4 done 2026-11-07, including the real HF proof (see the log); committed `4af0783`._ **Next action:** step 5, following PLAN 6.5b → "Step 5 plan" (items 0–8), **one item at a time**: report each item's details and wait for a go-ahead before the next. _Items 0–2 done 2026-11-07 (see the log); item 3 done (SLOTS 8, queue 4, rate limit 20/min burst 5); item 4 done (`space/README.md`, `push_space.py`). Committed `6eef970` (CI green). Next: item 5 (PLAN, detailed plan 5a–5d). Items 2–4 get committed together._ **Next action (2026-11-08):** 6.1 done. 2026-11-10 after 09:00 UTC: dmtest sleep check. Then 6.3 resume bullets. **Next action (2026-11-08, after 6.3):** 6.4 PARKING_LOT review; 2026-11-10 after 09:00 UTC dmtest sleep check. **Next action (2026-11-08, after 6.4):** push 6.3 + 6.4 (ask); 2026-11-10 after 09:00 UTC dmtest sleep check; then 6.5c-g, 6.6. **Next action (2026-11-09):** 6.5c done; next is the blog sub-plan for 6.5d (waiting for the user), and the user starts 6.6; schedule in the 6.4 notes.
- _Previous: 6.7 CI, first README, repo public (6.5a, without plots yet): done 2026-11-05, see the log._
- **Later (4.6):** per-channel INT8 weights for the Linear layers. Expect little speed gain on the 30M model (overhead-bound); the real test is TinyLlama in 4.9d. _Done 2026-11-04: as expected, quality Δ ≈ 0 and no speed gain here (0.78× at B=1); rerun on TinyLlama in 4.9d._
- **Blockers:** none
- **Open decisions:** none
- **Order (D-022):** 4.4 + 4.5 moved before 4.6 (4.1–4.3 finished a day early). Dates unchanged. Checkpoint 2026-11-09: if 4.9 hasn't started, cut scope instead of moving v1-preview. **D-024:** 4.9a → 4.9b → 4.9c → 4.11 (CUDA graphs) → 4.9d → README + public.
- **Scope (D-021):** **v1-preview** by about 2026-11-11 (4.1 → 4.2 → 4.3 → 4.6 → 4.9a–d, first README, public repo); **v1** by about 2026-11-18 (4.4 + 4.5 → 4.8, rest of Phase 6). 4.7 and 4.10 only if ahead. Phase 5 and Phase 7 deferred. _2026-11-07: 4.10 (both kernels) and the HF replicas are now in v1 (D-016 amendments, D-031); 4.7 stays optional._
- **Daily:** ~30 min explaining the day's work out loud (feeds 6.6).
- **Housekeeping:** push the repo to a public GitHub repo (no remote is set up yet; 20 commits so far). _Done 2026-11-04: pushed to github.com/cooldip00/tiny-llm, private for now; make it public at v1-preview (6.5a)._
- **Housekeeping:** delete the old doc copies in `D:\ResumeProject\tiny-llm` (keep `MOVED.md`). My automatic delete was blocked.
- **Reminder:** the W&B/Forge trial ends around 2026-12-02. Downgrade to Free then.
