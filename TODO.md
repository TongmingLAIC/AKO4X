# TODO — Deferred Harness Improvements

Friction points surfaced during multiple kernel optimization sessions (DSA
sparse attention, DSA TopK indexer). **Purpose-only** — a future agent
picking up any of these should design the concrete approach themselves.
Keep this file short: one bullet per item, one line of problem + one line
of intent.

## Agent onboarding

- **Spawned children don't inherit later parent fixes.** A child env spawned
  before a harness fix (e.g. a `run_modal_profile.py` fix) keeps the old
  `scripts/` until re-spawned. **Intent:** a sync path
  (e.g. `spawn.py --sync-scripts <child>`) or an explicit workflow note.
- **ITERATIONS.md is slow to scan for proven-dead paths.** After 50+ iters
  the file is 200+ lines of mixed summary / details / rejections / sweeps;
  new agents waste iterations re-trying known dead ends. **Intent:** make
  dead-ends and "IMPROVED-only" rows directly accessible without reading
  the whole log — either a standalone `DEAD_ENDS.md` short-list, or a
  `scripts/history.sh` that parses the summary table into compact views
  (timeline, dead-ends, IMPROVED only, grep), or both.

## Iteration cost

- **Noise-aware bench reporting.** Modal bench takes 3–5 min with ±1.5%
  run-to-run noise, so candidates need ≥2 runs to distinguish signal from
  jitter. **Intent:** surface per-workload std from the existing 5-trial
  data (don't raise `num_trials`), and have `diff.sh` label small deltas
  as "within noise, not signal" so agents stop re-benching marginal
  changes.
- **Local compile pre-check.** CUDA/Python typos only surface after Modal
  image boot (~30s each). ~80% are pure syntax issues catchable without
  a GPU. **Intent:** a cheap `scripts/check.sh` doing Python AST parse,
  CUDA `-fsyntax-only` on extracted inline sources, and a dry `import`,
  so most dumb compile errors are caught locally.

## Housekeeping

- **Trajectory folder clutter.** After a long session `trajectory/` holds
  100+ folders of which 90% are unlabeled smoke tests / vc reruns; labeled
  iterations are buried. **Intent:** separate labeled vs smoke snapshots
  (e.g. `trajectory/labeled/`), or a `--prune` flag. Disk use is fine;
  only discoverability matters.

## Low value (listed for completeness)

- **Triton codegen introspection.** When comparing hand-CUDA MMA vs
  `tl.dot`, the Triton-emitted PTX/SASS lives in the Modal container cache
  and is not easy to retrieve. **Intent:** optional `--dump-triton` that
  tars `~/.triton/cache/` back to local trajectory. Narrow audience
  (codegen-level investigation), probably not worth doing until a session
  actually needs it.
