# Native Metal inference campaigns

This branch adds `metal_campaign.py` for native Apple-silicon projects. It
reuses the Codex/Claude launch conventions, skills and cross-attempt lessons,
while snapshotting the whole Git project, including nested `.metal`, `.mm`,
headers and build definitions. The original `spawn.py` / `master` workflow
continues to target FlashInfer; do not send a Metal project through its
`solution/kernel.py` archive. The native runner has its own fixed command
contract and evidence archive, with no CUDA, PyTorch or MLX dependency.

## Start with the included operator

Requirements: Python 3.10+, Git, CMake 3.24+, Xcode Command Line Tools (SDK 15+)
and a Metal-capable Mac with macOS 15+ for the included example. Codex CLI must already be installed/authenticated for
`iterate`. No `pip install .` is needed for the native runner.

From a checkout of `codex/metal-inference`:

```bash
# Creates a separate workspace from a committed revision. No GPU/model work.
python3 metal_campaign.py init --repo . --ref HEAD \
  --config examples/metal-rmsnorm.json --output ../ako-metal-rmsnorm --agent codex

# Optional host compilation only. Metal source is compiled later, at runtime.
python3 ../ako-metal-rmsnorm/runtime/metal_campaign.py build ../ako-metal-rmsnorm

# Authorizes five bounded coding/validation/measurement attempts on an idle GPU.
python3 ../ako-metal-rmsnorm/runtime/metal_campaign.py iterate ../ako-metal-rmsnorm \
  --rounds 5 --agent-timeout 1800 --allow-gpu \
  --prompt "Optimize RMSNorm while preserving its float32 contract and dispatch interface."

python3 ../ako-metal-rmsnorm/runtime/metal_campaign.py status ../ako-metal-rmsnorm
```

The operator starts with a simple one-thread-per-row Metal implementation. Its
frozen Objective-C++ host checks against a double-precision CPU oracle with
explicit tolerances, tests output guard regions, checks command-buffer errors,
and measures GPU timestamps. Small tests include tail dimensions and different
seeds. Timing invocations also check their outputs. This is a small standalone
operator example, not an inference engine or a production speed claim.

For manual work, edit `../ako-metal-rmsnorm/workspace/` and evaluate it with:

```bash
python3 ../ako-metal-rmsnorm/runtime/metal_campaign.py bench ../ako-metal-rmsnorm \
  --candidate ../ako-metal-rmsnorm/workspace --label experiment-1 --allow-gpu
```

Each label is immutable. Reusing it fails. An unchanged candidate is recorded
as `no_source_change` without GPU work. Manual workspaces are never reset;
automatic attempts always start from the last accepted source snapshot. No
commit, branch, remote, or index in the supplied source project is modified.
The runner makes local snapshot commits only in its own campaign directories.
It never pushes, opens PRs, changes project defaults, or merges anything.

## Configuration and acceptance

Start from [examples/metal-rmsnorm.json](../examples/metal-rmsnorm.json). One
campaign has one frozen `evaluation.json`; changing it requires a new campaign.
Use a separate campaign for each operator/model/measurement contract.

| Field | Contract |
| --- | --- |
| `editable` | Explicit relative source paths/globs; changes elsewhere fail before any GPU work. |
| `build` | Nonempty list of CPU-only build commands, each an argv array; use out-of-source builds. |
| `smoke` | Small tests of the changed kernel, run on incumbent and candidate with the Metal validation layer. |
| `correctness` | Nonempty list of real output-check commands. Every exit code must be zero. |
| `benchmark.command` | Writes a new JSON file to `{output}` on each invocation. |
| `benchmark.metric` | Positive finite numeric field, e.g. `latency_ms` or `results.decode.median_tps`. |
| `benchmark.direction` | `lower` for latency, `higher` for throughput. |
| `pairs` | At least two interleaved pairs, A1 B1 A2 B2. |
| `min_speedup` | Each pair must exceed this ratio against the incumbent, greater than 1. |
| `max_relative_spread` | Reject if `(max - min) / median` exceeds this for either build. |

Benchmark JSON must include `"invalid": []`; invalid, missing, zero, NaN, or
negative metrics cannot pass. Each pair must also exceed the observed
same-build spread. The record includes every individual measurement and ratio;
a favorable median alone cannot pass. These gates are conservative evidence
checks, not a statistical confidence interval or proof of unrelated workloads.
Correctness depends on the configured oracle's coverage and exit-code contract.

Command placeholders are `{repo}`/`{build}` for the current leg,
`{baseline}`/`{baseline_build}` for the initial fixed numerical reference,
`{reference}`/`{reference_build}` for the current incumbent,
`{candidate}`/`{candidate_build}`, `{evidence}`, `{output}`, `{python}`, and
keys in `variables`. No shell expansion is performed. `env` supplies explicit
environment overrides. Paths containing spaces stay single argv entries.

The runner builds all required versions, then takes a single exclusive flock
on `BENCH_GPU_LOCK` (default `/tmp/kapitsa-gpu.lock`) for smoke tests, output
checks and the complete timing sequence. Children get `NI_LOCK_HELD=1` and
`NACE_GPU_LOCK_HELD=1` to avoid taking the same lock again. The lock coordinates
cooperating programs; arrange an idle GPU before using `--allow-gpu`. A timeout
or interrupt kills the command's process group before releasing the lock.
Timing runs with the validation layer disabled after small tests pass.

The layer is enabled by default. For the documented macOS threadgroup-memory
double-counting limitation, a frozen `shader_validation` object can set
`enabled: false` only with `static_threadgroup_bytes` above 16,384 and at most
32,768, plus a nonempty `reason`. It still runs the smoke test and records the
exception. Verify this applies to the actual kernel/OS; never use it to waive
an unexplained failure.

## Kapitsa integration

[examples/kapitsa.json](../examples/kapitsa.json) is an explicit template:
set `model`, the committed model/host-appropriate `floor`, the exact editable
kernel paths, and `kernel_smoke` before initializing it against a Kapitsa repo.
The latter is an executable accepting `REPO BUILD` that runs the changed
kernel at small sizes. There is no universal small-kernel command for Kapitsa;
provide the targeted test required by that change. The placeholder deliberately
fails instead of treating an ordinary build or a missing test as validation.
Keep that executable and the model inputs fixed for the campaign.

The template builds `kapitsa` and `logit_dump`, compares candidate outputs to
the **initial** reference using its frozen `tools/logit_compare.py` and floor,
then measures current-best vs candidate using `kapitsa bench --quick --json
--out`. It consumes `results.decode.median_tps` and rejects any top-level
`invalid` entries. Numerical-reference and incumbent roles diverge after the
first accepted attempt, preventing cumulative numerical drift from becoming a
new reference. All raw results remain available to inspect prefill, follow-up
TTFT and other regressions before a human changes a default.

Follow Kapitsa's own `AGENTS.md` and output-check policy. This single-metric
example does not automatically cover other models, server behavior, speculative
decoding, temperature-0 continuations, or the extra evidence for `--reorder`.
Never lower a floor, add `--no-gate`, or call a merely passing comparison
bitwise-identical. No Kapitsa code or benchmark data is changed by this port.

## Evidence and CPU verification

`rounds/<label>/candidate/` is the complete source snapshot;
`rounds/<label>/evidence/` contains the configuration, OS/device information,
all argv/exit/timeout records, stdout/stderr, raw measurements, a binary-capable
patch, result and agent lessons/events. Coding sessions work in a separate
`rounds/<label>/work/`; evaluation uses a clean checkout of the source snapshot
so ignored build caches cannot enter the measured candidate. Git ignores build
products and models when taking source snapshots. Submodules and source symlinks
outside the repository are currently unsupported. Local
build directories and large logit dumps are retained and can consume disk.

`history.jsonl` is append-only. `state.json` changes its best pointer only after
all checks pass and source snapshots/configuration are still unchanged. A new
attempt reads previous results/lessons and starts from this best pointer.
Attempts are bounded by `--rounds` and per-agent/per-command timeouts. The
native workflow implements fixed-evaluator iteration; it does not implement
Mode-3 evaluator co-evolution or a scheduled daily job.

`bench` exits 0 only for an accepted candidate, and 1 for rejection/evaluation
failure. `iterate` exits 0 when the bounded loop completes without execution
errors, even if no candidate improved; inspect each result's `accepted` and the
best pointer. Invalid invocation/configuration exits 2; interruption exits 130.

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q agent_runtime.py metal_campaign.py metal_runtime.py tests
cmake -S examples/metal-rmsnorm -B /tmp/ako-metal-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/ako-metal-build --parallel 4
```

Tests use fake agent/benchmark processes and temporary lock files, including
real subprocess launch, best-variant inheritance, complete native source
archival, gate failures, missing session events, timeout cleanup and interrupts.
They do not load a model or submit GPU work. Host compilation does not verify
the Metal kernel. Real Codex optimization sessions and GPU runs remain to be
validated on the intended Mac; report that limitation when reviewing the port.
