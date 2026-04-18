# Compute-Sanitizer Reference

NVIDIA `compute-sanitizer` runs four orthogonal GPU correctness checks. Use it when a solution is producing `INCORRECT_NUMERICAL` or `RUNTIME_ERROR` results and the cause isn't visible from the source — sanitizer often points at the exact misbehaving instruction.

## The Four Tools

| Tool | Catches | Common trigger |
|------|---------|----------------|
| `memcheck` (default) | Out-of-bounds global-memory access, misaligned loads/stores, device-side leaks | Bad index arithmetic; stride/offset bug; kernel launches with wrong grid; int32↔int64 overflow on pointer math |
| `initcheck` | Reads of uninitialized global or register memory | `cute.make_rmem_tensor(...)` without `.fill(0.0)`; allocated buffer never written before read; partial writes then broadcast reads |
| `racecheck` | Unsynchronized conflicting writes/reads on shared memory | Warps writing the same SMEM slot without a barrier; `T.alloc_shared` updated inside `T.Parallel` without a fence; cross-warp reductions without `__syncthreads()` |
| `synccheck` | Incorrect use of `__syncthreads` / `__syncwarp` under divergent control flow | Barriers inside `if (tid < N)` where some threads skip; early return before the sync; warp-level syncs with partial mask |

**Skip sanitizer when:**

- The kernel currently **passes** — sanitizer only helps for broken runs. If all workloads PASS but scores are low, reach for the profiler instead.
- You already know the root cause from kernel source / error message (e.g., a div-by-zero is visible without tooling).

## When to Pick Which Tool

Start with the one that matches the failure class; `--tool all` only when you have no hypothesis (it runs all four sequentially and can take 4× longer).

- **Wrong numbers, output mostly looks like garbage (NaN, Inf, wildly-off values)** → `initcheck` first. Reading uninit memory produces arbitrary bit patterns; this is the #1 cause of unreproducible `INCORRECT_NUMERICAL` on CuTe DSL / CUDA rewrites.
- **Wrong numbers, but values look *almost* right (one element off, or one row off)** → `racecheck`. Suggests two warps updating the same SMEM cell.
- **`RUNTIME_ERROR` / "illegal memory access" / "misaligned address"** → `memcheck`. Almost always an index or pointer bug.
- **Kernel hangs / deadlocks / produces partial results** → `synccheck`. Divergent barrier is the usual cause.
- **No hypothesis / reproducing flake** → `--tool all`.

## Running Sanitizer

```bash
# List workloads (same table as bench.sh / profile.sh)
bash scripts/sanitize.sh --list

# Default: memcheck on one workload
bash scripts/sanitize.sh --index 5

# Pick a tool
bash scripts/sanitize.sh --index 5 --tool initcheck
bash scripts/sanitize.sh --index 5 --tool racecheck
bash scripts/sanitize.sh --index 5 --tool synccheck

# Run all four (slower — useful when you have no hypothesis)
bash scripts/sanitize.sh --index 5 --tool all

# Batch / filter (same syntax as profile.sh)
bash scripts/sanitize.sh --index 0,3,5 --tool memcheck
bash scripts/sanitize.sh --index 2-8 --tool initcheck
bash scripts/sanitize.sh --index 0-18 --group 32-901 --tool memcheck

# Extend timeout for slow kernels (default: 300s per tool)
bash scripts/sanitize.sh --index 5 --tool all --timeout 900

# Truncate long outputs
bash scripts/sanitize.sh --index 5 --max-lines 500
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--index` | (required) | Workload indices (e.g. `5`, `0,3,5`, `2-8`) |
| `--list` | — | Print workload table with indices |
| `--tool` | `memcheck` | One of `memcheck`, `racecheck`, `initcheck`, `synccheck`, `all` |
| `--group` | (all) | Filter by group axis values (e.g. `8,16` or `32-901`) |
| `--exclude-group` | (none) | Exclude by group axis values |
| `--timeout` | `300` | Per-tool timeout in seconds |
| `--max-lines` | (unlimited) | Truncate output to N lines |

## Reading Sanitizer Output

Each tool prints a header, then one block per detected issue, then a summary:

```
========= COMPUTE-SANITIZER
========= <TOOL_NAME>: <category> detected in <kernel_name>
=========     at <PC>:<assembly or CU:line> in <kernel_name>
=========     by thread (x,y,z) in block (bx,by,bz)
=========     Address <hex> [is out of bounds | is not initialized | ...]
=========
========= Saved host backtrace up to driver entry point at <error/warning> level
... (user Python stack if enabled)
=========
========= ERROR SUMMARY: N errors
```

**What to focus on:**

- **`thread (tx,ty,tz) in block (bx,by,bz)`** — the exact lane and block that triggered it. For scalar kernels, this usually points at `threadIdx.x` in a single block.
- **`Address <hex>`** — compare to the allocated tensor's base address (via `data_ptr()`) and stride to figure out which element / row / channel is bad.
- **`is out of bounds`** — your index arithmetic went past the allocation.
- **`is not initialized`** — an initcheck hit; the source PC usually maps to an LDG that happened before the matching store retired. Fix by explicit init (`.fill(0.0)`, `torch.zeros_like`, etc.) before first read.
- **`race condition with read from thread ...`** — racecheck hit; two lanes touched the same SMEM slot without a sync between them.

**`ERROR SUMMARY: 0 errors`** = clean run for that tool. Doesn't prove there's no bug — it only proves *this tool* found nothing. Try another tool if correctness is still off.

## Interpreting Zero-Error Runs

- **memcheck clean, numbers still wrong** → the issue is not OOB / alignment. Likely algorithmic or uninit; try `initcheck`.
- **initcheck clean, numbers still wrong** → all reads touch previously-written memory; the *write* itself is wrong. Try `racecheck` (maybe two writes clobbered each other) or inspect the kernel logic directly.
- **racecheck clean, numbers still wrong** → no SMEM race; look at algorithmic bugs (wrong index formula, softmax numerics, etc.). At this point sanitizer has done its job — switch back to reading the source.
- **All four clean, bench still `INCORRECT_NUMERICAL`** → bench tolerance issue (check `rtol`/`atol` vs reference), bf16 precision loss, or a divergence in the reference implementation you're comparing against.

## Tips

- **Run sanitizer against a FAILING workload, not a passing one.** Healthy kernels produce zero-error output — not informative. Pick the exact workload UUID that hit `INCORRECT_NUMERICAL` in `bench.sh`.
- **Small workloads first.** Sanitizer is 5-50× slower than normal execution. Prefer the smallest workload that reproduces the bug (usually the `T=1` or `num_tokens=1` variant).
- **One tool at a time.** `--tool all` is convenient but doesn't parallelize; if you already have a hypothesis, pick one tool and keep iteration time low.
- **Output is auto-saved** to `sanitizer/w<index>_<tool>_<timestamp>.json` so you can diff across kernel versions.
- **Graph-captured kernels.** Same caveat as NCU: kernels inside `torch.cuda.CUDAGraph` are launched via `cuGraphLaunch`, which still surfaces to sanitizer but may have less source info in stack traces. If stack traces are unhelpful, temporarily disable graph capture to get the eager-path report.
- **Triton / TileLang kernels.** Sanitizer reports will reference PTX/SASS addresses, not the high-level Triton/TileLang source. Use the kernel name and the address offset to narrow down which tl.dot / T.Parallel / T.gemm produced the issue; the harness prints kernel names as they launch.

## Workflow: Bug Hunt

Typical triage sequence when `bench.sh` reports `INCORRECT_NUMERICAL` or `RUNTIME_ERROR` on exactly one workload:

1. Run `bash scripts/bench.sh --first 1 --index <failing_idx>` — confirm it's reproducible, not a one-off flake.
2. Pick a tool based on the failure class (see "When to Pick Which Tool" above).
3. `bash scripts/sanitize.sh --index <failing_idx> --tool <tool>` — inspect the output block.
4. Fix the kernel, re-run step 3 to confirm zero errors, then `bash scripts/bench.sh --first 1 --index <failing_idx>` to confirm the number is correct.
5. Once sanitizer + per-workload bench both pass, run the full `bash scripts/bench.sh` to verify no regression elsewhere.

## Gotchas

- **Sanitizer replays the kernel**, so any race or nondeterminism may reproduce differently than a plain run. Treat a single clean sanitizer report as a positive signal, not a guarantee.
- **initcheck false positives on atomic accumulators.** Kernels that atomic-add into a pre-zeroed buffer can look "read before write" to initcheck if the pre-zero is in a different kernel. Verify by adding an explicit `torch.zeros_like` right before the call — if it silences initcheck, the warning was a false positive.
- **racecheck false positives on benign races.** Idempotent writes (writing the same value from multiple threads) still flag as races. If you intentionally have all threads write the same LSE / scalar, those hits are expected.
- **timeout on all-tool mode.** `--tool all` runs 4 tools sequentially, each under `--timeout`. If one tool hangs, the rest won't run. Bump `--timeout 600` or more for slow kernels.
