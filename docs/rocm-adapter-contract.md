# ROCm benchmark adapter — porting contract

This doc is the design surface for the deferred ROCm
`benchmark_adapter.py` implementation. The stub lives at
`scripts/benchmark_adapter_rocm.py` — every function there raises
`NotImplementedError` with a pointer back here.

## Why this is deferred

The NVIDIA-side `scripts/benchmark_adapter.py` is the **sole** importer of
[`flashinfer_bench`](https://github.com/flashinfer-ai/flashinfer-bench)
in this repo — it wraps 11 plain-data functions (`run` / `pack` /
`solution_meta` / `list_workloads` / `profile` / `list_ncu_options` /
`sanitize` / `cheat_check` plus internal helpers) so that the rest of
AKO4X never touches FIB types directly.

A complete ROCm equivalent needs answers to questions that aren't yet
resolved:

1. **Which ROCm benchmark do we wrap?** There is no canonical AMD
   equivalent of `flashinfer-bench` today. Options:
   - **AITER `op_tests/`** — per-op test scripts with reference +
     correctness + timing. Not a framework, no standardized dataset
     schema; would need substantial scaffolding.
   - **torchbench-rocm** — exists but operates at full-model granularity,
     not kernel-level.
   - **A new minimal plain-data bench** — write it specifically for AKO4X's
     needs. Smallest scope; doesn't bring a community-shared dataset.
   - **Port `flashinfer-bench` itself to ROCm** — entirely separate
     project; the dataset is NVIDIA-measured.
2. **Where do workload definitions live?** FIB uses
   `definitions/<op_type>/<op>.json` + `workloads/<op_type>/<op>.jsonl`
   under a dataset path. A ROCm port could reuse this layout (and benefit
   from `spawn.py`'s existing discovery code) or define a new schema with
   ROCm-specific axes.
3. **How is the Solution blob shaped?** FIB has a Pydantic Solution model
   with `name` / `definition` / `author` / build spec / source files. A
   ROCm Solution can either reuse this schema with a different
   `target_hardware` enum value, or define a new schema with ROCm-specific
   build metadata (e.g. `offload-arch`).

The recommended path is to resolve question 1 first — with maintainer
input on whether AKO4X should support multiple-bench multiplexing, port
FIB, or define a minimal AKO4X-native ROCm bench.

## The plain-data contract (function-by-function)

Every function listed here is in the stub. They must take/return ONLY
`str` / `list[str]` / `dict` / `bool` / `int` / `float` — no benchmark
types cross this boundary. This is the constraint that makes the seam
swappable in the first place.

### `list_workloads(dataset_path, definition) → [{"uuid", "axes"}, ...]`

Resolves an operator definition (e.g. `"fp8_gemm_dsr1"`) to its workload
list. Each entry is a plain dict with at least:
- `uuid`: stable hash of the workload's input axes (the same input shape
  must always produce the same uuid across runs and across hosts —
  the dataset-stable hashing is what lets `bench_utils` compare runs).
- `axes`: dict of input shape parameters (e.g. `{"M": 4096, "N": 7168, "K": 2048}`).

**Must preserve dataset order** — `cheat_check` probes specific indices
and `compute_score` averages in order.

### `pack(source_dir, build_cfg, *, name, definition, author) → blob:str`

Reads a directory of kernel source files (e.g. `kernel.hip` +
`binding.cpp`) and produces a solution-blob (JSON text). `build_cfg`
keys (mirroring the NVIDIA adapter):
- `language` — kernel language tag; `spawn.py`'s `infer_language` returns
  `"hip"` for `.hip` files (the spawn.py changes in this PR add `.hip`
  to the dir-inference order before `.cpp` and to the copy-allowlists,
  completing the wiring started in #3).
- `entry_point` — `.cu` uses `"binding.py::kernel"` (Python-side
  cppyy/load_inline); `.hip` uses `"binding.cpp::kernel"` (sibling
  pybind11). Both `spawn.py::infer_language` and the case study at
  `docs/case-studies/fp8-gemm-dsr1-mi355x/` agree on this convention.
- `destination_passing_style` — bool, defaults False.
- `target_hardware` — list of supported hardware tags. The NVIDIA adapter
  defaults to `["cuda"]`; ROCm adapter should default to `["rocm"]` or
  more specific (`["rocm-gfx942"]`, `["rocm-gfx950"]`).

Must round-trip with `solution_meta`.

### `solution_meta(blob) → {"name", "definition", "author"}`

Sole sanctioned blob introspection. Must round-trip with `pack`.

### `run(blob, uuids, params, *, dataset_path, capture_logs=False, capture_autotune=False) → normalized_result_dict`

The load-bearing function. Reconstructs solution + workload list, runs
the benchmark, returns the normalized result dict (shape below). Key
contract bits:

- **Hard error on missing uuids.** If any uuid in `uuids` is not found
  in the dataset for the solution's definition, raise `ValueError` with
  the count + an example. The runner treats partial matches as a
  corruption signal (stale workload list vs dataset divergence); silent
  filter would corrupt the score.
- **`capture_autotune=True`** changes the return shape to
  `{"results": <dict>, "autotune_log": <str>}`.
- **`capture_logs`** routes per-workload error logs into the result
  dict's `log` field.

#### Normalized result dict shape (must match exactly)

```python
{
  definition_name: {                       # e.g. "fp8_gemm_dsr1"
    workload_uuid: {                       # e.g. "abc12345..."
      "status": <str>,                     # one of STATUS_* (see stub constants)
      "solution": <str>,                   # solution name
      "axes": {<axis>: <value>, ...},
      "latency_ms": <float>,               # present when PASSED
      "reference_latency_ms": <float>,
      "speedup_factor": <float>,
      "max_abs_error": <float|"NaN">,      # present when correctness ran
      "max_rel_error": <float|"NaN">,
      "error_log": <str>,                  # present for non-PASSED
      "log": <str>,                        # present with capture_logs
    },
  },
}
```

The `bench_utils.compute_score` / `load_baseline` / `save_baseline` code
consumes this verbatim. **Do not deviate** — the frozen-for-comparability
contract in `bench_utils` depends on it.

### `profile(blob, uuid, opts, *, dataset_path, env_pairs=None) → str`

Profiler-rocprof-equivalent of the NVIDIA NCU profile. Wrap the profiled
call in a roctx range named by the `ROCPROF_RANGE` constant (the
NVIDIA-side analogue is `NCU_NVTX_RANGE = "flashinfer_bench_ncu_profile"`).
Returns the profiler output as plain text.

`env_pairs` must be applied BEFORE the kernel imports and restored on
exit (matches the NVIDIA adapter's order — module-level env gates that
the kernel evaluates at its own import time will see the requested
values).

### `list_ncu_options() → str`

Keep the name `list_ncu_options` (not `list_rocprof_options`) so the
profiler skill in `scripts/` doesn't switch on adapter identity. Return
the rocprof equivalent listing (something like
`rocprof --list-counters` output).

### `sanitize(blob, uuid, opts, *, dataset_path) → str`

There is no direct `compute-sanitizer` on ROCm. Options:
- AddressSanitizer-on-GPU (`-fsanitize=address` for ROCm builds).
- `rocgdb` instrumented runs.
- Or return a clear "ROCm-side sanitizer is not yet wired up" message
  and verify that the sanitizer-skill code in `scripts/` tolerates a
  no-sanitizer-available path.

### `cheat_check(blob, uuids, *, dataset_path, n_iters=4) → dict`

Varying-inputs correctness audit. Mutates inputs in place across
`n_iters`, flags kernels whose outputs don't change (cached or
capture-stale returns). Return shape MUST match the NVIDIA adapter
exactly so the cheat-check runner is benchmark-agnostic:

```python
{
  "status": "PASS" | "FAIL" | "ERROR",
  "definition": <str>,
  "n_iters": <int>,
  "workloads": {
    <wl_uuid[:8]>: {
      "axes": dict, "n_iters": int,
      "unique_hashes": int,
      "all_iters_differ": bool,
      "status": "PASS" | "FAIL" | "ERROR",
      "reason": <str>,   # present for FAIL / ERROR
    },
  },
}
```

## Suggested implementation milestones

Once question 1 is resolved, implement in this order — each milestone is
independently useful and gates a specific AKO4X feature:

1. **Milestone A — minimal `run` + `list_workloads` + `pack` +
   `solution_meta`.** Unblocks single-operator benchmarking via the
   adapter. Operator: `fp8_gemm_dsr1` (the case study in PR #5 is the
   ready-made reference for *what* the operator looks like; the adapter
   wraps it).

2. **Milestone B — `cheat_check`.** Unblocks the parent-side correctness
   audit (`scripts/cheat_check_modal.py` equivalent for ROCm — could be
   `cheat_check_local.py` if there's no Modal-equivalent yet).

3. **Milestone C — `profile` + `list_ncu_options` (rocprof).** Unblocks
   the `profiler-rocprof` skill's "what counter to look at" loop. Lowest
   priority; profiling is useful but not load-bearing for the closed
   loop.

4. **Milestone D — `sanitize`.** Optional / lowest priority. The
   sanitizer skill is the least-exercised in the existing AKO4X
   campaigns; a noop-with-clear-message may be acceptable for v1.

## Worked example: what `run(blob_for_v7_8wave, [uuid_4096_7168_2048], ...)` should do

The case study in PR #5 provides a ready-made operator that an adapter
implementation can use as a target:

- Input: a solution blob produced by `pack("reference/fp8-gemm-dsr1-rocm/variants/v7_8wave/", ...)`.
- Single uuid corresponding to (M=4096, N=7168, K=2048).
- Expected: the adapter compiles the `.hip` + `binding.cpp`, runs them
  against the same `torch_naive` reference used in the case study, times
  100 iterations after 5 warmup, and returns a normalized result dict
  with `latency_ms ≈ 0.1016` and `speedup_factor ≈ 1.21` against the
  hip_stock baseline (or whatever baseline the adapter wires up).

The standalone `reference/fp8-gemm-dsr1-rocm/benchmark_fp8_gemm.py` is
already doing all of this — the adapter's job is to expose it through
the plain-data interface.

## Relation to the rest of the ROCm port series

- **#3** added `spawn.py` GPU detection + `[rocm]` extra (made AKO4X
  tooling importable on AMD hosts).
- **#4** added `hip` + `aiter` SKILLs (gave sub agents the AMD-specific
  kernel-writing knowledge).
- **#5** added the FP8 GEMM operator archive + case study (validated
  the loop pattern works on AMD via a dataset-less driver).
- **This PR (#6)** adds the adapter surface stub + this contract doc.
  Lands no executable functionality; pins the contract for the deferred
  real port.
