"""ROCm-side benchmark adapter — surface stub + porting contract.

Companion to ``scripts/benchmark_adapter.py``. The NVIDIA-side adapter is the
sole importer of ``flashinfer_bench``; this file is the **ROCm-side
equivalent**, exposing the same plain-data public surface so that AKO4X's
runners, ``bench_utils``, ``pack_solution``, and cheat-check can route to a
ROCm benchmark without touching the FIB-bound NVIDIA path.

.. warning::
   **Surface parity is hand-synced** — there is no automated check that this
   module's public functions, signatures, ``STATUS_*`` constants, and
   normalized-result-dict shape stay in lockstep with
   ``scripts/benchmark_adapter.py``. When the NVIDIA adapter's public
   surface changes, this file must be updated by the same patch (or its
   very next one). The ``ROCPROF_RANGE`` rename and the omission of the
   NVIDIA-only Modal image pins are the only intentional surface deltas;
   everything else is meant to match byte-for-byte in shape. See
   ``docs/rocm-adapter-contract.md`` for the full data-contract spec.

**Status: STUB.** Every function below raises ``NotImplementedError`` with a
pointer to ``docs/rocm-adapter-contract.md`` describing what a real
implementation must do. The stub exists to:

  1. Pin the plain-data contract every ROCm function must honor, in code, so
     future implementers do not re-derive it from prose.
  2. Let downstream code import this module and dispatch to it (e.g. a future
     ``_select_adapter()`` helper that picks adapter by hardware) without
     waiting for a complete implementation.
  3. Make the surface diff against the canonical NVIDIA adapter visible at
     code-review time (signatures, constants, normalized result shape).

A working ROCm adapter is a deferred follow-up — see the contract doc for
the design decision space (which ROCm bench to wrap, dataset schema, etc.)
and the suggested implementation milestones.

Public surface (mirrors ``scripts/benchmark_adapter.py``)
---------------------------------------------------------
Discovery   : ``list_workloads(dataset_path, definition) -> [{"uuid","axes"}, ...]``
Packing     : ``pack(source_dir, build_cfg, *, name, definition, author) -> blob:str``
              ``solution_meta(blob) -> {"name","definition","author"}``
Execution   : ``run(blob, uuids, params, *, dataset_path, capture_logs=False,
              capture_autotune=False) -> normalized_result_dict``
Profiling   : ``profile(blob, uuid, opts, *, dataset_path, env_pairs=None) -> str``
              ``list_ncu_options() -> str``   [rocprof equivalent]
Sanitizer   : ``sanitize(blob, uuid, opts, *, dataset_path) -> str``
Cheat-check : ``cheat_check(blob, uuids, *, dataset_path, n_iters=4) -> dict``
Constants   : ``STATUS_*``; ``DATASET_PATH_ENV`` / ``LEGACY_DATASET_PATH_ENV``;
              ``ROCPROF_RANGE``

The solution **blob** is the ``solution.json`` text. ``params`` is the
``BenchmarkConfig`` kwargs dict. ``run`` builds a single-operator trace set
from ``uuids``, runs the engine, and flattens the result to the normalized
dict below.

Normalized result dict (``run``'s output; consumed by the benchmark-agnostic
scoring / baseline code in ``bench_utils``)::

    {definition_name: {workload_uuid: {
        "status": <str>,                  # one of STATUS_* below
        "solution": <str>,
        "axes": {<axis>: <value>, ...},
        "latency_ms": <float>,            # present when PASSED
        "reference_latency_ms": <float>,
        "speedup_factor": <float>,
        "max_abs_error": <float|"NaN">,   # present when correctness ran
        "max_rel_error": <float|"NaN">,
        "error_log": <str>,               # present for non-PASSED workloads
        "log": <str>,                     # present with capture_logs
    }}}

A ROCm port must make ``run`` yield this exact shape; the scoring
(``compute_score``) and baseline (``load_baseline`` / ``save_baseline``)
logic in ``bench_utils`` then works unchanged.
"""

# --- Status enum (mirrors the NVIDIA adapter; same plain strings) ------------
# Keep these IDENTICAL to the NVIDIA-side adapter so bench_utils' scoring code
# does not need to be parameterized on adapter identity.
STATUS_PASSED = "PASSED"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_INCORRECT_NUMERICAL = "INCORRECT_NUMERICAL"
STATUS_RUNTIME_ERROR = "RUNTIME_ERROR"
STATUS_TIMEOUT = "TIMEOUT"

# --- Dataset discovery -------------------------------------------------------
# Same env vars as the NVIDIA adapter — the dataset *path* is shared even if
# the dataset *schema* differs. spawn.py / bench_utils consult these to find
# the trace-set path (local backend); the ROCm adapter consumes the same env.
DATASET_PATH_ENV = "AKO_DATASET_PATH"
LEGACY_DATASET_PATH_ENV = "FIB_DATASET_PATH"

# --- ROCm profiling range name ----------------------------------------------
# The ROCm-side equivalent of ``NCU_NVTX_RANGE``. rocprof / rocprofv3 supports
# named regions via roctx; the profiler-rocprof skill should filter to this
# name when extracting per-kernel counters. (Stub: rename if the eventual
# rocprof integration prefers a different convention.)
ROCPROF_RANGE = "ako4x_rocm_profile"


# ===========================================================================
# Plain-data public surface (the data-contract seam)
# ===========================================================================
# Every function below takes/returns only plain data — ``str`` solution-blobs,
# ``list[str]`` workload uuids, ``dict`` params, and the normalized result
# dict. NO ROCm runtime types cross this boundary. This is what lets the
# ROCm adapter be reimplemented behind these ~8 functions without touching
# the runners or bench_utils.

_NOT_IMPLEMENTED_MSG = (
    "ROCm adapter: this function is not yet implemented. See "
    "docs/rocm-adapter-contract.md for the data contract and the suggested "
    "implementation milestones."
)


def list_workloads(dataset_path, definition):
    """Return ``[{"uuid": str, "axes": dict}, ...]`` for ``definition``, in dataset order.

    A ROCm implementation must:
      - Resolve ``definition`` (e.g. ``fp8_gemm_dsr1``) to a workload list.
      - Return a list of plain dicts with at least ``uuid`` (stable hash of the
        workload's input axes) and ``axes`` (input shape parameters).
      - Preserve dataset order: the runner uses index-based stability across
        runs and the cheat-check probes specific positions.

    The dataset schema is intentionally unspecified — see the contract doc.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def pack(source_dir, build_cfg, *, name, definition, author):
    """Pack kernel sources from ``source_dir`` into a solution-blob (solution.json text).

    ``build_cfg`` keys: ``language``, ``entry_point``, ``destination_passing_style``
    (default False), ``target_hardware`` (default ``["rocm"]`` for this adapter).
    Returns the solution.json text as a string (opaque to callers — only
    ``solution_meta`` is sanctioned to introspect it).

    A ROCm implementation must:
      - Read ``source_dir`` containing the kernel source files (``.hip``,
        ``binding.cpp``, etc. — see ``spawn.py``'s ``infer_language`` for
        the language tag → source-layout convention).
      - Produce a solution-blob whose ``solution_meta`` round-trips correctly.
      - Either define a ROCm-side Solution schema or reuse FIB's schema with
        a different ``target_hardware`` enum value.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def solution_meta(blob):
    """``{"name", "definition", "author"}`` from a solution-blob.

    The single sanctioned place that introspects a blob's internals, so callers
    can treat the blob as opaque.

    Must round-trip with ``pack``: ``solution_meta(pack(..., name=N, ...))``
    must return ``{"name": N, ...}``.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def run(blob, uuids, params, *, dataset_path, capture_logs=False, capture_autotune=False):
    """Run the benchmark engine over the workloads named by ``uuids``.

    Reconstructs the solution from ``blob``, builds a single-operator trace
    set from ``uuids``, constructs the equivalent of ``BenchmarkConfig(**params)``,
    runs the engine, and returns the normalized result dict (shape documented
    at the module docstring). When ``capture_autotune=True`` returns
    ``{"results": <dict>, "autotune_log": <str>}``.

    Hard contract for any ROCm implementation:
      - Must raise ``ValueError`` if any uuid in ``uuids`` is missing from the
        dataset, with the count + an example uuid in the message. The runner
        treats partial matches as a corruption signal; do not silently filter.
      - Must produce the normalized result shape ``{definition: {uuid: {...}}}``
        so ``bench_utils.compute_score`` can consume it unchanged.
      - ``capture_logs`` and ``capture_autotune`` are best-effort but may not
        capture C-level stderr (matches the NVIDIA adapter's caveat).

    Implementation note: the standalone driver in
    ``reference/fp8-gemm-dsr1-rocm/benchmark_fp8_gemm.py`` (PR #5 in the ROCm
    port series) is a working example of *what* a ROCm bench must do at the
    operator level; the adapter wraps that into the plain-data interface.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def profile(blob, uuid, opts, *, dataset_path, env_pairs=None):
    """rocprof-profile one workload. ROCm-side equivalent of NCU profile.

    Wraps the profiled call in a roctx range named ``ROCPROF_RANGE`` and
    filters rocprof to it (analogous to how the NVIDIA adapter filters NCU to
    ``NCU_NVTX_RANGE``).

    ``env_pairs`` lets the caller set env vars BEFORE the kernel imports
    (e.g. ``NO_GRAPH=1`` for capture-disabled profiling). Must snapshot and
    restore prior values so subsequent in-process calls don't inherit leaked
    env.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def list_ncu_options():
    """Return the profiler's option/section listing (``rocprof --list-counters`` equivalent).

    Named ``list_ncu_options`` (not ``list_rocprof_options``) so the
    profiler-skill code in ``scripts/`` doesn't need to switch on adapter
    identity. The NVIDIA adapter wraps ``ncu --list-sets``; the ROCm
    implementation should return the equivalent rocprof counter/derived-metric
    listing.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def sanitize(blob, uuid, opts, *, dataset_path):
    """compute-sanitizer-equivalent one workload.

    There is no direct compute-sanitizer on ROCm. Options for an
    implementation: AddressSanitizer-on-GPU (``-fsanitize=address`` for ROCm
    builds), ``rocgdb`` instrumented runs, or simply returning a stub
    message saying ROCm-side sanitizer is not yet wired up — provided the
    sanitizer skill's "no result" path tolerates that.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def cheat_check(blob, uuids, *, dataset_path, n_iters=4):
    """Varying-inputs correctness audit over the probe ``uuids``. Returns a plain dict.

    Mutates inputs in place across ``n_iters`` and flags kernels whose outputs
    don't change (cached / capture-stale returns). The probe-slice *selection*
    is the caller's job (benchmark-agnostic indexing); this owns only
    build + gen + run.

    Return shape (must match the NVIDIA adapter exactly so the cheat-check
    runner is benchmark-agnostic)::

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
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
