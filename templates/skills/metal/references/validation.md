# Numerical and performance acceptance

Validation starts with the operator contract, not a default tolerance. Sources
for numeric/synchronization rules are in [sources](sources.md), especially M1,
M4, M5 and M6. The tools here do not substitute for a repository's stronger gate.

## Freeze reference inputs and outputs

Use the original CUDA implementation when an authorized NVIDIA environment is
available. Synchronize its actual stream before exporting outputs. Otherwise
use a documented CPU oracle or existing fixtures with provenance and disclose
what was not executed. Export input bytes once and feed those same values to
the candidate; matching seed numbers across CUDA, C++ and Python generators
do not establish matching tensors.

Include shapes, strides, dtype/packing, scalar parameters, relevant environment,
source revision, input identity and producer device/backend. Run repeatability
checks on the reference when nondeterminism is expected. A baseline noise floor
or numerical budget is measured/approved independently of the candidate, never
adjusted after seeing a failure.

Exercise zero/small dimensions where supported, nonmultiples of vector/tile
width, large dimensions, noncontiguous views, alias rules, special values,
quantization extrema, attention masks/cache boundaries and repeated calls.
Refresh values while keeping shapes and storage addresses constant; then vary
addresses where allowed. Guard regions, race-sensitive repeats and shader
validation address failure modes that an ordinary output comparison can miss.

## Portable fixture comparator

`scripts/compare_fixtures.py` is a CPU-only helper for two manifests containing
packed little-endian tensors. Each manifest has this shape:

```json
{
  "version": 1,
  "case_id": "rmsnorm-rows7-cols137-seed413",
  "input_sha256": "SHA256_OF_THE_EXACT_INPUT_BUNDLE",
  "producer": {"backend": "cuda", "revision": "SOURCE_COMMIT"},
  "tensors": [
    {"name": "output", "dtype": "float32", "shape": [7, 137],
     "file": "output.bin", "sha256": "SHA256_OF_OUTPUT_BIN"}
  ]
}
```

The uppercase strings illustrate the required provenance values; supply the
actual digests and revision. Tensor payloads contain elements in logical
row-major order. Export strided tensors deliberately; this canonical form
does not by itself verify the original buffer strides, byte padding or aliases.
The input bundle hash must cover input bytes plus shape/layout/scalar parameters,
using one documented serialization shared by both exporters.

```bash
python3 scripts/compare_fixtures.py reference/manifest.json candidate/manifest.json \
  --atol APPROVED_ABSOLUTE_TOLERANCE --rtol APPROVED_RELATIVE_TOLERANCE \
  --json /path/to/comparison.json
```

The checker requires matching case/input identity, tensor names, dtype, shape,
valid hashes and exact payload lengths. It compares integer/bool tensors exactly
without conversion to float. Floating values use
`abs(candidate-reference) <= atol + rtol*abs(reference)`; the default is zero
tolerance. Default nonfinite handling rejects NaN/Inf; `--equal-nonfinite`
explicitly permits corresponding NaNs and same-sign infinities. An empty-only
comparison cannot pass. `--bitwise` additionally requires identical bytes, so
`+0` versus `-0` does not count as bitwise equal. Byte equality is reported
separately from numeric acceptance. Exit 0 passes, 1 rejects, 2 indicates an
invalid manifest/invocation. The script supports float16, bfloat16, float32 and
fixed-width integer/bool fixtures; supported fixture storage is not a claim
about which GPU operations support that dtype.

This helper measures pointwise error, not application accuracy, memory safety,
race freedom or a certified global error bound. For another metric or format,
use the appropriate frozen oracle rather than editing the checker to pass.
In a native campaign, the copied helper is at
`{skill_dir}/scripts/compare_fixtures.py`; keep reference fixtures frozen outside
`editable`. Bind the intended input hash from the trusted producer, not from
candidate-generated metadata alone.

## GPU checks and numerical claims

Run small kernel cases with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` before
large/model workloads, then the frozen output checks. Inspect command-buffer
errors and assert nonempty intended outputs before hashing or comparing them.
Successful host compilation alone says nothing about runtime shader compilation.

A known Kapitsa report describes static threadgroup-memory double counting by
the validation layer on M5 Pro/macOS 26.5.2 (build 25F84). This is a **local
reported tooling limitation**, not the Metal memory-cap definition and not
independently reproduced by this skill update. The existing AKO exception
requires a reason and static bytes in (16,384, 32,768]; use it only when the
target project's actual kernel/device/OS evidence justifies that exception.
Still run the small test, retain compiler/runtime allocation diagnostics and
report the skipped layer. Never turn an unexplained validation failure into a
waiver. Query the real device limit. [M1, M2, K1]

Safe/precise math settings are a starting point, not a CUDA-equivalence proof.
Accumulation order, reassociation, cast placement, FMA, subnormals and math
function implementations can change outputs. State whether a change preserves
bytes, falls within an approved numeric budget, or deliberately changes the
contract. Do not label those outcomes interchangeably.

## Performance acceptance and inference integration

Compare current Metal incumbent A with candidate B on the same idle Mac,
interleaved A B A B or more pairs. Fix work, build flags, cache/warmup conditions,
power/thermal conditions and the timing interval. Exclude compilation and
profiling from ordinary kernel timing; separately measure cold compilation or
first-token cost if those are the user's target. Keep every run and flag
contention, failed commands and missing/invalid/nonpositive measurements.

AKO requires every pair to exceed the configured minimum speedup and observed
same-build spread, and bounds same-build spread. This conservative policy is
not a statistical confidence interval. It does not justify comparing NVIDIA
CUDA latency directly to Apple GPU latency as a same-device optimization.

Microkernel success does not prove inference success. Where applicable, check
temperature-0 continuations, logits, prefill/TTFT, decode throughput, cache
updates, batching, speculative paths and memory usage. Kapitsa's frozen logit
floor, `--reorder` exception evidence, model coverage, cache-format versioning
and default-change rules remain authoritative. Neither a matching short output
nor a faster isolated GEMM waives those checks. [K1]
