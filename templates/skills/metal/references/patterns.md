# Optimization patterns to test

Use these to choose a hypothesis after establishing the operator contract and
checking the [hardware](hardware.md). They are candidates, not universal wins.
Evidence should explain the limiter being addressed and the new cost introduced.
For profile tools and counter interpretation, see [profiling](profiling.md).

## Elementwise and fused operations

Start with contiguous logical elements assigned to adjacent SIMD lanes. Check
the actual strides before vectorizing. A vector load needs the alignment and
in-bounds span of its full vector, not just its first element. Use a correct
tail path; a masked store cannot repair an earlier out-of-bounds vector load.

Fusion can remove intermediate traffic and dispatches but can increase live
state, change cast/FMA order and prevent independent scheduling. Measure the
small-shape dispatch cost separately from large-shape memory traffic. For a
copy-like operator, logical input/output bytes divided by time is an effective
traffic rate; cache reuse means it is not necessarily DRAM bandwidth. [M1, M7]

## RMSNorm, LayerNorm and reductions

Recover the exact formula first. `x * rsqrt(mean(x*x) + eps)` differs from
placing epsilon outside the square root; LayerNorm additionally centers the
input. Match weight/bias, accumulator precision and output cast points.

Possible decompositions are a SIMD-group per row, multiple SIMD-groups per row,
or multiple rows per group. Choose according to row width and the number of
rows, then test. Reduce lane-local partials with supported SIMD operations;
only communicate inter-SIMD partials through initialized threadgroup storage
with a valid barrier. Large rows may justify multiple passes, whose launch and
partial-buffer traffic must be included. Do not assume a different reduction
tree is bitwise identical. [M1 §§6.8, 6.10, M7]

Case coverage: widths smaller than SIMD width, nonmultiples, padded/strided
rows, few vs many rows, zero/large values, supported special values and aliases.
Welford-style accumulation is a candidate for mean/variance stability, not an
automatic numerically identical substitute for the source algorithm.

## GEMM and matrix paths

Determine transpose/layout, batch strides, accumulation precision, alpha/beta
and epilogue before choosing a tile. A useful approximate arithmetic intensity
for an unbatched GEMM is `2*M*N*K / bytes_moved`; the byte model must specify
whether operands are cached, tiled, packed or reread. It is a hypothesis model,
not a physical bandwidth measurement or a whole-inference throughput formula.

Try supported SIMD matrix or TensorOps APIs where their requirements fit.
Account for staging traffic, threadgroup memory, register accumulators and
tail tiles. Compare against a simple correct path and, where permitted by the
task, a relevant platform-library baseline. Do not import a FlashInfer contest's
library prohibition into unrelated applications. [M1 §§2.4, 6.8, 7, M8]

Split-K creates partial results and an extra reduction/atomic protocol; its
overhead and reassociation must be evaluated. Double-buffering increases live
storage and only helps when there is actual latency to hide. CUDA TMA pipelines
and warp-specialized CUTLASS schedules are not ready-made Metal implementations.

## Attention, prefill and decode

Keep query/key/value layouts, head grouping, scale, position/mask alignment,
paged-cache lookup, softmax output and cache writes explicit. Prefill and decode
often favor different decompositions; their compute/memory/dispatch balance is
a property of the shapes, cache state, precision and hardware, not a fixed rule.

For a blockwise softmax, preserve the rescaling invariant. Given the previous
maximum `m`, denominator `l`, unnormalized weighted sum `o`, and a score block
`s`, update `m' = max(m, max(s))`, `a = exp(m - m')`,
`l' = a*l + sum(exp(s - m'))` and
`o' = a*o + sum(exp(s - m') * v)`. Normalize with `o/l` at the end. This is a
mathematical description; accumulator/cast order and all-masked rows require
the source contract's handling. `-inf - -inf` is not a valid fallback for such
a row. Changing `exp` to an `exp2` formulation also changes the numeric path.

For decode, test batch-one/small-batch scheduling, long cache lengths, page
boundaries, grouped-query indexing, split reductions and cache-read reuse.
For prefill, test ragged/causal boundaries and tiling across the sequence.
Measure both kernel effects and application effects: a faster isolated GEMM
does not by itself establish improved TTFT or token throughput.

## Quantization and MoE

Treat a quantized format as an ABI: signed/unsigned code points, block layout,
group size/axis, packed bit order, zero points, scale precision and dequantization
cast order must match. Bitcast and numerical conversion are different
operations. Do not assume CUDA fp8 encodings, saturation or special values
match a newer Metal type with a similar name. [C1, M1]

Fusing dequantization with GEMM can reduce expanded-memory traffic, but repeated
unpacking/scaling and register pressure may dominate. Measure realistic batch
and expert counts and distributions. For MoE, preserve routing ties, top-k
selection, expert offsets, scatter/gather order and weighted accumulation.
Skewed routing, small experts and large experts exercise different bottlenecks.

Inference integration may change serialized KV, recurrent or draft state.
Follow the repository's cache-format/version policy. In Kapitsa, keep its
single-stream output contract, frozen logit floors and applicable speculative
decoding checks; a kernel microbenchmark is only one part of default acceptance.

## Experimental discipline

Record a concrete expectation before each change: which cost should fall,
which counter/timing should reflect it, and where it could regress. Examples:

- Better input locality should reduce the relevant memory/MMU limiter without
  losing enough parallelism to increase total duration.
- Fewer dispatches should remove CPU/GPU gaps for small shapes while preserving
  input freshness, buffer lifetime and numerics.
- A smaller tile may reduce register/scratch pressure and improve residency,
  even if it performs more loads; measure the tradeoff.

Keep failed hypotheses as observations with their shapes/device/version. A
past win, stall percentage or heuristic from another architecture is not a
guarantee on the target Mac. [M7, C2]
