# Apple GPU and Metal constraints

Use [sources](sources.md) for normative definitions. A language rule, an API's
availability, an advertised GPU-family feature and a measured optimization are
different kinds of evidence. Compiler acceptance does not prove device support,
and family support does not prove that a particular tile is efficient.

## Establish the actual target

Record Mac/device, OS **build**, Xcode/SDK, selected Metal language standard,
math flags, and workload. `scripts/metal_doctor.py` checks host tooling without
creating a Metal device or submitting work. Its optional compiled probe uses
`assets/device_caps.mm` for runtime device and optional pipeline queries; the
probe itself submits no command buffers. Honor the project's policy for device
access and driver compilation before running it.

| Question | Query or evidence |
| --- | --- |
| Which hardware family? | `MTLDevice.supportsFamily`, not an inferred CUDA compute capability. |
| Shared CPU/GPU physical memory? | `hasUnifiedMemory`; storage/access synchronization remains separate. |
| Per-dimension launch bounds? | Device `maxThreadsPerThreadgroup`. |
| Total threads for this compiled kernel? | Pipeline `maxTotalThreadsPerThreadgroup`. |
| SIMD width of this pipeline? | Pipeline `threadExecutionWidth`. |
| Static threadgroup allocation? | Pipeline `staticThreadgroupMemoryLength`. |
| Total threadgroup memory cap? | Device `maxThreadgroupMemoryLength`. |
| Counter and sampling support? | Device `counterSets` and `supportsCounterSampling` for the required point. |

Apple's May 2026 feature table maps M1→Apple7, M2→Apple8, M3/M4→Apple9 and
M5→Apple10. Metal 4's API programming model is listed from Apple7; it is not
synonymous with M5-only hardware. Newer matrix/tensor APIs do not guarantee the
same physical execution path or throughput across those families. Verify the
target SDK, OS, language and each feature before using it. [M2, M8]

## Thread geometry, registers and memory

Apple documented a SIMD width of 32 for the GPUs covered by its WWDC22 talk.
Use the pipeline query in code; this observation is not permission to hard-code
all Metal devices or to equate NVIDIA SM occupancy with Apple GPU occupancy.
Threadgroup sizes that are multiples of SIMD width are performance candidates,
subject to the algorithm and pipeline limit, not an unconditional optimum. [M3, M7]

`threadGroupSizeIsMultipleOfThreadExecutionWidth` is a compiler promise about
the dispatch, not a mechanism that pads or repairs it. Do not set it unless the
actual dispatches, including partial groups, meet the promise. A pipeline
descriptor's maximum-thread setting is not CUDA's two-parameter
`__launch_bounds__`; a setting that changes register allocation needs testing.

Count live values, tile accumulators, unrolled loops and address state. Larger
tiles may improve reuse but consume registers/threadgroup memory and reduce
resident work. A thread-local array may spill; source address space alone does
not identify its physical backing. Use compiler statistics, observed occupancy
and measured timings. CUDA register-file sizes, warp-allocation granularity,
bank-count formulas and occupancy calculators are not Apple GPU specifications.

For dynamic threadgroup arguments, each host allocation length must be a
multiple of 16 bytes. Static and all dynamic allocations count toward the
device cap. Do not confuse the pipeline's static allocation with the cap.
The current family table lists 32 KiB for Apple4–Apple10, but query the device
and keep the actual allocation in the evidence. [M2, M3]

`dispatchThreads` requires nonuniform-threadgroup support and may produce
smaller final groups. `dispatchThreadgroups` describes uniform groups. Tiled
reductions must account for actual participants and initialize every scratch
value that will be read. A guarded load plus neutral value is often easier to
prove than control-flow that drops participants before a collective. [M1, M3]

## Synchronization and atomics

The memory address space and synchronization scope must both be correct:

- A threadgroup barrier synchronizes its participating group, not the grid.
  `mem_threadgroup` orders relevant threadgroup storage; communication through
  device memory needs the appropriate device-memory flag. `mem_none` provides
  execution synchronization without the memory fence. [M1 §6.10]
- SIMD collectives and SIMD-matrix operations have participation/control-flow
  requirements. Inactive-lane values and fragment layout must not be invented.
  A conditional barrier reached by some live participants is not repaired by
  all lanes having shared one CUDA warp in the original implementation.
- MSL 4.1 explicitly says ended Apple-silicon threads no longer block a barrier.
  Do not misquote this as a guarantee for divergent live threads, uninitialized
  scratch entries or every other Metal device. Keep participation uniform
  through each reduction phase when possible. [M1 §6.10.1]
- Atomicity alone does not publish unrelated non-atomic writes or guarantee
  scheduling progress. Preserve both memory ordering and the progress argument.

Version-sensitive cases require particular care:

| Feature | Constraint to verify |
| --- | --- |
| Floating-point atomic type/operations | MSL 3.0+; relevant family feature is listed from Apple7. Check the specific operation and address space. |
| 64-bit atomics | Type availability does not mean every operation works on every family. The current table lists full operations from Apple9 and a narrower Apple8/macOS min/max exception. |
| Atomic fences/thread scope | MSL documents Apple-silicon support from 3.2; inspect the exact operation and scope. |
| Acquire/release atomic orderings | MSL 4.1 adds these; do not copy them into a 3.x/4.0 target or claim that all current Metal atomics are relaxed-only. |

MSL 4.1 §6.16 retains some legacy relaxed-only wording and also explicitly
describes the new versioned overloads. Use the function-specific requirements,
target-language compiler and a device test for the chosen protocol; do not
resolve this by assuming all C++ atomics are supported. For a baseline port,
separate dependent dispatches when that yields a simpler supported proof. [M1, M2]

## Matrix and tensor paths

MSL SIMD-group matrix operations are cooperative, require uniform control flow
within the SIMD-group, and have an **unspecified** element-to-thread mapping.
The documented 8×8 types are half/float and, from MSL 3.1, bfloat. Hardware and
operation support still apply. Never transplant CUDA WMMA fragment indexing,
PTX accumulator layouts or hand-coded lane maps into these types. [M1 §§2.4, 6.8]

For TensorOps/`cooperative_tensor`, use the documented layouts and APIs for the
selected language/device. The compiler/device determine cooperative partitioning;
it is not a CUDA register array with a new spelling. Keep an ordinary supported
fallback for other targets when required. Compare precision, rounding, tiling,
tails and memory traffic as well as throughput. [M1 §7, M8]

## Host memory and command ordering

Unified memory removes a separate physical CPU/GPU memory pool on Apple GPUs;
it does not eliminate resource ownership or synchronization. `shared` permits
CPU/GPU access, `private` is GPU-only, and `memoryless` is for eligible textures,
not general buffers. Complete CPU writes before GPU consumption and GPU writes
before CPU inspection/reuse. A CPU `.contents` pointer is not a completion fence.
Use the project's buffer lifetime/synchronization protocol. [M5]

With pre-Metal-4 `MTLCommandQueue`, automatic hazard tracking has documented
conditions, including tracked resources bound directly to encoders. Heap-backed
untracked resources and indirect resource references need their own analysis.
With `MTL4CommandQueue`, `hazardTrackingMode` has **no effect**; resolve access
conflicts explicitly. Ordinary compute encoders default to serial dispatch;
Metal 4 compute passes run concurrently. Map RAW/WAR/WAW dependencies using the
smallest sufficient barrier, fence or event scope. [M6]

## Floating-point behavior

On SDK/OS versions exposing them, set both `mathMode = MTLMathModeSafe` and
`mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise` for a
conservative baseline. The older `fastMathEnabled = NO` maps to both settings;
setting only `mathMode` does not choose precise intrinsics. Keep compilation
language and contraction/reassociation decisions in the result record. [M4]

Metal implements a subset of IEEE 754. MSL permits subnormal inputs/results to
flush to zero; safe/precise settings do not override every numerical rule or
promise CUDA bitwise identity. FP16/BF16 storage, FP32 accumulation, cast/FMA
order and transcendental approximations need the actual contract's tests.
Do not assume native FP64 arithmetic or emulate it silently as FP32. [M1 §§2, 8]
