# Sources and verification scope

Checked on **2026-10-30**. These are official documents, not performance claims
for the user's machine. Prefer the exact API/language/device requirement over
a general talk or an older example. PDFs at stable URLs can be replaced; the
versions/dates below identify the editions consulted.

| ID | Official source | What it establishes |
| --- | --- | --- |
| M1 | [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), **4.1, 2026-06-04** | §2 types/alignment/SIMD matrices, §4 address spaces, §6.8 matrix operations, §6.10 barriers/collectives, §6.16 versioned atomics, §7 TensorOps, §8 numerical compliance. |
| M2 | [Metal Feature Set Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf), **2026-05-21** | Device families, feature availability and limits; read the footnotes, including 64-bit atomics. |
| M3 | [threadExecutionWidth](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth), [pipeline thread limit](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/maxtotalthreadsperthreadgroup), [dynamic threadgroup memory](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)), [dispatchThreads](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreads(_:threadsperthreadgroup:)) | Pipeline-specific constraints, 16-byte dynamic-allocation alignment, partial groups and feature checks. |
| M4 | [MTLCompileOptions.mathMode](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathmode), [mathFloatingPointFunctions](https://developer.apple.com/documentation/metal/mtlcompileoptions/mathfloatingpointfunctions) | The two independent math controls and how legacy `fastMathEnabled` maps to both; availability from macOS 15/iOS 18. |
| M5 | [Storage modes on Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus), [Synchronizing CPU and GPU work](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work) | Unified memory still requires an access/lifetime protocol; private/shared/memoryless distinctions. |
| M6 | [Resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization), [GPU/CPU shared events](https://developer.apple.com/documentation/metal/synchronizing-events-between-a-gpu-and-the-cpu) | Metal 4 versus legacy hazard tracking, compute concurrency, barriers/fences/events and scopes. |
| M7 | [Scale compute workloads across Apple GPUs — WWDC22](https://developer.apple.com/videos/play/wwdc2022/10159/) | Profiling methodology, workload distribution, memory locality, atomics and explicit barriers for concurrent dispatch. Historical heuristics/numbers are not modern-device guarantees. |
| M8 | [Combine Metal 4 machine learning and graphics — WWDC25](https://developer.apple.com/videos/play/wwdc2025/262/) plus M1 §7 and M2 | Tensor-operation programming model; do not equate API availability with identical accelerator hardware or performance. |
| M9 | [Sampling GPU data into counter sample buffers](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers) | Counter-set/sampling support, encoding, resolution and interpretation for that API path. |
| M10 | [Capturing a Metal workload programmatically](https://developer.apple.com/documentation/xcode/capturing-a-metal-workload-programmatically), [MTLCaptureDescriptor](https://developer.apple.com/documentation/metal/mtlcapturedescriptor), [Apple sample](https://developer.apple.com/documentation/metal/capturing-metal-commands-programmatically) | Capture enablement (Info.plist or the macOS 14+ environment variable), command-buffer capture boundaries, destination support and start errors. |
| C1 | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Source-backend execution/memory model, warp collectives, independent scheduling, atomics, streams and graphs. |
| C2 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | NVIDIA memory/occupancy/profile guidance; not an Apple hardware specification. |
| C3 | Triton [load](https://triton-lang.org/main/python-api/generated/triton.language.load.html) and [Config](https://triton-lang.org/main/python-api/generated/triton.Config.html) | Masked-load behavior, including undefined values when `other` is absent, and compiler/autotuning configuration. |
| C4 | NVIDIA CUTLASS [CuTe Layout](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/01_layout.html) | Shape/stride and coordinate-to-index mapping; a source layout does not prescribe Metal scheduling. |
| K1 | [Kapitsa AGENTS.md](https://github.com/nace-ai/kapitsa/blob/main/AGENTS.md) and the checked-out target's own instructions | Project-specific output, lock, default-change and validation-exception policies. Local reported observations are not universal Metal rules. |
| T1 | Installed Xcode **26.6 (17F113)**: `xcrun xctrace help record`, `help export`, `list templates`; Metal SDK headers | CLI shape and locally exposed templates; SDK declarations/availability for capability and capture probes. Re-query on another installation. |

## Conflicting or versioned statements

- MSL 4.1 §6.16 includes legacy relaxed-only prose alongside new versioned
  ordering overloads. Read the versioned function requirements and verify the
  target compiler/device; do not flatten this into "all atomics are relaxed"
  or "all C++ atomic orders work everywhere".
- M3's dynamic-memory page links to the pipeline's static allocation when
  discussing limits. For the actual **device cap**, use
  `MTLDevice.maxThreadgroupMemoryLength`; count the pipeline static bytes plus
  dynamic allocations. This distinction was cross-checked against SDK headers.
- A family table's broad feature name does not establish every dtype, address
  space, precision, scope or performance property. Apply the language/API
  restrictions as well.
- The Kapitsa layer-doubling report was supplied by the target repository's
  instructions. This update does not reproduce that GPU behavior or certify
  that the same limitation holds on another OS build.

## What local checks can establish

CPU tests validate tooling and fixture/gate behavior. Host compilation checks
Objective-C++ declarations. Tool `--help` and template listing verify CLI
support. These do not prove shader correctness, capture contents, race freedom,
CUDA parity or speed. Report actual GPU/NVIDIA validation separately when it
has been authorized and performed. A missing Metal compiler component must
remain visible in the diagnostic result; no successful C++ build replaces it.
