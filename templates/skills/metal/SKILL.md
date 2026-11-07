---
name: ako-metal
description: Port CUDA, Triton or CuTe kernels to Metal and optimize native Metal kernels on Apple GPUs, including on-device inference. Covers semantic mapping, hardware capability checks, profiling and numerical validation, in an existing project or an AKO campaign. Profiling-only requests do not authorize implementation changes.
---

# CUDA to Metal and native optimization

Work from the operator's contract and measured evidence. Translate the algorithm
and its synchronization, not CUDA syntax or NVIDIA-specific performance tricks.
Read the target repository's instructions before editing or running commands.

## Choose the working context

- **Existing project:** inspect the CUDA/Triton/CuTe source or Metal kernel and
  its host call sites. Establish the task's permitted changes, target Mac,
  build, oracle and benchmark. `.ako/task.json` is not required in this mode.
- **AKO campaign:** when `.ako/task.json` exists, read its source scope, frozen
  evaluation commands and prior results. The coding session performs CPU work;
  the outer runner owns model/GPU execution and acceptance. Leave changes in
  the working tree and write hypotheses, checks and lessons to `.ako/LESSONS.md`.
  Requested profiling must use the campaign's configured profile phase.

Honor already-granted execution permission and the repository's GPU policy.
Installing, explaining or updating this skill does not itself authorize
model/GPU runs. For a profiling-only task, perform the requested measurements
within existing execution permission and leave the implementation unchanged.
Continue useful CPU work when GPU execution is unavailable; label the remaining
verification.

## Load the relevant guidance

| Task | Read |
| --- | --- |
| CUDA/Triton/CuTe source to a first Metal implementation | [Migration](references/migration.md) |
| Thread geometry, memory, SIMD, atomics or matrix operations | [Apple GPU and MSL constraints](references/hardware.md) |
| Choose an optimization for reduction, GEMM, attention, quantization or MoE | [Kernel patterns](references/patterns.md) |
| Diagnose dispatch overhead, stalls, memory or occupancy | [Profiling](references/profiling.md) |
| Compare CUDA/CPU/Metal outputs or accept a candidate | [Validation](references/validation.md) |
| Verify a hardware/API claim or its version | [Official sources](references/sources.md) |

## Working sequence

1. **Recover semantics.** Identify shapes, strides, packing, dtypes, accumulation,
   epsilon/scales/masks, output ownership and ordering. Include wrapper behavior.
   Preserve the source revision and reference inputs/outputs. Never claim CUDA
   equivalence if only a CPU approximation was actually checked.
2. **Establish the target.** Run the CPU-only `scripts/metal_doctor.py`. Query
   runtime device/pipeline limits when authorized; `assets/device_caps.mm`
   supplies a probe that submits no GPU commands. Compilation, OS, language
   version and hardware support are separate checks. A found compiler executable
   may still be unusable because its toolchain component is missing.
3. **Build a correct Metal baseline.** Make dispatch, buffer layout and
   synchronization explicit. Start with ordinary operations; add special matrix,
   asynchronous or atomic paths only after their requirements are established.
   A host build alone does not check a runtime-compiled `.metal` source.
4. **Validate, then profile.** Run small shader-validation cases before large
   workloads, then the frozen numerical contract and fresh-input tests. Use
   `scripts/compare_fixtures.py` for portable raw-tensor fixtures, or the
   project's stronger oracle. Profile the relevant execution region and inspect
   actual captured GPU work before attributing a bottleneck.
5. **Optimize one hypothesis.** Change one coherent mechanism, recheck outputs,
   and measure incumbent A vs candidate B on the same Mac with interleaved runs.
   Keep compilation, profiling and debug instrumentation out of acceptance
   timing. Keep every result, including failures and noisy measurements.
6. **Retain evidence.** Keep complete native sources, commands, device/OS/SDK,
   compiler/math flags, input identity, raw results and lessons. Preserve the
   prior working variant when a candidate fails. Use the original numerical
   reference throughout; do not permit accumulated drift between successive
   incumbents to redefine correctness.

## Constraints that change decisions

- Query `threadExecutionWidth`, pipeline `maxTotalThreadsPerThreadgroup` and
  device memory limits; CUDA's block dimensions and occupancy formulas do not
  determine an optimal Metal launch. SIMD matrix fragment layouts are opaque.
- Match the actual API generation. `MTL4CommandQueue` does not apply a resource's
  automatic hazard-tracking mode; old `MTLCommandQueue` rules cannot be copied
  blindly. MSL 4.1 atomic orderings also require version-specific handling.
- Keep numerical controls explicit. `mathMode` and
  `mathFloatingPointFunctions` are separate settings. Safe/precise settings
  still do not promise CUDA bitwise identity or preservation of subnormals.
- Use the project's GPU lock and retain it through child process cleanup.
  AKO uses `BENCH_GPU_LOCK`, with `NI_LOCK_HELD=1` / `NACE_GPU_LOCK_HELD=1` for
  children. Arrange an idle GPU; the lock cannot exclude unrelated applications.
- Do not turn a particular validation-layer bug into a device limit or a generic
  waiver. Preserve the small test and the documented kernel/OS-specific evidence.
- For Kapitsa, retain its logit floor, default-change policy, model coverage and
  continuation checks. `--reorder` requires the repository's reassociation
  evidence; `--no-gate` or a retuned floor cannot establish acceptance.
