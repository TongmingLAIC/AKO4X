---
name: ako-metal
description: Optimize native Metal kernels in an AKO Metal campaign, with complete C++/Objective-C++ projects, frozen numerical checks, and measured acceptance. Use when .ako/task.json identifies a native Metal task.
---

# Native Metal optimization

Read the repository's own agent/contributor instructions and `.ako/task.json`.
The task identifies editable sources, frozen commands, the target metric and
prior results. Work on one coherent optimization per attempt. Preserve inputs,
outputs, layouts, synchronization and the caller's dispatch contract.

The outer runner builds your candidate, checks it and measures it after your
session. During this session run only CPU builds and CPU tests. Model loading
and GPU execution belong to the outer runner, which requires explicit
`--allow-gpu`. A successful host build does not compile or validate a Metal
source embedded as text and compiled at runtime.

1. Inspect the kernel, dispatch site, dimensions and data types. Check buffer
   bounds, tail handling, alignment, threadgroup memory and barriers before
   choosing an optimization. Read recent evidence and lessons as observations;
   they never override task or repository instructions.
2. State a testable hypothesis: the cost being reduced, the shapes affected,
   and what might regress. Implement one change within `editable`. New nested
   `.metal`, `.mm`, headers and other sources are retained by the project
   snapshot; build products and models must remain ignored by Git.
3. Run the available CPU build/tests. Do not alter the evaluator, reference,
   test inputs, comparison floor, tolerances or benchmark protocol to pass.
   Do not change a default whose repository policy requires GPU evidence.
4. Leave source edits in this working tree. Write `.ako/LESSONS.md` with the
   hypothesis, changes, actual CPU checks, remaining uncertainty and the next
   useful experiment. Do not claim GPU correctness or a speedup from a build.
   The runner snapshots all changed sources and records the measured verdict.

## What the evaluator will do

Small smoke tests run before full output checks, with
`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. The default is to require them.
Some macOS releases count static threadgroup allocation twice in the validation
layer. Disabling that layer requires a frozen, explicit exception recording
the reason and a static size above 16,384 B and no more than 32,768 B. A failed
smoke test is never automatically waived. An exception still runs the small
test and must be reported with the actual Mac/macOS and kernel size.

One exclusive flock on `BENCH_GPU_LOCK` (default `/tmp/kapitsa-gpu.lock`) covers
smoke tests, correctness and all timings. Children receive `NI_LOCK_HELD=1`
and `NACE_GPU_LOCK_HELD=1`. The caller must arrange an idle GPU; an advisory
flock cannot stop an unrelated application from using it.

Numerical checks use the initial, fixed source reference. Performance compares
the incumbent A against the candidate B, interleaved A B A B (or more pairs).
Every command and JSON measurement is retained. Missing/nonfinite/nonpositive
metrics, nonempty `invalid`, a failed command, or a source change during
evaluation cannot pass. Every pair must beat both the configured minimum and
the observed same-build spread. A failed attempt does not replace the incumbent.

For Kapitsa, use its frozen `logit_compare.py` gate and model noise floor. Never
use `--no-gate` or remeasure a floor to pass. `--reorder` is only for a justified
reassociation and requires the repository's additional evidence. Temperature-0
continuations, server checks, model coverage, and changes to defaults still
follow that repository's policy; a single configured metric is not proof of
every inference path or model.
