# Profiling native Metal kernels

Use diagnostics to distinguish CPU encoding/compilation, GPU scheduling,
memory behavior and arithmetic. A trace is evidence only after confirming
the intended dispatches and workload are present. Source IDs: [sources](sources.md).

## Tool availability before a capture

Run `python3 scripts/metal_doctor.py` relative to this skill directory. It
records OS/tool versions, the selected developer directory, SDK, whether the
Metal compiler actually starts, and available Instruments templates. It neither
downloads tools nor changes `xcode-select`, privacy settings or global flags.

Command Line Tools and a full Xcode installation are different environments.
Use an explicitly selected installed Xcode via a per-command `DEVELOPER_DIR`
when needed. `xcrun --find metal` may locate a stub that fails because the Metal
Toolchain component is missing; a successful version probe is a separate check.
Report absence instead of presenting a C++ build as a Metal compiler check.

The optional doctor `--compile-probe DIR` builds `assets/device_caps.mm` with
Foundation/Metal. Running that executable reports device capabilities; passing
`--source FILE --function NAME --language 3.1` also compiles a pipeline and
reports its execution width, thread limit and static threadgroup bytes. It
submits no GPU commands, but device access and driver compilation still follow
the project's execution policy. Query the actual optimized function, not an
unrelated empty kernel with different resource usage.

## Select the tool for the question

| Question | Tool/evidence | Avoid concluding |
| --- | --- | --- |
| CPU encode/compile/wait cost or GPU gaps? | Instruments Metal System Trace and relevant CPU timelines | A CPU wall-clock interval is purely GPU execution time. |
| Wrong bindings, sizes, order or missing dispatch? | Xcode Metal GPU capture/debugger, resource and encoder inspection | A nonempty capture proves the intended kernel ran correctly. |
| Register pressure, occupancy, source hotspots? | Xcode shader profiler/compiler statistics and available GPU counters | Maximum theoretical occupancy equals achieved occupancy or optimal speed. |
| Memory, cache or MMU limiter? | Supported device counters and a representative workload trace | One stall percentage gives a universal speedup prediction. |
| Reproducible custom per-pass measurements? | `counterSets`, supported sampling points and counter sample buffers | Every Apple GPU exposes the same counters or sampling granularity. |

Metal counter sample buffers and Metal 4 performance counter heaps belong to
different API paths. Query the selected device, follow that API's resolve and
completion protocol, and interpret units/sampling points from its documentation.
Do not hard-code counter indices from another machine. [M9]

## Instruments capture

First obtain real command support with `xcrun xctrace help record` and
`xcrun xctrace list templates`. Choose the observed template name. On the
Xcode version checked for this skill, the CLI supports:

```bash
xcrun xctrace record --template 'Metal System Trace' \
  --time-limit 10s --output /path/to/new-run.trace \
  --launch -- /absolute/path/to/benchmark ARGUMENTS
```

This command **runs the target**. Use it only when that GPU/model run is already
authorized, under the repository's exclusive-lock and process-lifetime protocol.
The benchmark must last long enough to capture the intended region; a model's
initial loading/compilation may consume the entire trace interval. In an AKO
campaign, use configured profile commands rather than starting a competing run.

When a parent already holds Kapitsa's GPU lock, the launched target must receive
`BENCH_GPU_LOCK`, `NI_LOCK_HELD=1` and `NACE_GPU_LOCK_HELD=1`; `xctrace record`
has `--env KEY=VALUE` for explicitly passing launch environment variables.
Do not blindly inherit a "lock held" flag without an actual holder. Do not
assume a tracing time limit proves its launched application and helper processes
have terminated. Verify their lifecycle before releasing the lock.

For inspection, use `xcrun xctrace help export` / `export --input RUN.trace
--toc` and select schemas actually present in that trace. Schema names and
counter layouts vary. Privacy/tool failures, empty GPU regions and wrong target
PIDs are incomplete captures, not zero-duration successful kernels. Do not
silently change system permissions or suppress permission prompts. [T1, M7]

## Programmatic GPU capture

Use `MTLCaptureManager` with `MTLCaptureDescriptor` for a bounded region in a
host that supports it. Check `supportsDestination(MTLCaptureDestinationGPUTraceDocument)`,
set the capture object and output URL, check the start error, encode the intended
work, and call `stopCapture` on every exit path. Enable capture support in the
app with `MetalCaptureEnabled = YES` in its Info.plist, or on macOS 14+ with
`MTL_CAPTURE_ENABLED=1` in the launch environment. Even enabling capture has
a measurable CPU cost. Create the intended command buffers after starting
capture and commit them before stopping it. Inspect the resulting `.gputrace`
in Xcode. [M10]

The AKO RMSNorm host has `--capture NEW.gputrace`. Its profile run launches
with `MTL_CAPTURE_ENABLED=1`; a missing/unsupported capture destination fails.
Captured timing is marked invalid for benchmark acceptance. The separate
`examples/metal-rmsnorm-profile.json` demonstrates the frozen optional `profile`
command, executed after ordinary A/B timings. The default example has no capture.
These paths require an actual Metal run; compiling them does not prove capture
works on the target device.

## Keep profiling separate from timing

Profile after basic correctness and small-size shader validation. Use a size
representative of the suspected limiter, and label the region/kernel so it can
be found in the capture. Start by checking dispatch count, bindings, actual
dimensions and intended input changes; fix an invalid workload before tuning it.

Use counters to test a hypothesis, then rerun ordinary unprofiled benchmarks.
GPU capture, validation, counter instrumentation and replay can perturb timing.
Command-buffer GPU timestamps cover that command buffer's interval, not a
universal per-kernel active-cycle metric; CPU wall-clock additionally includes
encoding, waiting and other overhead. State exactly which interval is reported.

Use the smallest sufficient synchronization scope. Minimize unjustified CPU
waits and missing-work gaps, but do not remove dependencies merely to make a
timeline busier. More concurrent work, larger tiles or higher occupancy are
not objectives by themselves; valid application performance is the objective.
Retain all traces and ordinary A/B results with the Mac/OS/SDK and commits. [M6, M7]
