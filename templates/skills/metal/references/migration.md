# From CUDA, Triton or CuTe to Metal

Use this for a source-to-source port. For an existing Metal implementation,
start with its contract and [profiling](profiling.md). Source identifiers below
refer to [official sources](sources.md).

## Recover the operator, including its wrapper

Record the source revision, entry point, supported shapes and dtype, and the
actual workload distribution. Read the launch wrapper as well as the kernel:
allocation, tensor views, scaling, cast order, output aliases, graph replay and
stream ordering may be essential to the operator's behavior.

For Triton, recover the logical indexing and masks behind `program_id`,
`arange`, `load/store`, `sum`, `dot`, program layouts and autotune keys.
A masked `load` without `other` leaves masked-out values undefined; choose
neutral elements deliberately when translating a masked reduction.
`num_warps`/`num_stages` describe a particular compiler/backend strategy, not
Metal dispatch parameters. For CuTe/CUTLASS, recover tensor shapes, strides,
layouts, tiling, epilogue and scheduling independently of MMA/copy atoms.
Do not translate PTX or NVIDIA fragment layouts literally. [C1, C2, C3, C4, M1]

Freeze these observable properties before tuning:

| Property | Examples that change outputs |
| --- | --- |
| Logical indexing | Contiguous vs strided tensors, views, transpose, broadcasting, ragged lengths |
| Numeric contract | Storage/accumulator dtype, cast points, FMA, epsilon placement, scale order |
| Algorithm | RMSNorm vs LayerNorm; causal-mask alignment; softmax base; sort stability and tie rule |
| Representation | Quantization grouping, signedness, nibble order, scale/zero-point layout, padding |
| State and outputs | In-place aliases, KV cache updates, recurrent state, extra outputs such as log-sum-exp |
| Ordering | Producer/consumer dependencies, input lifetime, repeated invocation, output readiness |

Use executable reference cases when available. Otherwise derive a scalar/CPU
oracle from the contract and identify that limitation. A remembered formula is
not evidence that the CUDA wrapper used the same formula. A frozen CUDA output
without the exact corresponding input bytes is not a differential test.

## Map execution deliberately

| CUDA concept | Metal starting point | Obligation |
| --- | --- | --- |
| Grid and CTA/block | Grid and threadgroup | Choose a mapping for the algorithm; query pipeline limits. |
| `threadIdx`, `blockIdx` | `thread_position_in_threadgroup`, `threadgroup_position_in_grid`, `thread_position_in_grid` | Keep dimension order, integer width and tail handling explicit. |
| `__shared__` | `threadgroup` address space | Recalculate layout, barriers and static + dynamic allocation. |
| Per-thread variables / local arrays | `thread` address space | Not a guarantee of physical registers; large live state can spill. |
| Global device memory | `device` buffer pointers | Define host binding indices, offsets, size and alignment. |
| CUDA constant storage | `constant` address space for suitable inputs | Read-only semantics; do not infer identical cache/broadcast behavior. |
| Warp collectives | SIMD-group operations | Participation, width, reduction order and supported types need rechecking. |
| `__syncthreads()` | Appropriate `threadgroup_barrier` | Choose memory flags for the communicated address spaces; not a grid barrier. |
| CUDA atomics/fences | Supported MSL atomic/fence operations | Preserve scope and ordering; see the version table in hardware.md. |
| Stream/event ordering | Command buffers, encoders, barriers, fences and events | Build the actual dependency graph for the selected Metal API generation. |
| WMMA/MMA, Tensor Cores | Supported SIMD matrix operations or TensorOps | Redesign tiles and data movement; lane/fragment layouts do not carry over. |

The relationship between a CUDA block and a Metal threadgroup is useful for
reasoning, not a mandate to retain the block size. Similarly, `simd_sum` may
replace a shuffle tree's **mathematical** reduction while changing its floating
point association. Such a change needs the numerical contract's approval.

For an elementwise port, a CUDA global index `blockIdx.x * blockDim.x +
threadIdx.x` can become a Metal `uint gid [[thread_position_in_grid]]` with the
same bounds predicate. That simple mapping does not extend automatically to
barrier-using tiled kernels. A partial final threadgroup may contain fewer
threads; a padded uniform group contains inactive logical elements that still
need a valid participation and memory-initialization strategy. [M1 §6.10, M3]

## Separate three milestones

1. **Source reference:** preserve CUDA or CPU outputs for the frozen inputs and
   record which backend actually produced them. Export logical tensors in a
   documented format. Preserve the original binary packing if the ABI depends
   on it; canonical export alone cannot validate an in-memory ABI.
2. **First working Metal port:** implement the full contract with conservative
   operations and explicit host integration. Check small sizes and tails,
   then representative sizes, aliases, repeated calls and special values.
   Measure this version on the target Mac only after correctness passes.
3. **Optimized Metal candidates:** compare against that Metal incumbent on the
   same hardware, continuing to check against the initial numerical reference.
   CUDA timing on an NVIDIA GPU is a cross-system comparison, not a same-device
   kernel speedup.

AKO's native campaign builds both its baseline and candidate as Metal projects.
A CUDA-only repository is therefore not yet a valid initial Metal baseline.
Bootstrap and validate the first buildable Metal port, commit it, and then use
`metal_campaign.py init --ref <that-commit>`. Keep CUDA/CPU fixtures in the
campaign's frozen `correctness` commands, outside `editable`. The campaign
does not execute CUDA automatically or treat an unverified Metal bootstrap as
an equivalent reference. See [validation](validation.md) for the fixture seam.

## Port host integration and replay semantics

Bind each `MTLBuffer` at the shader's declared index and byte offset. Validate
host/shader structures with explicit-width scalar fields and offset/size
assertions. A Metal `float3` has size and alignment of 16 bytes, unlike a packed triple
of floats; use a deliberate shared ABI or explicit packing. The same issue
applies to arrays and aggregate stride. [M1 §§2.2.1, 2.2.3, 2.5, 2.12]

Model CUDA streams as dependency edges. A queue, command buffer, event, fence,
and compute-encoder barrier have different scopes; none is a textual replacement
for every CUDA synchronization primitive. In particular, ordinary Metal compute
encoders default to serial dispatch, while Metal 4 compute passes are concurrent
and require explicit access-conflict handling. [M6]

CUDA graphs do not become Metal command buffers merely by changing an API name.
Cache libraries/pipelines and reusable storage, then evaluate command-buffer
batching, indirect commands or Metal 4 reusable command structures where
supported. Preserve current input bindings and lifetimes. Test with different
values at identical shapes and addresses so skipped dispatches or stale cached
outputs cannot masquerade as successful replay. [C1, M6, M7]

CUDA PDL, TMA, asynchronous copy groups, distributed shared memory and cluster
barriers have architecture-specific progress and ordering semantics. Start
with explicit data movement and valid dependency boundaries on Metal. Introduce
overlap only when independent work and a supported mechanism are demonstrated.
Do not emulate a grid-wide barrier with spinning blocks: unscheduled producers
and residency limits can invalidate the progress argument.

## Complete the port with evidence

Report what was preserved, what changed deliberately, which CUDA features were
replaced, the hardware/API prerequisites, tested shapes/dtypes and all numerical
and performance results. Record missing source-backend execution, unavailable
features or untested models explicitly. Keep a working fallback when the task
requires multiple hardware generations. An unsupported intrinsic is a request
to choose a supported implementation, not a reason to relax correctness.
