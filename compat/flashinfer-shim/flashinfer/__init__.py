"""Shim for the real `flashinfer` package.

Exists solely so that `flashinfer_bench` — which eager-imports symbols from
`flashinfer.*` at package import time — can be loaded on hosts without a
CUDA toolchain (e.g. macOS driving a Modal backend).

Nothing in this shim performs real work. Any shim symbol that is actually
invoked at runtime raises `_ShimUnavailable`, so accidental local execution
fails loudly instead of silently producing wrong results.

Do NOT install this on a real Linux + NVIDIA host — it will mask the real
`flashinfer` package.
"""


class _ShimUnavailable(RuntimeError):
    pass


def _unavailable(*_args, **_kwargs):
    raise _ShimUnavailable(
        "flashinfer shim: the real flashinfer package is not installed on "
        "this host. Kernel execution must run remotely (e.g. Modal backend)."
    )
