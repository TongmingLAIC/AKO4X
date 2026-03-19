"""
TVM FFI Bindings Template for C++ Kernels.

This file provides Python bindings for your C++ kernel using TVM FFI.
The entry point function name should match the `entry_point` setting in config.toml.

See docs/definition.json for the operator specification and reference implementation.
"""

import ctypes
from tvm.ffi import register_func


@register_func("flashinfer.kernel")
def kernel():
    """
    Python binding for your C++ kernel.

    TODO: Implement the binding according to docs/definition.json.
    This function should:
    1. Accept the inputs as specified in docs/definition.json
    2. Call your C++ implementation
    3. Return outputs as specified in docs/definition.json
    """
    pass
