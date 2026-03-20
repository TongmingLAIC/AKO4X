## Language Reference

| Language | Files in `solution/` | `config.toml` language | `config.toml` entry_point |
|----------|---------------------|----------------------|--------------------------|
| Python / PyTorch | `kernel.py` | `python` | `kernel.py::run` |
| Triton | `kernel.py` | `triton` | `kernel.py::run` |
| TileLang | `kernel.py` | `tilelang` | `kernel.py::run` |
| CuTe DSL | `kernel.py` | `python` | `kernel.py::run` |
| CUDA | `kernel.cu` + `binding.py` | `cuda` | `binding.py::kernel` |
| C++ | `kernel.cpp` + `binding.py` | `cpp` | `binding.py::kernel` |

When you choose a language, read the corresponding guide in `docs/languages/` for patterns, code examples, and constraints.
