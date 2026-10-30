# Native Windows and CUDA

Use 64-bit Python 3.12, Visual Studio 2022 with **Desktop development with C++**,
an NVIDIA driver, CUDA Toolkit 12.8, Git, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
Validated on 2026-10-29 with an RTX 2060 SUPER (sm_75), MSVC 14.39,
Python 3.12 and CUDA 12.8 on 64-bit Windows.
Compilation and execution use Windows executables throughout.

Clone with LF line endings: model/resource fingerprints use the exact source bytes.

```powershell
git -c core.autocrlf=false clone --recursive https://github.com/A2R-Lab/GRiD.git
cd GRiD
powershell -ExecutionPolicy Bypass -File install/developer_install.ps1
```

For the generator and NumPy runtime alone, use `install/base_install.ps1`.
The developer installer also installs a CUDA PyTorch wheel and the independent
Pinocchio 3.9 oracle from conda-forge. Its isolated prefix is `.cache/pinocchio`;
a venv-local `.pth` exposes its Python modules and registers its DLL directory.
The second-order oracle extension builds with MSVC on first use.

In each new PowerShell session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. ./install/activate_windows.ps1
python -c "import grid_rbd; print(grid_rbd.__version__)"
python -m pytest -q test/python_wrappers/test_iiwa14_torch_smoke.py
```

Activation configures the x64 Visual Studio compiler and Windows SDK, selects
the CUDA installation containing `nvcc.exe` on PATH, enables UTF-8 Python I/O,
and activates `.venv`. It changes only the current process.

Python registration also initializes the installed x64 compiler environment when
started from a regular terminal or a Jupyter/VS Code kernel. Select
`.venv/Scripts/python.exe` for the examples under `examples/notebooks/` and
restart existing kernels after updating GRiD. UTF-8 source I/O is explicit, so
the runtime does not depend on the Windows language or `PYTHONUTF8` setting.

Test lanes mirror the upstream CPU lane and the two GPU directories:

```powershell
powershell -ExecutionPolicy Bypass -File test/run_windows.ps1 -Lane cpu
powershell -ExecutionPolicy Bypass -File test/run_windows.ps1 -Lane cuda
powershell -ExecutionPolicy Bypass -File test/run_windows.ps1 -Lane wrappers
powershell -ExecutionPolicy Bypass -File test/run_windows.ps1 -Lane all
```

Cold robot builds can take several minutes. `GRID_RBD_BUILD_JOBS` controls the
parallel wrapper compilation count; set it to `2` on a memory-constrained host.
Large humanoid models can take over an hour per CUDA translation unit. The build
log records each unit when it finishes; increasing CPU time in `cicc.exe` or
`ptxas.exe` indicates that compilation is still active.
Build logs and cached objects live under the directory returned by
`grid_rbd._cache.default_cache_dir()`.

On Windows, `force_rebuild=True` compiles a fresh DLL image and updates the cache
for subsequent loads. Existing handles keep using their original image; a mapped
DLL is never deleted during a forced rebuild.

Inline CUDA callers can keep large centroidal body Jacobians in caller-owned
workspace: `com_device<T, false>`, `ccrba_device<T, false>` and
`energy_device<T, false>` take an additional final `T*` pointing to
`6 * NUM_VEL * NUM_BODIES` elements per concurrently executing block. Size the
remaining shared scratch with `CENTROIDAL_DEVICE_SHARED_MEM_BYTES<T, false>()`.
The default `true` keeps the Jacobians in shared memory. Both paths run the same
centroidal computation.

JAX's NVIDIA GPU backend is unavailable on native Windows according to the
[JAX platform matrix](https://docs.jax.dev/en/latest/installation.html#supported-platforms).
The Windows installer therefore installs NumPy and CUDA PyTorch. Tests requiring
JAX retain their existing optional-dependency behavior. This platform limitation
does not affect the generated CUDA kernels or the NumPy/PyTorch CUDA runtime.

The upstream signed Linux GPU receipt is separate evidence; local Windows test
results do not refresh or replace it.
