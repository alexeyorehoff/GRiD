"""Native C++ shared-library commands for the CPU benchmark helpers."""
import os
from pathlib import Path
import shutil
import subprocess


def compiler_identity():
    compiler = shutil.which("cl" if os.name == "nt" else "g++")
    if compiler is None:
        raise RuntimeError("A native C++ compiler is required for the benchmark helpers")
    result = subprocess.run([compiler] if os.name == "nt" else [compiler, "--version"],
                            capture_output=True, text=True)
    return compiler + ":" + result.stdout + result.stderr


def shared_command(source, output, *, exports=(), include_dirs=()):
    if os.name == "nt":
        return [shutil.which("cl"), "/nologo", "/LD", "/O2", "/MD", "/EHsc", "/std:c++17", "/utf-8",
                *(f"/I{p}" for p in include_dirs), str(source),
                f"/Fo{Path(output).with_suffix('.obj')}", f"/Fe{output}",
                "/link", *(f"/EXPORT:{symbol}" for symbol in exports)]
    return [shutil.which("g++"), "-std=c++17", "-O2", "-shared", "-fPIC", "-pthread",
            *(f"-I{p}" for p in include_dirs), str(source), "-ldl", "-o", str(output)]
