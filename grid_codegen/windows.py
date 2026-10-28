"""Native MSVC setup for Python and notebook processes launched outside a dev shell."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

_ENV_LOCK = threading.Lock()


def ensure_msvc_environment(*, required: bool = True) -> None:
    """Import the installed x64 Visual Studio environment into this process.

    Existing developer shells are reused. Other platforms are unaffected.
    ``required=False`` allows cache inspection on machines without a compiler.
    """
    if sys.platform != "win32":
        return
    with _ENV_LOCK:
        if shutil.which("cl.exe") and os.environ.get("INCLUDE") and os.environ.get("LIB"):
            return
        installer = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        vswhere = installer / "Microsoft Visual Studio/Installer/vswhere.exe"
        try:
            if not vswhere.is_file():
                raise RuntimeError("Visual Studio C++ tools were not found")
            found = subprocess.run(
                [str(vswhere), "-latest", "-products", "*", "-requires",
                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath", "-utf8"],
                capture_output=True, encoding="utf-8", errors="replace", check=True)
            location = found.stdout.strip()
            if not location:
                raise RuntimeError("No Visual Studio installation has the x64 C++ tools")
            script = Path(location) / "Common7/Tools/VsDevCmd.bat"
            if not script.is_file():
                raise RuntimeError(f"Visual Studio environment script is missing: {script}")
            command = f'""{script}" -no_logo -arch=x64 -host_arch=x64 >nul && set"'
            # cmd parses its command tail itself; list2cmdline would backslash-
            # escape the script's quotes using the C argv convention instead.
            shell = subprocess.list2cmdline([os.environ.get("COMSPEC", "cmd.exe")])
            result = subprocess.run(shell + " /u /d /s /c " + command,
                                    capture_output=True, check=True)
            environment = {}
            for line in result.stdout.decode("utf-16-le").splitlines():
                key, sep, value = line.partition("=")
                if sep and key and not key.startswith("="):
                    environment[key] = value
            if not all(key in {k.upper() for k in environment} for key in ("PATH", "INCLUDE", "LIB")):
                raise RuntimeError("Visual Studio returned an incomplete compiler environment")
            os.environ.update(environment)
            if not shutil.which("cl.exe"):
                raise RuntimeError("Visual Studio did not expose the x64 cl.exe compiler")
        except (OSError, subprocess.SubprocessError, UnicodeError, RuntimeError) as exc:
            if required:
                raise RuntimeError(
                    "Native CUDA compilation needs Visual Studio 2022 with Desktop development "
                    "with C++. Install those tools or start Python from an x64 developer shell."
                ) from exc
