"""Source generation must work in a non-UTF8 notebook kernel."""
import io
from pathlib import Path

from grid_assets import robot_urdf
from grid_rbd._compile import generate_sources, _mjx_signature_flags


def test_codegen_and_wrapper_sources_ignore_the_ansi_locale(tmp_path, monkeypatch):
    original_open = io.open

    def ansi_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1251"
        return original_open(file, mode, buffering, encoding, *args, **kwargs)

    monkeypatch.setattr(io, "open", ansi_open)
    generate_sources(robot_urdf("iiwa14"),
                     {"floating_base": False, "algorithm_list": ["inverse_dynamics"]}, tmp_path)
    header = (tmp_path / "grid.cuh").read_text(encoding="utf-8")
    assert "BEGIN GLASS" in header
    assert any(ord(char) > 127 for char in header)
    assert (tmp_path / "wrapper.cu").read_text(encoding="utf-8")
    _mjx_signature_flags(tmp_path / "grid.cuh")
