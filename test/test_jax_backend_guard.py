"""A CPU-only jax is refused at JAX-handle construction with an install hint
(release check 2026-09-24: the plain `jax` wheel fails deep inside the first FFI
call with "No FFI handler registered ... on a platform Host")."""
import pytest

jax = pytest.importorskip("jax")


def test_cpu_only_jax_is_refused_with_a_hint(monkeypatch):
    from grid_rbd.jax import _require_gpu_backend
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match=r"jax\[cuda12\]"):
        _require_gpu_backend()


def test_gpu_jax_passes_the_guard(monkeypatch):
    from grid_rbd.jax import _require_gpu_backend
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    _require_gpu_backend()


def test_cpu_jax_registration_fails_before_compiling(monkeypatch):
    import grid_rbd.jax as backend
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(backend._grid_rbd, "register_robot",
                        lambda *args, **kwargs: pytest.fail("CPU JAX started a CUDA build"))
    with pytest.raises(RuntimeError, match=r"jax\[cuda12\]"):
        backend.register_robot("cpu_jax_probe", "/unused.urdf")
