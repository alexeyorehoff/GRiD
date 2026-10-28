"""An fp64 oracle must compare independent dynamics for the same geometry."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from bs4 import BeautifulSoup

from test.cuda_equivalents.cuda_harness import _fp64_oracle_source


@dataclass(frozen=True)
class _Source:
    urdf_path: str
    revision: str = "unchanged"


def test_fp64_oracle_normalizes_only_joint_origins(tmp_path):
    text = '''<robot name="geometry">
      <link name="body"><inertial><origin rpy="1.5708 0 0"/></inertial></link>
      <joint name="hinge" type="revolute"><origin rpy="3.1416 0.27931 0"/></joint>
    </robot>'''
    path = tmp_path / "original.urdf"
    path.write_text(text, encoding="utf-8")
    source = _Source(str(path))
    matched = _fp64_oracle_source(source, tmp_path)
    soup = BeautifulSoup(Path(matched.urdf_path).read_text(encoding="utf-8"), "xml")
    assert soup.inertial.origin["rpy"] == "1.5708 0 0"
    np.testing.assert_array_equal([float(v) for v in soup.joint.origin["rpy"].split()],
                                  [np.pi, 0.27931, 0.0])
    assert matched.revision == source.revision
    assert path.read_text(encoding="utf-8") == text


def test_gen3_matched_pinocchio_geometry_has_no_fp64_error_floor(tmp_path):
    pytest.importorskip("pinocchio")
    from RBDReference.tests import MANIFEST_PATH
    from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
    from RBDReference.equivalents import build_adapter
    from RBDReference.equivalents.reference_backend import build_project_adapter

    spec = next(case["spec"] for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed")
                if case["spec"].robot_id == "gen3")
    resolved = resolve_robot_spec(spec)
    project = build_project_adapter(spec, resolved, "fixed")
    literal = build_adapter(spec, resolved, "fixed", backend="pinocchio")
    matched = build_adapter(spec, _fp64_oracle_source(resolved, tmp_path), "fixed", backend="pinocchio")
    q = np.linspace(-0.8, 0.6, project.nq)
    qd = np.linspace(-0.3, 0.4, project.nv)
    qdd = np.linspace(0.2, -0.5, project.nv)
    expected = project.inverse_dynamics(q, qd, qdd)
    assert np.max(np.abs(literal.inverse_dynamics(q, qd, qdd) - expected)) > 1e-6
    np.testing.assert_allclose(matched.inverse_dynamics(q, qd, qdd), expected, rtol=1e-12, atol=1e-12)
