"""Regression tests for ryplotspherical.py — spherical trajectory utilities."""

import os
import sys
import tempfile
import types

import numpy as np
import pytest

# ── scipy stub ────────────────────────────────────────────────────────────────
# ryplotspherical.py has `from scipy.interpolate import interp1d` at module
# level even though interp1d is only used in the __main__ demo block and not in
# any production function.  Stub it so the test suite does not require scipy.
# This import will be moved to lazy / __main__-only in the refactoring step.
if 'scipy' not in sys.modules:
    _scipy_stub = types.ModuleType('scipy')
    sys.modules['scipy'] = _scipy_stub
if 'scipy.interpolate' not in sys.modules:
    _scipy_interp_stub = types.ModuleType('scipy.interpolate')
    _scipy_interp_stub.interp1d = None   # never called in production code
    sys.modules['scipy.interpolate'] = _scipy_interp_stub
    sys.modules['scipy'].interpolate = _scipy_interp_stub  # type: ignore[attr-defined]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import ryplotspherical as rps


# ── shared fixtures ───────────────────────────────────────────────────────────

SPHERE_0_12 = os.path.join(os.path.dirname(__file__), '..', 'data', 'plotspherical', 'sphere_0_12.off')


@pytest.fixture
def tmp_off_file(tmp_path):
    """Write a minimal valid 4-vertex / 2-face OFF file and return its path."""
    content = (
        "OFF\n"
        "4 2 0\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "-1.0 0.0 0.0\n"
        "3 0 1 2\n"
        "3 1 2 3\n"
    )
    p = tmp_path / "test.off"
    p.write_text(content)
    return str(p)


# ── readOffFile ───────────────────────────────────────────────────────────────

class TestReadOffFile:
    def test_returns_two_arrays(self, tmp_off_file):
        result = rps.readOffFile(tmp_off_file)
        assert len(result) == 2

    def test_vertices_shape(self, tmp_off_file):
        vertices, _ = rps.readOffFile(tmp_off_file)
        assert vertices.shape == (4, 3)

    def test_triangles_shape(self, tmp_off_file):
        _, triangles = rps.readOffFile(tmp_off_file)
        assert triangles.shape == (2, 3)

    def test_first_vertex_values(self, tmp_off_file):
        vertices, _ = rps.readOffFile(tmp_off_file)
        np.testing.assert_allclose(vertices[0], [1.0, 0.0, 0.0])

    def test_last_vertex_values(self, tmp_off_file):
        vertices, _ = rps.readOffFile(tmp_off_file)
        np.testing.assert_allclose(vertices[3], [-1.0, 0.0, 0.0])

    def test_first_triangle_indices(self, tmp_off_file):
        _, triangles = rps.readOffFile(tmp_off_file)
        np.testing.assert_array_equal(triangles[0], [0, 1, 2])

    def test_second_triangle_indices(self, tmp_off_file):
        _, triangles = rps.readOffFile(tmp_off_file)
        np.testing.assert_array_equal(triangles[1], [1, 2, 3])

    def test_not_off_header_returns_none(self, tmp_path):
        bad = tmp_path / "bad.off"
        bad.write_text("INVALID\n4 2 0\n1 0 0\n0 1 0\n0 0 1\n-1 0 0\n3 0 1 2\n3 1 2 3\n")
        vertices, triangles = rps.readOffFile(str(bad))
        assert vertices is None
        assert triangles is None

    def test_real_sphere_0_12(self):
        """Read the bundled sphere_0_12.off and check declared counts."""
        if not os.path.exists(SPHERE_0_12):
            pytest.skip("sphere_0_12.off not found")
        vertices, triangles = rps.readOffFile(SPHERE_0_12)
        assert vertices is not None
        assert triangles is not None
        assert vertices.shape == (12, 3)
        assert triangles.shape == (20, 3)

    def test_real_sphere_vertex_unit_norm(self):
        """Vertices of sphere_0_12 should have unit norm (they're on a unit sphere)."""
        if not os.path.exists(SPHERE_0_12):
            pytest.skip("sphere_0_12.off not found")
        vertices, _ = rps.readOffFile(SPHERE_0_12)
        norms = np.linalg.norm(vertices, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ── getRotateFromOffFile ──────────────────────────────────────────────────────

class TestGetRotateFromOffFile:
    def test_returns_8_tuple(self, tmp_off_file):
        result = rps.getRotateFromOffFile(tmp_off_file, 0, 0, 0)
        assert len(result) == 8

    def test_shapes_consistent(self, tmp_off_file):
        x, y, z, roll, pitch, yaw, vertices, triangles = rps.getRotateFromOffFile(
            tmp_off_file, 0, 0, 0)
        assert x.shape == y.shape == z.shape == roll.shape == pitch.shape == yaw.shape
        assert x.shape[0] == 4

    def test_position_constant_at_xpos(self, tmp_off_file):
        xPos, yPos, zPos = 10.0, 20.0, 30.0
        x, y, z, _, _, _, _, _ = rps.getRotateFromOffFile(tmp_off_file, xPos, yPos, zPos)
        np.testing.assert_allclose(x, xPos)
        np.testing.assert_allclose(y, yPos)
        np.testing.assert_allclose(z, zPos)

    def test_roll_all_zeros(self, tmp_off_file):
        _, _, _, roll, _, _, _, _ = rps.getRotateFromOffFile(tmp_off_file, 0, 0, 0)
        np.testing.assert_array_equal(roll, np.zeros_like(roll))

    def test_vertices_and_triangles_returned(self, tmp_off_file):
        *_, vertices, triangles = rps.getRotateFromOffFile(tmp_off_file, 0, 0, 0)
        assert vertices.shape == (4, 3)
        assert triangles.shape == (2, 3)

    def test_pitch_and_yaw_are_finite(self, tmp_off_file):
        _, _, _, _, pitch, yaw, _, _ = rps.getRotateFromOffFile(tmp_off_file, 0, 0, 0)
        assert np.isfinite(pitch).all()
        assert np.isfinite(yaw).all()

    def test_invalid_file_returns_nones(self, tmp_path):
        bad = tmp_path / "bad.off"
        bad.write_text("INVALID\n4 2 0\n1 0 0\n0 1 0\n0 0 1\n-1 0 0\n3 0 1 2\n3 1 2 3\n")
        result = rps.getRotateFromOffFile(str(bad), 0, 0, 0)
        assert result == (None, None, None, None, None, None, None, None)


# ── getOrbitFromOffFile ───────────────────────────────────────────────────────

class TestGetOrbitFromOffFile:
    def test_returns_8_tuple(self, tmp_off_file):
        result = rps.getOrbitFromOffFile(tmp_off_file, 0, 0, 0, 100)
        assert len(result) == 8

    def test_shapes_consistent(self, tmp_off_file):
        x, y, z, roll, pitch, yaw, vertices, triangles = rps.getOrbitFromOffFile(
            tmp_off_file, 0, 0, 0, 100)
        assert x.shape == y.shape == z.shape == roll.shape == pitch.shape == yaw.shape
        assert x.shape[0] == 4

    def test_roll_all_zeros(self, tmp_off_file):
        _, _, _, roll, _, _, _, _ = rps.getOrbitFromOffFile(tmp_off_file, 0, 0, 0, 100)
        np.testing.assert_array_equal(roll, np.zeros_like(roll))

    def test_sensor_at_correct_distance(self, tmp_off_file):
        """Each sensor position should be exactly `distance` from the target."""
        distance = 500.0
        xT, yT, zT = 10.0, 20.0, 30.0
        x, y, z, _, _, _, vertices, _ = rps.getOrbitFromOffFile(
            tmp_off_file, xT, yT, zT, distance)
        sensor = np.hstack([x, y, z])
        target = np.array([xT, yT, zT])
        dists = np.linalg.norm(sensor - target, axis=1)
        np.testing.assert_allclose(dists, distance, rtol=1e-10)

    def test_vertices_and_triangles_returned(self, tmp_off_file):
        *_, vertices, triangles = rps.getOrbitFromOffFile(tmp_off_file, 0, 0, 0, 100)
        assert vertices.shape == (4, 3)
        assert triangles.shape == (2, 3)

    def test_pitch_yaw_finite(self, tmp_off_file):
        _, _, _, _, pitch, yaw, _, _ = rps.getOrbitFromOffFile(tmp_off_file, 0, 0, 0, 100)
        assert np.isfinite(pitch).all()
        assert np.isfinite(yaw).all()

    def test_invalid_file_returns_nones(self, tmp_path):
        bad = tmp_path / "bad.off"
        bad.write_text("INVALID\n4 2 0\n1 0 0\n0 1 0\n0 0 1\n-1 0 0\n3 0 1 2\n3 1 2 3\n")
        result = rps.getOrbitFromOffFile(str(bad), 0, 0, 0, 100)
        assert result == (None, None, None, None, None, None, None, None)


# ── getRotateFromElevAzim ─────────────────────────────────────────────────────

class TestGetRotateFromElevAzim:
    @pytest.fixture(autouse=True)
    def azel(self):
        self.az = np.linspace(0, 2 * np.pi, 5)    # 5 azimuth samples
        self.el = np.linspace(-np.pi / 4, np.pi / 4, 3)   # 3 elevation samples
        self.n = len(self.az) * len(self.el)       # 15 total grid points

    def test_returns_7_tuple(self):
        result = rps.getRotateFromElevAzim(self.az, self.el, 0, 0, 0)
        assert len(result) == 7

    def test_output_shapes(self):
        x, y, z, roll, pitch, yaw, azel = rps.getRotateFromElevAzim(
            self.az, self.el, 0, 0, 0)
        for arr in (x, y, z, roll, pitch, yaw):
            assert arr.shape == (self.n, 1), f"expected ({self.n}, 1), got {arr.shape}"

    def test_azel_shape(self):
        *_, azel = rps.getRotateFromElevAzim(self.az, self.el, 0, 0, 0)
        assert azel.shape == (self.n, 2)

    def test_position_constant(self):
        xPos, yPos, zPos = 5.0, -3.0, 1.5
        x, y, z, _, _, _, _ = rps.getRotateFromElevAzim(
            self.az, self.el, xPos, yPos, zPos)
        np.testing.assert_allclose(x, xPos)
        np.testing.assert_allclose(y, yPos)
        np.testing.assert_allclose(z, zPos)

    def test_roll_all_zeros(self):
        _, _, _, roll, _, _, _ = rps.getRotateFromElevAzim(self.az, self.el, 0, 0, 0)
        np.testing.assert_array_equal(roll, np.zeros_like(roll))

    def test_yaw_matches_azimuth_grid(self):
        _, _, _, _, _, yaw, _ = rps.getRotateFromElevAzim(self.az, self.el, 0, 0, 0)
        azim_grid, _ = np.meshgrid(self.az, self.el)
        np.testing.assert_allclose(yaw, azim_grid.reshape(-1, 1))

    def test_pitch_matches_elevation_grid(self):
        _, _, _, _, pitch, _, _ = rps.getRotateFromElevAzim(self.az, self.el, 0, 0, 0)
        _, elev_grid = np.meshgrid(self.az, self.el)
        np.testing.assert_allclose(pitch, elev_grid.reshape(-1, 1))


# ── getOrbitFromElevAzim ──────────────────────────────────────────────────────

class TestGetOrbitFromElevAzim:
    @pytest.fixture(autouse=True)
    def azel(self):
        self.az = np.linspace(0, 2 * np.pi, 4)
        self.el = np.linspace(-np.pi / 6, np.pi / 6, 3)
        self.n = len(self.az) * len(self.el)
        self.distance = 200.0

    def test_returns_7_tuple(self):
        result = rps.getOrbitFromElevAzim(self.az, self.el, 0, 0, 0, self.distance)
        assert len(result) == 7

    def test_output_shapes(self):
        x, y, z, roll, pitch, yaw, azel = rps.getOrbitFromElevAzim(
            self.az, self.el, 0, 0, 0, self.distance)
        for arr in (x, y, z, roll, pitch, yaw):
            assert arr.shape == (self.n, 1), f"expected ({self.n}, 1), got {arr.shape}"

    def test_azel_shape(self):
        *_, azel = rps.getOrbitFromElevAzim(self.az, self.el, 0, 0, 0, self.distance)
        assert azel.shape == (self.n, 2)

    def test_roll_all_zeros(self):
        _, _, _, roll, _, _, _ = rps.getOrbitFromElevAzim(
            self.az, self.el, 0, 0, 0, self.distance)
        np.testing.assert_array_equal(roll, np.zeros_like(roll))

    def test_sensor_at_correct_distance_from_target(self):
        xT, yT, zT = 5.0, -3.0, 2.0
        x, y, z, _, _, _, _ = rps.getOrbitFromElevAzim(
            self.az, self.el, xT, yT, zT, self.distance)
        sensor = np.hstack([x, y, z])
        target = np.array([xT, yT, zT])
        dists = np.linalg.norm(sensor - target, axis=1)
        np.testing.assert_allclose(dists, self.distance, rtol=1e-10)

    def test_pitch_yaw_finite(self):
        _, _, _, _, pitch, yaw, _ = rps.getOrbitFromElevAzim(
            self.az, self.el, 0, 0, 0, self.distance)
        assert np.isfinite(pitch).all()
        assert np.isfinite(yaw).all()


# ── writeOSSIMTrajOFFFile ─────────────────────────────────────────────────────

class TestWriteOSSIMTrajOFFFile:
    def test_rotate_creates_lut_file(self, tmp_off_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajOFFFile(
            tmp_off_file, 'Rotate', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        luts = list(tmp_path.glob("*.lut"))
        assert len(luts) == 1

    def test_rotate_creates_triangles_dat(self, tmp_off_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajOFFFile(
            tmp_off_file, 'Rotate', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        dats = list(tmp_path.glob("*Triangles*.dat"))
        assert len(dats) == 1

    def test_rotate_creates_vertices_dat(self, tmp_off_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajOFFFile(
            tmp_off_file, 'Rotate', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        dats = list(tmp_path.glob("*Vertices*.dat"))
        assert len(dats) == 1

    def test_orbit_creates_lut_file(self, tmp_off_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajOFFFile(
            tmp_off_file, 'Orbit', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        luts = list(tmp_path.glob("*.lut"))
        assert len(luts) == 1

    def test_unknown_traj_type_creates_no_files(self, tmp_off_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajOFFFile(
            tmp_off_file, 'UNKNOWN', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        # tmp_path already contains the input test.off fixture file; check that
        # no output files (.lut, .dat) were created by the unknown trajectory type.
        output_files = [f for f in tmp_path.iterdir()
                        if f.suffix in ('.lut', '.dat')]
        assert output_files == []


# ── writeOSSIMTrajElevAzim ────────────────────────────────────────────────────

class TestWriteOSSIMTrajElevAzim:
    def test_rotate_creates_lut_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajElevAzim(
            5, 'testout', 'Rotate', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        luts = list(tmp_path.glob("*.lut"))
        assert len(luts) == 1

    def test_orbit_creates_lut_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajElevAzim(
            5, 'testout', 'Orbit', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        luts = list(tmp_path.glob("*.lut"))
        assert len(luts) == 1

    def test_creates_azel_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajElevAzim(
            5, 'testout', 'Rotate', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        azels = list(tmp_path.glob("*Azel*.dat"))
        assert len(azels) == 1

    def test_unknown_traj_type_creates_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        rps.writeOSSIMTrajElevAzim(
            5, 'testout', 'UNKNOWN', 1000,
            0, 0, -1000, 100, 0, 0, 1, 0.01)
        assert list(tmp_path.iterdir()) == []
