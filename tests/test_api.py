"""Tests for the high-level Python API — real end-to-end with ORB models."""

import pytest

from gpuma.api import (
    optimize_batch_multi_xyz_file,
    optimize_batch_xyz_directory,
    optimize_ensemble_smiles,
    optimize_single_smiles,
    optimize_single_xyz_file,
)
from gpuma.config import Config
from gpuma.structure import Structure

from conftest import (
    DEVICE,
    MULTI_XYZ,
    MULTI_XYZ_DIR,
    SINGLE_XYZ,
    requires_gpu,
    requires_hf_token,
)


def _orb_sequential_config(**overrides):
    """Create an ORB sequential config with optional section overrides."""
    base = {
        "optimization": {
            "batch_optimization_mode": "sequential",
            "force_convergence_criterion": 0.5,
        },
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": DEVICE},
    }
    for section, vals in overrides.items():
        base.setdefault(section, {}).update(vals)
    return Config(base)


def _orb_batch_config(**overrides):
    """Create an ORB batch config with optional section overrides."""
    base = {
        "optimization": {
            "batch_optimization_mode": "batch",
            "force_convergence_criterion": 0.5,
        },
        "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
        "technical": {"device": DEVICE, "max_atoms_to_try": 1000},
    }
    for section, vals in overrides.items():
        base.setdefault(section, {}).update(vals)
    return Config(base)


# ---------------------------------------------------------------------------
# Single structure from SMILES
# ---------------------------------------------------------------------------


class TestOptimizeSingleSmiles:
    """End-to-end: SMILES -> 3D -> optimize -> output."""

    def test_optimize_methane(self, tmp_path):
        """Methane SMILES optimizes and saves to XYZ."""
        out = tmp_path / "methane.xyz"
        result = optimize_single_smiles("C", output_file=str(out), config=_orb_sequential_config())
        assert isinstance(result, Structure)
        assert result.energy is not None
        assert result.n_atoms == 5
        assert out.exists()

    def test_optimize_without_saving(self):
        """Optimization works without specifying an output file."""
        result = optimize_single_smiles("C", config=_orb_sequential_config())
        assert isinstance(result, Structure)
        assert result.energy is not None

    @requires_hf_token
    def test_optimize_smiles_fairchem(self, tmp_path):
        """Fairchem backend works for SMILES optimization."""
        out = tmp_path / "out.xyz"
        config = Config({
            "optimization": {
                "batch_optimization_mode": "sequential",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "fairchem", "model_name": "uma-s-1p1"},
            "technical": {"device": DEVICE},
        })
        result = optimize_single_smiles("C", output_file=str(out), config=config)
        assert result.energy is not None


# ---------------------------------------------------------------------------
# Single structure from XYZ file
# ---------------------------------------------------------------------------


class TestOptimizeSingleXyz:
    """End-to-end: XYZ file -> optimize -> output."""

    def test_optimize_from_file(self, tmp_path):
        """XYZ file is read, optimized, and saved."""
        out = tmp_path / "optimized.xyz"
        result = optimize_single_xyz_file(
            str(SINGLE_XYZ), output_file=str(out), config=_orb_sequential_config(),
        )
        assert isinstance(result, Structure)
        assert result.energy is not None
        assert out.exists()

    def test_file_not_found(self):
        """Missing input file raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            optimize_single_xyz_file("/nonexistent.xyz", config=_orb_sequential_config())


# ---------------------------------------------------------------------------
# Ensemble from SMILES
# ---------------------------------------------------------------------------


class TestOptimizeEnsemble:
    """End-to-end: SMILES -> conformers -> batch optimize -> output."""

    def test_ensemble_methane(self, tmp_path):
        """Conformer ensemble is generated, optimized, and saved."""
        out = tmp_path / "ensemble.xyz"
        config = _orb_sequential_config(
            conformer_generation={"max_num_conformers": 3, "conformer_seed": 42},
        )
        results = optimize_ensemble_smiles("C", output_file=str(out), config=config)
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert r.energy is not None
        assert out.exists()


# ---------------------------------------------------------------------------
# Batch from multi-XYZ file
# ---------------------------------------------------------------------------


class TestOptimizeBatchMultiXyz:
    """End-to-end: multi-XYZ file -> batch optimize -> output."""

    @requires_gpu
    def test_batch_from_multi_xyz(self, tmp_path):
        """GPU batch optimization of multi-XYZ file."""
        out = tmp_path / "batch_out.xyz"
        results = optimize_batch_multi_xyz_file(
            str(MULTI_XYZ), output_file=str(out), config=_orb_batch_config(),
        )
        assert len(results) >= 1
        for r in results:
            assert r.energy is not None
        assert out.exists()

    def test_batch_sequential_from_multi_xyz(self, tmp_path):
        """Sequential optimization of multi-XYZ file (no GPU required)."""
        out = tmp_path / "seq_out.xyz"
        results = optimize_batch_multi_xyz_file(
            str(MULTI_XYZ), output_file=str(out), config=_orb_sequential_config(),
        )
        assert len(results) >= 1

    def test_file_not_found(self):
        """Missing input file raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            optimize_batch_multi_xyz_file("/nonexistent.xyz", config=_orb_batch_config())


# ---------------------------------------------------------------------------
# Batch from XYZ directory
# ---------------------------------------------------------------------------


class TestOptimizeBatchXyzDir:
    """End-to-end: XYZ directory -> batch optimize -> output."""

    @requires_gpu
    def test_batch_from_directory(self, tmp_path):
        """GPU batch optimization from a directory of XYZ files."""
        out = tmp_path / "dir_out.xyz"
        results = optimize_batch_xyz_directory(
            str(MULTI_XYZ_DIR), output_file=str(out), config=_orb_batch_config(),
        )
        assert len(results) >= 1
        for r in results:
            assert r.energy is not None
        assert out.exists()

    def test_sequential_from_directory(self, tmp_path):
        """Sequential optimization from a directory (no GPU required)."""
        out = tmp_path / "seq_dir_out.xyz"
        results = optimize_batch_xyz_directory(
            str(MULTI_XYZ_DIR), output_file=str(out), config=_orb_sequential_config(),
        )
        assert len(results) >= 1
