"""Tests for batch optimization — real ORB models."""

import pytest

from gpuma.config import Config
from gpuma.optimizer import optimize_structure_batch
from gpuma.structure import Structure

from conftest import DEVICE, MULTI_XYZ, SMALL_BATCH_XYZ, requires_gpu


class TestOptimizeSequential:
    """Sequential (one-by-one) batch optimization."""

    def test_sequential_batch(self, methane, ethanol, orb_sequential_config):
        """Multiple structures are optimized sequentially with energies."""
        results = optimize_structure_batch([methane, ethanol], orb_sequential_config)
        assert len(results) == 2
        for r in results:
            assert r.energy is not None

    def test_sequential_on_cpu(self, methane):
        """Sequential optimization works on CPU."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "sequential",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": "cpu"},
        })
        results = optimize_structure_batch([methane], config)
        assert len(results) == 1
        assert results[0].energy is not None


class TestOptimizeBatch:
    """GPU-accelerated batch optimization via torch-sim."""

    @requires_gpu
    def test_batch_ethanol(self, ethanol, orb_config):
        """Batch optimization of identical structures produces valid energies."""
        results = optimize_structure_batch([ethanol] * 3, orb_config)
        assert len(results) == 3
        for r in results:
            assert r.energy is not None

    @requires_gpu
    def test_batch_from_file(self, orb_config):
        """Batch optimization of real structures from test data file."""
        import gpuma

        structures = gpuma.read_multi_xyz(str(SMALL_BATCH_XYZ))[:10]
        results = optimize_structure_batch(structures, orb_config)
        assert len(results) >= 1
        for r in results:
            assert r.energy is not None

    @requires_gpu
    def test_batch_fallback_on_cpu(self, methane):
        """Batch mode with CPU device falls back to sequential."""
        config = Config({
            "optimization": {
                "batch_optimization_mode": "batch",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": "cpu"},
        })
        results = optimize_structure_batch([methane], config)
        assert len(results) == 1

    @requires_gpu
    def test_batch_unknown_mode(self, methane, orb_config):
        """Unknown optimization mode raises ValueError."""
        orb_config.optimization.batch_optimization_mode = "unknown"
        with pytest.raises(ValueError, match="Unknown optimization mode"):
            optimize_structure_batch([methane], orb_config)

    @requires_gpu
    def test_batch_mixed_size_structures(self, ethanol):
        """Autobatcher handles structures with different atom counts."""
        import gpuma

        structures = gpuma.read_multi_xyz(str(MULTI_XYZ))
        config = Config({
            "optimization": {
                "batch_optimization_mode": "batch",
                "force_convergence_criterion": 0.5,
            },
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE, "max_atoms_to_try": 10000},
        })
        results = optimize_structure_batch(structures, config)
        assert len(results) == len(structures)
        for r in results:
            assert r.energy is not None

    @requires_gpu
    def test_batch_preserves_structure_count(self, orb_config):
        """Batch optimization returns exactly as many structures as input."""
        import gpuma

        structures = gpuma.read_multi_xyz(str(MULTI_XYZ))
        n = len(structures)
        results = optimize_structure_batch(structures, orb_config)
        assert len(results) == n


class TestSequentialVsBatchConsistency:
    """Sequential and batch modes should produce comparable results."""

    @requires_gpu
    def test_energies_are_comparable(self):
        """Same structures give similar energies in sequential and batch modes."""
        import gpuma

        all_structures = gpuma.read_multi_xyz(str(MULTI_XYZ))

        def _copy(structs):
            return [
                Structure(
                    symbols=list(s.symbols),
                    coordinates=[list(c) for c in s.coordinates],
                    charge=s.charge,
                    multiplicity=s.multiplicity,
                )
                for s in structs
            ]

        common = {
            "optimization": {"force_convergence_criterion": 5e-2},
            "model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"},
            "technical": {"device": DEVICE, "max_atoms_to_try": 10000},
        }

        seq_config = Config({
            **common,
            "optimization": {
                **common["optimization"],
                "batch_optimization_mode": "sequential",
            },
        })
        batch_config = Config({
            **common,
            "optimization": {
                **common["optimization"],
                "batch_optimization_mode": "batch",
            },
        })

        seq_results = optimize_structure_batch(_copy(all_structures), seq_config)
        batch_results = optimize_structure_batch(_copy(all_structures), batch_config)

        assert len(seq_results) == len(batch_results)

        for s, b in zip(seq_results, batch_results):
            assert abs(s.energy - b.energy) < 1.0, (
                f"Sequential ({s.energy:.4f}) and batch ({b.energy:.4f}) "
                f"energies differ by more than 1 eV"
            )
