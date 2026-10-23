"""Tests for Fairchem model loading — requires HF_TOKEN."""

import pytest

from gpuma.config import Config
from gpuma.models import (
    AVAILABLE_FAIRCHEM_MODELS,
    load_calculator,
    load_torchsim_model,
)

from conftest import DEVICE, requires_gpu, requires_hf_token

from .conftest import METHANE


class TestFairchemCalculator:
    """Fairchem ASE calculator and torch-sim model loading."""

    @requires_hf_token
    @requires_gpu
    @pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
    def test_load_and_forward_pass(self, model_name):
        """Each Fairchem model loads and produces non-zero energy on methane."""
        config = Config({
            "model": {"model_type": "fairchem", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        calc = load_calculator(config)
        assert calc is not None

        atoms = METHANE.copy()
        atoms.calc = calc
        atoms.info = {"charge": 0, "spin": 1}
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert energy != 0.0

    @requires_hf_token
    @requires_gpu
    @pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
    def test_load_torchsim(self, model_name):
        """Each Fairchem torch-sim model loads successfully."""
        config = Config({
            "model": {"model_type": "fairchem", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert model is not None


class TestFairchemD3Correction:
    """D3 dispersion correction for Fairchem/UMA models (torch-sim D3DispersionModel)."""

    @requires_hf_token
    @requires_gpu
    def test_d3_changes_energy(self):
        """Enabling D3 correction produces a different energy than without."""
        config_no_d3 = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
                "d3_correction": False,
            },
            "technical": {"device": DEVICE},
        })
        config_d3 = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
                "d3_correction": True,
                "d3_functional": "PBE",
                "d3_damping": "BJ",
            },
            "technical": {"device": DEVICE},
        })

        calc_no_d3 = load_calculator(config_no_d3)
        calc_d3 = load_calculator(config_d3)

        atoms1 = METHANE.copy()
        atoms1.calc = calc_no_d3
        atoms1.info = {"charge": 0, "spin": 1}
        e_no_d3 = atoms1.get_potential_energy()

        atoms2 = METHANE.copy()
        atoms2.calc = calc_d3
        atoms2.info = {"charge": 0, "spin": 1}
        e_d3 = atoms2.get_potential_energy()

        assert e_no_d3 != e_d3, "D3 correction should change energy"

    @requires_hf_token
    @requires_gpu
    def test_d3_torchsim_returns_sum_model(self):
        """D3-enabled UMA torch-sim model is a SumModel(uma, d3)."""
        from torch_sim.models.interface import SumModel  # type: ignore

        config = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
                "d3_correction": True,
                "d3_functional": "PBE",
                "d3_damping": "BJ",
            },
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert isinstance(model, SumModel)
        assert len(list(model.models)) == 2

    @requires_hf_token
    @requires_gpu
    def test_d3_contribution_matches_orb_path(self):
        """The D3 energy contribution must agree across UMA and ORB paths.

        Both backends share the same ``nvalchemiops`` kernel, so the
        Δ(D3 - no D3) on a fixed geometry must be numerically equivalent
        within float-precision (the underlying neighbor-list layouts
        differ slightly between paths; we tolerate ~1e-3 eV).
        """
        d3_kwargs = {"d3_correction": True, "d3_functional": "PBE", "d3_damping": "BJ"}

        def _delta(model_type: str, model_name: str) -> float:
            cfg_off = Config({
                "model": {"model_type": model_type, "model_name": model_name},
                "technical": {"device": DEVICE},
            })
            cfg_on = Config({
                "model": {"model_type": model_type, "model_name": model_name, **d3_kwargs},
                "technical": {"device": DEVICE},
            })
            atoms_off = METHANE.copy()
            atoms_off.calc = load_calculator(cfg_off)
            atoms_off.info = {"charge": 0, "spin": 1}
            atoms_on = METHANE.copy()
            atoms_on.calc = load_calculator(cfg_on)
            atoms_on.info = {"charge": 0, "spin": 1}
            return atoms_on.get_potential_energy() - atoms_off.get_potential_energy()

        delta_uma = _delta("fairchem", "uma-s-1p1")
        delta_orb = _delta("orb", "orb_v3_direct_omol")

        assert abs(delta_uma - delta_orb) < 1e-3, (
            f"D3 contribution mismatch: UMA={delta_uma:.6f} ORB={delta_orb:.6f}"
        )
        assert delta_uma < 0, f"D3(BJ) should be attractive; got {delta_uma:+.6f}"


class TestFairchemModelRegistry:
    """Fairchem model name registry."""

    def test_fairchem_model_names_exist(self):
        """AVAILABLE_FAIRCHEM_MODELS contains all expected UMA models."""
        assert set(AVAILABLE_FAIRCHEM_MODELS) == {
            "uma-s-1p2",
            "uma-s-1p2p1",
            "uma-s-1p2p1-ft-d4rev",
            "uma-s-1p1",
            "uma-m-1p1",
        }


class TestFairchemD3Caching:
    """_FairchemD3Calculator must evaluate each geometry once, not twice.

    Both the Fairchem calculator and the D3 model are stubs here: the property
    under test is how many times the wrapper calls through for a given
    geometry, which real models would only make slower to measure.
    """

    @staticmethod
    def _wrapper():
        import numpy as np
        import torch

        from gpuma.models.dispersion import _FairchemD3Calculator

        calls = {"ml": 0, "d3": 0}

        class FakeFairchem:
            def __init__(self):
                self.results = {}

            def calculate(self, atoms=None, properties=None, system_changes=None):
                calls["ml"] += 1
                self.results = {
                    "energy": float(len(atoms)),
                    "forces": np.zeros((len(atoms), 3)),
                }

        class FakeD3:
            dtype = torch.float64

            def forward(self, state):
                calls["d3"] += 1
                n = state.positions.shape[0]
                return {
                    "energy": torch.zeros(1, dtype=torch.float64),
                    "forces": torch.zeros((n, 3), dtype=torch.float64),
                }

        wrapper = _FairchemD3Calculator(FakeFairchem(), FakeD3(), torch.device("cpu"))
        return wrapper, calls

    def test_energy_then_forces_evaluates_once(self):
        """The ASE step order that used to cost two full evaluations."""
        wrapper, calls = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        wrapper.get_potential_energy(atoms)
        wrapper.get_forces(atoms)

        assert calls == {"ml": 1, "d3": 1}

    def test_moving_the_atoms_recomputes(self):
        """Caching must key on geometry, not on the Atoms object's identity.

        ASE optimizers mutate one Atoms in place, so an identity-keyed cache
        would return stale forces for every step after the first.
        """
        wrapper, calls = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        wrapper.get_potential_energy(atoms)
        atoms.positions[0][0] += 0.1
        wrapper.get_potential_energy(atoms)

        assert calls == {"ml": 2, "d3": 2}

    def test_changing_charge_recomputes(self):
        """Charge and spin feed the UMA prediction, so they invalidate too."""
        wrapper, calls = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        wrapper.get_potential_energy(atoms)
        atoms.info["charge"] = -1
        wrapper.get_potential_energy(atoms)

        assert calls == {"ml": 2, "d3": 2}

    def test_calculation_required_tracks_the_cache(self):
        """ASE consults this before deciding to call calculate()."""
        wrapper, _ = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        assert wrapper.calculation_required(atoms, ("energy",)) is True
        wrapper.get_potential_energy(atoms)
        assert wrapper.calculation_required(atoms, ("energy",)) is False
        atoms.positions[0][0] += 0.1
        assert wrapper.calculation_required(atoms, ("energy",)) is True


class TestFairchemLocalCheckpoint:
    """A config supplying only ``model_path`` must work on both paths.

    These stub out ``FairChemModel``/``pretrained_mlip`` rather than downloading
    a checkpoint: the behaviour under test is purely which config fields each
    loader insists on *before* it constructs anything, so the real model would
    add a token requirement without adding coverage.
    """

    @staticmethod
    def _checkpoint_only_config(tmp_path):
        ckpt = tmp_path / "local.pt"
        ckpt.write_bytes(b"")
        return Config({
            "model": {"model_type": "fairchem", "model_name": "", "model_path": str(ckpt)},
            "technical": {"device": "cpu"},
        })

    def test_torchsim_accepts_checkpoint_without_model_name(self, tmp_path, monkeypatch):
        """Batch mode used to demand model_name even with a checkpoint given.

        _verify_model_name_and_cache_dir raises on an empty model_name, and it
        was called unconditionally, so this config failed in batch mode while
        working in sequential mode.
        """
        import torch_sim.models.fairchem as ts_fairchem

        sentinel = object()
        seen = {}

        def fake_model(**kwargs):
            seen.update(kwargs)
            return sentinel

        monkeypatch.setattr(ts_fairchem, "FairChemModel", fake_model)

        from gpuma.models.fairchem import _load_fairchem_torchsim

        assert _load_fairchem_torchsim(self._checkpoint_only_config(tmp_path)) is sentinel
        assert seen["model"].name == "local.pt"

    def test_calculator_accepts_checkpoint_without_model_name(self, tmp_path, monkeypatch):
        """The sequential path already worked; pin it so the two stay in step."""
        import fairchem.core as fc

        sentinel = object()
        monkeypatch.setattr(
            fc.pretrained_mlip, "load_predict_unit", lambda **kw: sentinel
        )
        monkeypatch.setattr(fc, "FAIRChemCalculator", lambda **kw: sentinel)

        from gpuma.models.fairchem import _load_fairchem_calculator

        assert _load_fairchem_calculator(self._checkpoint_only_config(tmp_path)) is sentinel

    def test_model_name_still_required_without_a_checkpoint(self):
        """Dropping the check entirely would defer the error to a confusing place."""
        from gpuma.models.fairchem import _load_fairchem_torchsim

        config = Config({
            "model": {"model_type": "fairchem", "model_name": ""},
            "technical": {"device": "cpu"},
        })
        with pytest.raises(ValueError, match="Model name must be specified"):
            _load_fairchem_torchsim(config)


def test_d3_build_rejects_dispersion_included_model():
    """D3 switched on after construction still fails before the model loads."""
    import torch

    from gpuma.models.dispersion import _build_d3_dispersion_model

    config = Config({"model": {"model_type": "fairchem", "model_name": "uma-s-1p2p1-ft-d4rev"}})
    config.model.d3_correction = True
    with pytest.raises(ValueError, match="already includes dispersion"):
        _build_d3_dispersion_model(config, torch.device("cpu"), torch.float64, compute_stress=False)


@pytest.mark.parametrize("loader", ["get_predict_unit", "load_predict_unit"])
def test_calculator_uses_batch_inference_settings(monkeypatch, tmp_path, loader):
    """Both ASE-path loaders ask fairchem for the "batch" settings, like torch-sim does."""
    from fairchem.core import pretrained_mlip

    seen = {}

    def fake_loader(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(pretrained_mlip, loader, fake_loader)
    model = {"model_type": "fairchem"}
    if loader == "load_predict_unit":
        checkpoint = tmp_path / "uma.pt"
        checkpoint.write_bytes(b"")
        model["model_path"] = str(checkpoint)
    config = Config({"model": model, "technical": {"device": "cpu"}})
    with pytest.raises(RuntimeError, match="stop"):
        load_calculator(config)
    assert seen["inference_settings"] == "batch"
