"""DFT-D3(BJ) dispersion helpers for the Fairchem backend.

ORB and SevenNet ship their own D3 implementations; the helpers here layer
torch-sim's ``D3DispersionModel`` on top of Fairchem/UMA predictions — as a
``SumModel`` for the batch path and via :class:`_FairchemD3Calculator` for
the single-structure (ASE) path.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from ..config import Config, check_d3_allowed, warn_d3_double_counting

logger = logging.getLogger(__name__)


def _build_d3_dispersion_model(
    config: Config,
    device: torch.device,
    dtype: torch.dtype,
    *,
    compute_stress: bool,
):
    """Construct a torch-sim ``D3DispersionModel`` from the config.

    Reuses orb-models' bundled D3 reference-data file and BJ damping
    parameter table so we don't duplicate them inside gpuma.
    """
    from orb_models.forcefield.inference.d3_model import AlchemiDFTD3  # type: ignore
    from torch_sim.models.dispersion import D3DispersionModel  # type: ignore

    check_d3_allowed(config)
    warn_d3_double_counting(config)
    functional = str(config.model.d3_functional)
    damping = str(config.model.d3_damping)
    coeffs = AlchemiDFTD3.get_d3_coefficients(functional, damping)
    d3_params = AlchemiDFTD3.load_d3_parameters().to(device=device, dtype=dtype)
    logger.info(
        "Applying D3 dispersion correction to Fairchem (functional=%s, damping=%s)",
        functional,
        damping,
    )
    return D3DispersionModel(
        a1=coeffs["a1"],
        a2=coeffs["a2"],
        s8=coeffs["s8"],
        s6=coeffs["s6"],
        d3_params=d3_params,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=compute_stress,
    )


class _FairchemD3Calculator:
    """Thin ASE-style wrapper that adds D3 corrections to a Fairchem calculator.

    We delegate to the underlying ``FAIRChemCalculator`` for the ML
    energy/forces and add the ``D3DispersionModel`` contributions on top.
    Behaves like an ASE calculator for single-structure use.
    """

    def __init__(self, fairchem_calc: Any, d3_model: Any, device: torch.device) -> None:
        self._fairchem = fairchem_calc
        self._d3_model = d3_model
        self._device = device
        self.implemented_properties = ("energy", "forces")
        self.results: dict[str, Any] = {}
        #: Geometry the values in ``results`` belong to, or ``None`` when there
        #: is nothing cached. See :meth:`_state_key`.
        self._results_key: tuple | None = None

    @staticmethod
    def _state_key(atoms: Any) -> tuple:
        """Identify the geometry a result belongs to.

        Keyed on the actual positions, species and charge/spin rather than on
        object identity, because ASE optimizers mutate one ``Atoms`` in place:
        identity is constant across the whole run and would never invalidate.
        """
        return (
            atoms.get_positions().tobytes(),
            atoms.get_atomic_numbers().tobytes(),
            atoms.info.get("charge"),
            atoms.info.get("spin"),
        )

    def calculate(
        self,
        atoms: Any = None,
        properties: tuple[str, ...] = ("energy", "forces"),
        system_changes: Any = None,
    ) -> None:
        """Run Fairchem then add D3 contributions to energy and forces."""
        import numpy as np
        from ase.calculators.calculator import all_changes
        from torch_sim.io import atoms_to_state  # type: ignore

        target = atoms if atoms is not None else getattr(self._fairchem, "atoms", None)
        if target is None:
            raise ValueError("FairchemD3Calculator.calculate requires an Atoms object")

        # An ASE step asks for the energy and then the forces, and both are
        # produced by the same pass. Without this the geometry was evaluated
        # twice per step -- two UMA forward passes and two D3 passes -- for
        # results that are identical by construction.
        key = self._state_key(target)
        if key == self._results_key and self.results:
            return

        self._fairchem.calculate(
            atoms=target,
            properties=list(properties),
            system_changes=system_changes or all_changes,
        )
        e_ml = float(self._fairchem.results["energy"])
        f_ml = np.asarray(self._fairchem.results["forces"], dtype=float)

        state = atoms_to_state(target, device=self._device, dtype=self._d3_model.dtype)
        with torch.no_grad():
            d3_out = self._d3_model.forward(state)
        e_d3 = float(d3_out["energy"][0].item())
        f_d3 = d3_out["forces"].detach().cpu().numpy()

        self.results = {"energy": e_ml + e_d3, "forces": f_ml + f_d3}
        self._results_key = key

    def get_potential_energy(
        self, atoms: Any = None, force_consistent: bool = False
    ) -> float:
        """ASE-style accessor: trigger calculation and return the energy."""
        self.calculate(atoms=atoms, properties=("energy", "forces"))
        return float(self.results["energy"])

    def get_forces(self, atoms: Any = None):
        """ASE-style accessor: trigger calculation and return forces."""
        self.calculate(atoms=atoms, properties=("energy", "forces"))
        return self.results["forces"]

    def get_property(self, name: str, atoms: Any = None, allow_calculation: bool = True):
        """ASE-style accessor for a single property."""
        if not allow_calculation and name not in self.results:
            return None
        self.calculate(atoms=atoms, properties=("energy", "forces"))
        return self.results.get(name)

    def calculation_required(self, atoms: Any, properties) -> bool:
        """Report whether ``atoms`` differs from what ``results`` was computed for."""
        if atoms is None or not self.results:
            return True
        return self._state_key(atoms) != self._results_key
