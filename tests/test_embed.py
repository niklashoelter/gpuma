"""Tests for batched SMILES -> 3D structure generation.

No mocking — real RDKit and morfeus throughout, matching the rest of the suite.
Tests pin ``device="cpu"`` and pass a small ``n_confs`` so they stay fast: the
default CONF_BUDGET generates 50-300 conformers per molecule, which is far more
work than any of these assertions need.
"""

import copy

import numpy as np
import pytest

from gpuma.config import Config, load_config_from_file
from gpuma.conformer_generation.embed import (
    CONF_BUDGET,
    DEFAULT_PRUNE_RMS,
    _conf_budget,
    _etkdg_params,
    _force_field_for,
    _gpu_ids_from_device,
    _prepare,
    _prune_by_rmsd,
    _rank_by_energy,
    generate_ensembles,
    generate_structures,
)
from gpuma.structure import Structure

# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

#: Small and rigid, so embedding is quick and the atom counts are known.
ETHANOL = "CCO"
METHANE = "C"
BENZOIC_ACID = "c1ccccc1C(=O)O"

#: Pinacol boronate: no MMFF94 parameters, but UFF covers it. Around 0.6% of
#: a typical library falls in this gap, so the UFF rung matters.
UFF_ONLY = "CC1(C)OB(c2ccc(C(=O)O)cc2)OC1(C)C"

#: Parses as a molecule but has neither MMFF94 nor UFF parameters.
NO_FORCE_FIELD_PARAMS = "[Fe](Cl)(Cl)Cl"

#: Rejected by RDKit's SMILES parser.
INVALID_SMILES = "not_a_smiles"

#: Keep conformer counts low; these tests check plumbing, not search quality.
FEW_CONFS = 3

#: Two t-butyls and a methyl: many rotor orientations that are the same
#: conformer once the molecular graph's automorphisms are allowed.
ROTOR_RICH = "Cc1ccccc1C(C)(C)C"

#: Its global minimum needs an intramolecular O-H...O=C hydrogen bond. Rotating
#: that O-H barely moves a heavy atom, so pruning raw embeddings collapsed the
#: molecule to a single conformer and which rotamer you got was a lottery.
SALICYLIC_ACID = "OC(=O)c1ccccc1O"


@pytest.fixture
def cpu_config() -> Config:
    """Config pinned to the CPU backend, so tests never depend on a GPU."""
    config = copy.deepcopy(load_config_from_file())
    config.technical.device = "cpu"
    return config


# ---------------------------------------------------------------------------
# Device string -> gpuIds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", None),
        ("CPU", None),
        ("  cpu  ", None),
        ("cuda", []),
        ("cuda:0", [0]),
        ("cuda:3", [3]),
    ],
)
def test_gpu_ids_from_device(device, expected):
    """Recognised device strings map to nvMolKit gpuIds; None means CPU."""
    assert _gpu_ids_from_device(device) == expected


@pytest.mark.parametrize("device", ["", "weird", "gpu", None])
def test_gpu_ids_from_device_unrecognised_falls_back_to_cpu(device):
    """An unusable device string degrades to CPU rather than raising."""
    assert _gpu_ids_from_device(device) is None


def test_gpu_ids_from_device_bad_index_uses_all_gpus():
    """A malformed CUDA index still selects the GPU backend, not the CPU."""
    assert _gpu_ids_from_device("cuda:x") == []


# ---------------------------------------------------------------------------
# Conformer budget
# ---------------------------------------------------------------------------


def test_conf_budget_follows_rotatable_bond_tiers():
    """Budget increases with flexibility, per CONF_BUDGET."""
    rigid, _ = _prepare(BENZOIC_ACID)
    flexible, _ = _prepare("CCCCCCCCCC(=O)O")
    very_flexible, _ = _prepare("CCCCCCCC/C=C\\CCCCCCCC(=O)O")

    assert _conf_budget(rigid, None) == CONF_BUDGET[0][1]
    assert _conf_budget(flexible, None) == CONF_BUDGET[1][1]
    assert _conf_budget(very_flexible, None) == CONF_BUDGET[2][1]


def test_conf_budget_override_bypasses_tiers():
    """An explicit count applies uniformly regardless of rotatable bonds."""
    rigid, _ = _prepare(BENZOIC_ACID)
    very_flexible, _ = _prepare("CCCCCCCC/C=C\\CCCCCCCC(=O)O")

    assert _conf_budget(rigid, 7) == 7
    assert _conf_budget(very_flexible, 7) == 7


def test_conf_budget_override_is_at_least_one():
    """A nonsensical override is clamped rather than producing zero conformers."""
    mol, _ = _prepare(ETHANOL)
    assert _conf_budget(mol, 0) == 1
    assert _conf_budget(mol, -5) == 1


# ---------------------------------------------------------------------------
# Force field selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (ETHANOL, "MMFF94"),
        (METHANE, "MMFF94"),
        (BENZOIC_ACID, "MMFF94"),
        (UFF_ONLY, "UFF"),
        (NO_FORCE_FIELD_PARAMS, None),
    ],
)
def test_force_field_ladder(smiles, expected):
    """MMFF94 where it applies, UFF next, None when neither does."""
    mol, _ = _prepare(smiles)
    assert _force_field_for(mol) == expected


def test_uff_only_molecule_is_minimized(cpu_config):
    """A molecule without MMFF94 parameters still gets a real geometry via UFF."""
    (structure,) = generate_structures([UFF_ONLY], cpu_config, n_confs=FEW_CONFS)

    assert isinstance(structure, Structure)
    assert "B" in structure.symbols
    assert len({tuple(xyz) for xyz in structure.coordinates}) > 1


# ---------------------------------------------------------------------------
# SMILES preparation
# ---------------------------------------------------------------------------


def test_prepare_adds_hydrogens_and_reads_charge():
    """Hydrogens are explicit and formal charge is read before AddHs."""
    mol, charge = _prepare(ETHANOL)
    assert mol.GetNumAtoms() == 9
    assert charge == 0


def test_prepare_reads_nonzero_formal_charge():
    """Charged species keep their formal charge."""
    _, anion = _prepare("CC(=O)[O-]")
    _, cation = _prepare("C[NH3+]")
    assert anion == -1
    assert cation == 1


@pytest.mark.parametrize("smiles", [INVALID_SMILES, "", "   ", None])
def test_prepare_returns_none_for_unparseable(smiles):
    """Unparseable input yields None rather than raising."""
    assert _prepare(smiles) is None


# ---------------------------------------------------------------------------
# generate_structures
# ---------------------------------------------------------------------------


def test_generate_structures_returns_one_per_input(cpu_config):
    """Every input gets exactly one slot in the result."""
    smiles = [ETHANOL, METHANE, BENZOIC_ACID]
    results = generate_structures(smiles, cpu_config, n_confs=FEW_CONFS)

    assert len(results) == len(smiles)
    assert all(isinstance(s, Structure) for s in results)


def test_generate_structures_geometry_is_valid(cpu_config):
    """Structures have matching symbols/coordinates and are not degenerate."""
    (structure,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)

    assert structure.n_atoms == 9
    assert len(structure.symbols) == len(structure.coordinates)
    assert {"C", "O", "H"} == set(structure.symbols)
    assert all(len(xyz) == 3 for xyz in structure.coordinates)
    # A collapsed embedding would put every atom at the origin.
    assert len({tuple(xyz) for xyz in structure.coordinates}) > 1


def test_generate_structures_preserves_order_around_failures(cpu_config):
    """Failed molecules become None in place, leaving neighbours aligned."""
    smiles = [ETHANOL, INVALID_SMILES, METHANE, "", BENZOIC_ACID]
    results = generate_structures(smiles, cpu_config, n_confs=FEW_CONFS)

    assert len(results) == len(smiles)
    assert results[1] is None
    assert results[3] is None
    assert results[0].n_atoms == 9
    assert results[2].n_atoms == 5
    assert results[4].n_atoms == 15


def test_generate_structures_empty_input(cpu_config):
    """An empty list is returned unchanged without touching a backend."""
    assert generate_structures([], cpu_config) == []


def test_generate_structures_all_invalid(cpu_config):
    """A batch of only bad SMILES yields all None rather than raising."""
    results = generate_structures([INVALID_SMILES, ""], cpu_config)
    assert results == [None, None]


def test_generate_structures_applies_multiplicity(cpu_config):
    """An explicit multiplicity reaches the returned structures."""
    (structure,) = generate_structures(
        [ETHANOL], cpu_config, multiplicity=3, n_confs=FEW_CONFS
    )
    assert structure.multiplicity == 3


def test_generate_structures_multiplicity_defaults_to_config(cpu_config):
    """Omitting multiplicity takes config.optimization.multiplicity."""
    cpu_config.optimization.multiplicity = 2
    (structure,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)
    assert structure.multiplicity == 2


def test_generate_structures_without_force_field_params(cpu_config):
    """A molecule with no MMFF94/UFF parameters still yields a geometry."""
    (structure,) = generate_structures(
        [NO_FORCE_FIELD_PARAMS], cpu_config, n_confs=FEW_CONFS
    )
    assert isinstance(structure, Structure)
    assert "Fe" in structure.symbols


def test_mixed_force_fields_in_one_batch(cpu_config):
    """MMFF94, UFF and unminimizable molecules survive the same batch."""
    smiles = [ETHANOL, UFF_ONLY, NO_FORCE_FIELD_PARAMS]
    results = generate_structures(smiles, cpu_config, n_confs=FEW_CONFS)

    assert all(isinstance(s, Structure) for s in results)
    assert "B" in results[1].symbols
    assert "Fe" in results[2].symbols


def test_generate_structures_loads_config_when_omitted():
    """Config is optional; the default is loaded on demand."""
    (structure,) = generate_structures([METHANE], n_confs=FEW_CONFS)
    assert isinstance(structure, Structure)


# ---------------------------------------------------------------------------
# generate_ensembles
# ---------------------------------------------------------------------------


def test_generate_ensembles_returns_lists(cpu_config):
    """Each molecule gets a list of conformers, capped at max_num_confs."""
    smiles = [ETHANOL, BENZOIC_ACID]
    results = generate_ensembles(smiles, 3, cpu_config, n_confs=8)

    assert len(results) == len(smiles)
    for conformers in results:
        assert 1 <= len(conformers) <= 3
        assert all(isinstance(s, Structure) for s in conformers)


def test_generate_ensembles_conformers_share_composition(cpu_config):
    """Conformers of one molecule differ in geometry, not in atoms."""
    (conformers,) = generate_ensembles([BENZOIC_ACID], 3, cpu_config, n_confs=8)

    first = conformers[0]
    for other in conformers[1:]:
        assert other.symbols == first.symbols
        assert other.n_atoms == first.n_atoms


def test_generate_ensembles_agrees_with_generate_structures(cpu_config):
    """The single-structure entry point returns the ensemble's first conformer."""
    (single,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)
    (ensemble,) = generate_ensembles([ETHANOL], 3, cpu_config, n_confs=FEW_CONFS)

    assert single.n_atoms == ensemble[0].n_atoms
    assert single.symbols == ensemble[0].symbols


def test_generate_ensembles_preserves_order_around_failures(cpu_config):
    """Failures are None in place, as for generate_structures."""
    results = generate_ensembles(
        [ETHANOL, INVALID_SMILES, METHANE], 2, cpu_config, n_confs=FEW_CONFS
    )
    assert len(results) == 3
    assert results[1] is None
    assert isinstance(results[0], list)
    assert isinstance(results[2], list)


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_ensembles_rejects_nonpositive_count(cpu_config, bad):
    """max_num_confs must be positive; None (keep-all) is the one exception."""
    with pytest.raises(ValueError, match="max_num_confs"):
        generate_ensembles([ETHANOL], bad, cpu_config)


# ---------------------------------------------------------------------------
# Keep-all mode (max_num_confs=None)
# ---------------------------------------------------------------------------


def test_generate_ensembles_keep_all_returns_every_pruned_conformer(cpu_config):
    """None keeps whatever survives the embed-time RMSD prune -- no cap.

    A cap far above the conformer budget cannot bite, so it must agree with
    keep-all exactly; a cap of 1 must return a strict subset.
    """
    kwargs = {"n_confs": 8, "seed": 42}
    (keep_all,) = generate_ensembles([BENZOIC_ACID], None, cpu_config, **kwargs)
    (huge_cap,) = generate_ensembles([BENZOIC_ACID], 1000, cpu_config, **kwargs)
    (capped,) = generate_ensembles([BENZOIC_ACID], 1, cpu_config, **kwargs)

    assert len(keep_all) == len(huge_cap)
    assert len(capped) == 1
    assert len(keep_all) >= len(capped)


def test_generate_ensembles_keep_all_is_reproducible(cpu_config):
    """A fixed seed pins the keep-all ensemble, geometries included."""
    kwargs = {"n_confs": 8, "seed": 42}
    (first,) = generate_ensembles([ETHANOL], None, cpu_config, **kwargs)
    (second,) = generate_ensembles([ETHANOL], None, cpu_config, **kwargs)

    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.symbols == b.symbols
        assert np.array(a.coordinates) == pytest.approx(np.array(b.coordinates))


def test_generate_ensembles_keep_all_first_agrees_with_generate_structures(cpu_config):
    """Keep-all still sorts lowest energy first, like the single entry point."""
    kwargs = {"n_confs": FEW_CONFS, "seed": 42}
    (single,) = generate_structures([ETHANOL], cpu_config, **kwargs)
    (ensemble,) = generate_ensembles([ETHANOL], None, cpu_config, **kwargs)

    assert single.symbols == ensemble[0].symbols
    assert np.array(single.coordinates) == pytest.approx(np.array(ensemble[0].coordinates))


# ---------------------------------------------------------------------------
# RMSD pruning
# ---------------------------------------------------------------------------


def _square(scale: float = 1.0) -> list[tuple[float, float, float]]:
    return [(0.0, 0.0, 0.0), (scale, 0.0, 0.0), (scale, scale, 0.0), (0.0, scale, 0.0)]


def test_embedding_does_not_prune():
    """Pruning is deferred, so ETKDG parameters must carry RDKit's -1 sentinel."""
    assert _etkdg_params(42, random_coords=True).pruneRmsThresh == -1.0


def test_prune_collapses_identical_conformers():
    symbols = ["C", "C", "C", "C"]
    coords = [_square(), _square(), _square()]

    assert _prune_by_rmsd(symbols, coords, 0.35) == [0]


def test_prune_keeps_distinct_conformers():
    symbols = ["C", "C", "C", "C"]
    coords = [_square(1.0), _square(3.0)]

    assert _prune_by_rmsd(symbols, coords, 0.35) == [0, 1]


def test_prune_is_invariant_to_rigid_motion():
    """A translated, rotated copy is the same conformer."""
    symbols = ["C", "C", "C", "C"]
    shifted = [(y + 10.0, -x, z) for x, y, z in _square()]

    assert _prune_by_rmsd(symbols, [_square(), shifted], 0.35) == [0]


def test_prune_keeps_the_earliest_member_of_a_cluster():
    """Callers sort by energy first, so index 0 of a cluster is its minimum."""
    symbols = ["C", "C", "C", "C"]
    nudged = [(x + 0.01, y, z) for x, y, z in _square()]
    coords = [_square(), nudged, _square(3.0)]

    assert _prune_by_rmsd(symbols, coords, 0.35) == [0, 2]


@pytest.mark.parametrize("thresh", [0.0, -1.0])
def test_prune_disabled_by_nonpositive_threshold(thresh):
    symbols = ["C", "C", "C", "C"]
    coords = [_square(), _square()]

    assert _prune_by_rmsd(symbols, coords, thresh) == [0, 1]


def test_prune_passes_through_when_too_few_heavy_atoms():
    """Methane has one heavy atom; there is nothing to superimpose on."""
    symbols = ["C", "H", "H", "H", "H"]
    coords = [[(0.0, 0.0, 0.0)] * 5, [(1.0, 1.0, 1.0)] * 5]

    assert _prune_by_rmsd(symbols, coords, 0.35) == [0, 1]


def test_prune_ignores_hydrogen_positions():
    """The metric is heavy-atom only, so moving an H alone is not a difference."""
    symbols = ["C", "C", "C", "C", "H"]
    base = [*_square(), (0.0, 0.0, 1.0)]
    h_moved = [*_square(), (0.0, 0.0, -1.0)]

    assert _prune_by_rmsd(symbols, [base, h_moved], 0.35) == [0]


def test_hydrogen_bonded_rotamer_survives_pruning(cpu_config):
    """Regression: pruning raw embeddings discarded this conformer entirely.

    Salicylic acid's minimum carries an intramolecular O-H...O=C bond. The prune
    now runs after minimization, where that rotamer is a distinct -- and lower --
    minimum, so it has to come back.
    """
    import numpy as np

    (ensemble,) = generate_ensembles(
        [SALICYLIC_ACID], None, cpu_config, n_confs=50, seed=42
    )
    assert ensemble is not None
    # The old embedding-time prune collapsed this molecule to exactly one
    # conformer, so the count alone distinguishes the two behaviours.
    assert len(ensemble) > 1

    shortest = []
    for structure in ensemble:
        symbols = list(structure.symbols)
        coords = np.asarray(structure.coordinates, dtype=float)
        oxygens = [i for i, s in enumerate(symbols) if s == "O"]
        for h, symbol in enumerate(symbols):
            if symbol != "H":
                continue
            distances = sorted((float(np.linalg.norm(coords[h] - coords[o])), o) for o in oxygens)
            if not distances or distances[0][0] > 1.2:
                continue  # not a hydroxyl hydrogen
            shortest.append(distances[1][0])

    assert shortest, "no hydroxyl hydrogen found"
    assert min(shortest) < 2.2, f"no hydrogen-bonded rotamer kept (best {min(shortest):.2f} A)"


def test_prune_threshold_is_honoured_end_to_end(cpu_config):
    """A larger threshold merges more conformers, a disabled one merges none."""
    kwargs = {"n_confs": 20, "seed": 42}
    (default,) = generate_ensembles([BENZOIC_ACID], None, cpu_config, **kwargs)
    (coarse,) = generate_ensembles(
        [BENZOIC_ACID], None, cpu_config, prune_rms_thresh=3.0, **kwargs
    )
    (unpruned,) = generate_ensembles(
        [BENZOIC_ACID], None, cpu_config, prune_rms_thresh=0.0, **kwargs
    )

    assert len(coarse) <= len(default) <= len(unpruned)
    assert len(coarse) < len(unpruned)


def test_default_prune_threshold_unchanged():
    """The value is morfeus's; only where it is applied has moved."""
    assert DEFAULT_PRUNE_RMS == 0.35


# ---------------------------------------------------------------------------
# Symmetry-aware pruning
# ---------------------------------------------------------------------------


def _mol_for(smiles: str):
    """The prepared mol behind ``smiles``, as the backends hand it to the prune."""
    mol, _ = _prepare(smiles)
    return mol


def test_prune_symmetry_respects_the_heavy_atom_guard():
    """Methane's pass-through is checked before the metric, so the flag cannot skip it.

    Stripping carbon hydrogens leaves methane a single atom, which every
    symmetric comparison calls identical -- and would collapse an ensemble the
    caller cannot rebuild.
    """
    coords = [[(0.0, 0.0, 0.0)] * 5 for _ in range(3)]
    symbols = ["C", "H", "H", "H", "H"]

    kept = _prune_by_rmsd(
        symbols, coords, 0.35, symmetric=True, mol=_mol_for(METHANE), smiles=METHANE
    )

    assert kept == [0, 1, 2]


def test_prune_symmetry_collapses_rotor_duplicates(cpu_config):
    """Rotor-equivalent conformers are one basin, so the symmetric metric keeps fewer."""
    kwargs = {"n_confs": 20, "seed": 42}
    (blind,) = generate_ensembles([ROTOR_RICH], None, cpu_config, **kwargs)
    (symmetric,) = generate_ensembles(
        [ROTOR_RICH], None, cpu_config, prune_symmetry=True, **kwargs
    )

    assert len(symmetric) < len(blind)


def test_prune_symmetry_is_off_by_default(cpu_config):
    """The flag absent and the flag False must be the same run, geometries included."""
    kwargs = {"n_confs": 8, "seed": 42}
    (absent,) = generate_ensembles([BENZOIC_ACID], None, cpu_config, **kwargs)
    (explicit,) = generate_ensembles(
        [BENZOIC_ACID], None, cpu_config, prune_symmetry=False, **kwargs
    )

    assert len(absent) == len(explicit)
    for one, other in zip(absent, explicit, strict=True):
        assert np.array(one.coordinates) == pytest.approx(np.array(other.coordinates))


def test_prune_symmetry_keeps_the_lowest_energy_conformer_first(cpu_config):
    """Pruning only ever drops, so the survivor list still starts at the minimum."""
    kwargs = {"n_confs": 20, "seed": 42}
    (unpruned,) = generate_ensembles(
        [ROTOR_RICH], None, cpu_config, prune_rms_thresh=0.0, **kwargs
    )
    (symmetric,) = generate_ensembles(
        [ROTOR_RICH], None, cpu_config, prune_symmetry=True, **kwargs
    )

    assert np.array(symmetric[0].coordinates) == pytest.approx(np.array(unpruned[0].coordinates))


def test_prune_symmetry_survivors_are_a_subsequence_of_the_ensemble(cpu_config):
    """Energy order is preserved: survivors appear in the unpruned list in order."""
    kwargs = {"n_confs": 20, "seed": 42}
    (unpruned,) = generate_ensembles(
        [SALICYLIC_ACID], None, cpu_config, prune_rms_thresh=0.0, **kwargs
    )
    (symmetric,) = generate_ensembles(
        [SALICYLIC_ACID], None, cpu_config, prune_symmetry=True, **kwargs
    )

    positions = []
    for structure in symmetric:
        positions.append(
            next(
                i
                for i, candidate in enumerate(unpruned)
                if np.array(candidate.coordinates) == pytest.approx(np.array(structure.coordinates))
            )
        )
    assert positions == sorted(positions)


def test_prune_symmetry_falls_back_when_over_budget(cpu_config, monkeypatch):
    """An unaffordable symmetry group prunes as the default metric, not as nothing."""
    from gpuma.conformer_generation import rmsd as rmsd_module

    kwargs = {"n_confs": 20, "seed": 42}
    (blind,) = generate_ensembles([ROTOR_RICH], None, cpu_config, **kwargs)
    monkeypatch.setattr(rmsd_module, "SYMMETRIC_BUDGET_SECONDS", 0.0)
    (fallen_back,) = generate_ensembles(
        [ROTOR_RICH], None, cpu_config, prune_symmetry=True, **kwargs
    )

    assert len(fallen_back) == len(blind)


def test_generate_structures_accepts_prune_symmetry(cpu_config):
    """The single-structure entry point takes the flag too, and still returns one."""
    (structure,) = generate_structures(
        [BENZOIC_ACID], cpu_config, n_confs=8, seed=42, prune_symmetry=True
    )

    assert isinstance(structure, Structure)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_cuda_request_falls_back_to_cpu_when_unavailable(cpu_config):
    """A GPU request degrades to CPU rather than failing the run.

    nvMolKit is an optional dependency, so on a machine without a working
    install this exercises the fallback; where it does work, the GPU path runs
    and the same assertions hold.
    """
    cpu_config.technical.device = "cuda"
    (structure,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)

    assert isinstance(structure, Structure)
    assert structure.n_atoms == 9


def test_allow_cpu_fallback_false_surfaces_gpu_failure(cpu_config):
    """With fallback disabled, an unusable GPU raises instead of degrading.

    Skipped when nvMolKit works, since then there is no failure to surface.
    """
    try:
        import nvmolkit.embedMolecules  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("nvMolKit is usable, so the GPU path does not fail")

    cpu_config.technical.device = "cuda"
    with pytest.raises(Exception):  # noqa: B017 - backend decides the type
        generate_structures(
            [ETHANOL], cpu_config, n_confs=FEW_CONFS, allow_cpu_fallback=False
        )


def test_explicit_cpu_never_raises_on_fallback_flag(cpu_config):
    """allow_cpu_fallback is irrelevant when the CPU was requested outright."""
    results = generate_structures(
        [ETHANOL], cpu_config, n_confs=FEW_CONFS, allow_cpu_fallback=False
    )
    assert isinstance(results[0], Structure)


# ---------------------------------------------------------------------------
# Delegation from mol_utils
# ---------------------------------------------------------------------------


def test_mol_utils_smiles_to_structure_delegates(cpu_config):
    """The per-molecule helper returns the same geometry shape as the batch API."""
    from gpuma.conformer_generation.mol_utils import smiles_to_structure

    structure = smiles_to_structure(ETHANOL, config=cpu_config)
    assert isinstance(structure, Structure)
    assert structure.n_atoms == 9


def test_mol_utils_ensemble_delegates(cpu_config):
    """The per-molecule ensemble helper caps at max_num_confs."""
    from gpuma.conformer_generation.mol_utils import smiles_to_conformer_ensemble

    structures = smiles_to_conformer_ensemble(
        BENZOIC_ACID, max_num_confs=2, config=cpu_config
    )
    assert 1 <= len(structures) <= 2
    assert all(isinstance(s, Structure) for s in structures)


def test_mol_utils_raises_on_invalid_smiles(cpu_config):
    """The per-molecule helpers keep raising, unlike the batch API's None."""
    from gpuma.conformer_generation.mol_utils import smiles_to_structure

    with pytest.raises(ValueError):
        smiles_to_structure(INVALID_SMILES, config=cpu_config)


# ---------------------------------------------------------------------------
# Energy ranking (GPU backend)
# ---------------------------------------------------------------------------


def test_rank_by_energy_orders_lowest_first():
    """Position 0 is the winner, so it must carry the lowest energy."""
    assert _rank_by_energy([10, 11, 12], [0.5, -2.0, 1.0], ETHANOL) == [11, 10, 12]


def test_rank_by_energy_without_energies_keeps_embedding_order():
    """No force field applied means there is nothing to rank on."""
    assert _rank_by_energy([10, 11, 12], None, ETHANOL) == [10, 11, 12]


def test_rank_by_energy_length_mismatch_keeps_embedding_order():
    """A short energy list makes the pairing unknowable; do not guess it.

    Ranking on it would index the wrong conformers and silently promote a
    geometry that was never the minimum.
    """
    assert _rank_by_energy([10, 11, 12], [0.5, -2.0], ETHANOL) == [10, 11, 12]


def test_rank_by_energy_ignores_extra_energies():
    """Symmetric to the short case -- a long list is equally untrustworthy."""
    assert _rank_by_energy([10, 11], [0.5, -2.0, -9.0], ETHANOL) == [10, 11]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rank_by_energy_sorts_non_finite_last(bad):
    """A diverged minimization must never win the ensemble."""
    ranked = _rank_by_energy([10, 11, 12], [bad, 1.0, -1.0], ETHANOL)
    assert ranked[-1] == 10
    assert ranked[0] == 12


def test_rank_by_energy_nan_does_not_capture_position_zero():
    """NaN first in the input is the case that regresses if the key is dropped.

    ``sorted`` only ever calls ``<``, and every comparison against NaN is
    False, so an already-first NaN is never moved by a plain value sort.
    """
    ranked = _rank_by_energy([10, 11, 12], [float("nan"), 5.0, 3.0], ETHANOL)
    assert ranked[0] == 12


def test_rank_by_energy_is_stable_for_ties():
    """Equal energies keep embedding order, so runs stay reproducible."""
    assert _rank_by_energy([10, 11, 12], [1.0, 1.0, 1.0], ETHANOL) == [10, 11, 12]


def test_rank_by_energy_returns_a_new_list():
    """The caller indexes ``ranked`` while ``cids`` stays live; no aliasing."""
    cids = [10, 11, 12]
    assert _rank_by_energy(cids, None, ETHANOL) is not cids
