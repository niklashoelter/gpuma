from gpuma.api import (
    optimize_batch_multi_xyz_file,
    optimize_batch_xyz_directory,
    optimize_ensemble_smiles,
    optimize_single_smiles,
    optimize_single_xyz_file,
)
from gpuma.config import Config
from gpuma.structure import Structure


def test_optimize_single_smiles(tmp_path):
    output_file = tmp_path / "out.xyz"
    # mocked calculator returns energy -50.0
    res = optimize_single_smiles("C", output_file=str(output_file))

    assert isinstance(res, Structure)
    assert res.energy == -50.0
    assert output_file.exists()

def test_optimize_single_xyz_file(tmp_path, sample_xyz_content):
    input_file = tmp_path / "in.xyz"
    input_file.write_text(sample_xyz_content)
    output_file = tmp_path / "out.xyz"

    res = optimize_single_xyz_file(str(input_file), output_file=str(output_file))

    assert isinstance(res, Structure)
    assert res.energy == -50.0
    assert output_file.exists()

def test_optimize_ensemble_smiles(tmp_path):
    output_file = tmp_path / "ensemble.xyz"
    # Default batch mode might try to use GPU/batch optimizer if configured,
    # but defaults are usually sequential or cpu fallback.
    # We should ensure we test what we expect.

    # By default, config uses sequential/cpu if no GPU, which our conftest mocks.
    # Actually conftest mocks load_model_* but _optimize_batch_sequential works with
    # mocked calculator.

    results = optimize_ensemble_smiles("C", output_file=str(output_file))

    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].energy == -50.0
    assert output_file.exists()

def test_optimize_batch_multi_xyz_file(tmp_path, sample_multi_xyz_content):
    input_file = tmp_path / "multi.xyz"
    input_file.write_text(sample_multi_xyz_content)
    output_file = tmp_path / "out.xyz"

    results = optimize_batch_multi_xyz_file(str(input_file), output_file=str(output_file))

    assert len(results) == 2
    assert results[0].energy == -50.0
    assert output_file.exists()

def test_optimize_batch_xyz_directory(tmp_path, sample_xyz_content):
    d = tmp_path / "batch_dir"
    d.mkdir()
    (d / "1.xyz").write_text(sample_xyz_content)
    (d / "2.xyz").write_text(sample_xyz_content)
    output_file = tmp_path / "out.xyz"

    results = optimize_batch_xyz_directory(str(d), str(output_file))

    assert len(results) == 2
    assert results[0].energy == -50.0
    assert output_file.exists()

def test_api_with_config_override(tmp_path):
    output_file = tmp_path / "out.xyz"
    cfg = Config({"optimization": {"charge": 1}})

    # We need to make sure read_xyz or smiles_to_xyz respects this charge if passed via config
    # optimize_single_smiles:
    #   multiplicity = getattr(config.optimization, "multiplicity", 1)
    #   structure = smiles_to_xyz(smiles, multiplicity=multiplicity)

    # Wait, charge is NOT passed to smiles_to_xyz in optimize_single_smiles.
    # It seems charge is derived from SMILES in smiles_to_xyz.
    # But for optimize_single_xyz_file:
    #   eff_charge = int(getattr(config.optimization, "charge", 0))
    #   structure = read_xyz(..., charge=eff_charge, ...)

    input_file = tmp_path / "in.xyz"
    input_file.write_text("1\nH\nH 0 0 0")

    res = optimize_single_xyz_file(str(input_file), str(output_file), config=cfg)
    assert res.charge == 1
