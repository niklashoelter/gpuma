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
    res = optimize_single_smiles("C", output_file=str(output_file))

    assert isinstance(res, Structure)
    assert res.energy == -50.0
    assert output_file.exists()


def test_optimize_single_smiles_orb(tmp_path):
    """Single SMILES optimization with ORB backend."""
    output_file = tmp_path / "out.xyz"
    cfg = Config({"model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"}})
    res = optimize_single_smiles("C", output_file=str(output_file), config=cfg)

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
    # Force sequential mode to avoid torch-sim device issues in mocked environment
    cfg = Config({"optimization": {"batch_optimization_mode": "sequential"}})

    results = optimize_ensemble_smiles("C", output_file=str(output_file), config=cfg)

    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].energy == -50.0
    assert output_file.exists()


def test_optimize_ensemble_smiles_orb(tmp_path):
    """Ensemble optimization with ORB backend (sequential)."""
    output_file = tmp_path / "ensemble_orb.xyz"
    cfg = Config({
        "model": {
            "model_type": "orb",
            "model_name": "orb_v3_direct_omol",
        },
        "optimization": {
            "batch_optimization_mode": "sequential",
        },
    })

    results = optimize_ensemble_smiles("C", output_file=str(output_file), config=cfg)

    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0].energy == -50.0


def test_optimize_batch_multi_xyz_file(tmp_path, sample_multi_xyz_content):
    input_file = tmp_path / "multi.xyz"
    input_file.write_text(sample_multi_xyz_content)
    output_file = tmp_path / "out.xyz"

    # Force sequential mode to avoid torch-sim device issues in mocked environment
    cfg = Config({"optimization": {"batch_optimization_mode": "sequential"}})
    results = optimize_batch_multi_xyz_file(
        str(input_file), output_file=str(output_file), config=cfg
    )

    assert len(results) == 2
    assert results[0].energy == -50.0
    assert output_file.exists()


def test_optimize_batch_xyz_directory(tmp_path, sample_xyz_content):
    d = tmp_path / "batch_dir"
    d.mkdir()
    (d / "1.xyz").write_text(sample_xyz_content)
    (d / "2.xyz").write_text(sample_xyz_content)
    output_file = tmp_path / "out.xyz"

    # Force sequential mode to avoid torch-sim device issues in mocked environment
    cfg = Config({"optimization": {"batch_optimization_mode": "sequential"}})
    results = optimize_batch_xyz_directory(str(d), str(output_file), config=cfg)

    assert len(results) == 2
    assert results[0].energy == -50.0
    assert output_file.exists()


def test_api_with_config_override(tmp_path):
    output_file = tmp_path / "out.xyz"
    cfg = Config({"optimization": {"charge": 1}})

    input_file = tmp_path / "in.xyz"
    input_file.write_text("1\nH\nH 0 0 0")

    res = optimize_single_xyz_file(str(input_file), str(output_file), config=cfg)
    assert res.charge == 1
