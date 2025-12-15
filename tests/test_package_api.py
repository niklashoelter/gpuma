import gpuma as pkg
from gpuma.config import Config


def test_package_exports_and_metadata():
    assert hasattr(pkg, "Structure")
    assert hasattr(pkg, "read_xyz")
    assert hasattr(pkg, "optimize_structure_batch")
    assert hasattr(pkg, "load_model_fairchem")
    assert hasattr(pkg, "load_model_torchsim")


def test_optimize_structure_batch_empty_list_returns_empty():
    assert pkg.optimize_structure_batch([]) == []


def test_optimize_single_xyz_file_uses_given_charge_and_multiplicity(monkeypatch, tmp_path):
    cfg = Config()
    cfg.optimization.charge = 0
    cfg.optimization.multiplicity = 1

    src = tmp_path / "in.xyz"
    src.write_text("""1
H
H 0.0 0.0 0.0
""")

    captured = {}

    def fake_read_xyz(path, charge, multiplicity):
        captured["path"] = path
        captured["charge"] = charge
        captured["multiplicity"] = multiplicity
        return pkg.Structure(["H"], [(0.0, 0.0, 0.0)], charge=charge, multiplicity=multiplicity)

    def fake_optimize_single_structure(struct, config):  # noqa: D401
        # do not call the real optimizer; just echo back
        return struct

    # Patch symbols used in api.optimize_single_xyz_file
    monkeypatch.setattr("gpuma.api.read_xyz", fake_read_xyz)
    monkeypatch.setattr("gpuma.api.optimize_single_structure", fake_optimize_single_structure)

    out = tmp_path / "out.xyz"
    result = pkg.optimize_single_xyz_file(
        input_file=str(src),
        output_file=str(out),
        config=cfg,
        charge=-2,
        multiplicity=3,
    )

    assert captured["charge"] == -2
    assert captured["multiplicity"] == 3
    assert isinstance(result, pkg.Structure)
