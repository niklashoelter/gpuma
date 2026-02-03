"""Command Line Interface for GPUMA.

This module provides a command-line interface for molecular geometry
optimization using Fairchem UMA models. The CLI supports three main
optimization modes:

1. Single Structure Optimization: Optimize individual molecular structures.
2. Ensemble Optimization (SMILES): Optimize multiple conformers generated
   from a SMILES string using batch inference.
3. Batch Optimization (Files): Optimize multiple structures from multi-XYZ
   files or directories.

The interface is intentionally similar to the original one but branded and
implemented purely for GPUMA.
"""

import argparse
import logging
import sys
import warnings

from .api import (
    optimize_batch_multi_xyz_file,
    optimize_batch_xyz_directory,
    optimize_ensemble_smiles,
    optimize_single_smiles,
    optimize_single_xyz_file,
)
from .config import Config, load_config_from_file, save_config_to_file
from .io_handler import (
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from .logging_utils import configure_logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def _level_from_string(name: str) -> int:
    """Map a log level name to its numeric value.

    Parameters
    ----------
    name:
        Name of the logging level (e.g. ``"INFO"``).

    Returns
    -------
    int
        Corresponding :mod:`logging` module integer level. Defaults to
        :data:`logging.INFO` if the name is unknown.

    """
    levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    return levels.get((name or "").upper(), logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command line argument parser for GPUMA.

    Returns
    -------
    argparse.ArgumentParser
        Configured :class:`ArgumentParser` instance with all subcommands and
        options.

    """
    parser = argparse.ArgumentParser(
        description=("GPUMA - Optimize molecular structures using Fairchem UMA models"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OPTIMIZATION MODES:

1. Single Structure Optimization:
   Optimize individual molecular structures from SMILES or XYZ files.

2. Ensemble Optimization (SMILES):
   Optimize multiple conformers of the same molecule generated from SMILES
   using batch inference.

3. Batch Optimization (Files):
   Optimize multiple structures from a multi-XYZ file or a directory of
   XYZ files.

UTILITY COMMANDS:

4. Structure Conversion:
   Convert SMILES to 3D coordinates without optimization.

5. Conformer Generation:
   Generate multiple conformers from SMILES without optimization.

6. Configuration Management:
   Create default configuration files or validate existing ones.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single structure optimization
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Optimize a single molecular structure",
        description=("Optimize a single molecular structure from a SMILES string or an XYZ file."),
    )
    optimize_input = optimize_parser.add_mutually_exclusive_group(required=True)
    optimize_input.add_argument(
        "--smiles",
        type=str,
        help="SMILES string of the molecule to optimize",
    )
    optimize_input.add_argument(
        "--xyz",
        type=str,
        help="Path to XYZ file containing the structure to optimize",
    )
    optimize_parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total charge for XYZ input (ignored for SMILES). Default: 0",
    )
    optimize_parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help=("Spin multiplicity for XYZ input (ignored for SMILES). Default: 1"),
    )
    optimize_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output XYZ file path for optimized structure",
    )
    optimize_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # 'smiles' alias for single-structure optimization from SMILES only
    smiles_parser = subparsers.add_parser(
        "smiles",
        help="Optimize a single structure from a SMILES string",
        description=("Shorthand for optimizing a single structure directly from a SMILES string."),
    )
    smiles_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string of the molecule to optimize",
    )
    smiles_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output XYZ file path for optimized structure",
    )
    smiles_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Ensemble optimization (conformers of same molecule from SMILES)
    ensemble_parser = subparsers.add_parser(
        "ensemble",
        help="Optimize conformer ensemble from SMILES using batch inference",
        description=(
            "Optimize multiple conformers of the same molecule generated from "
            "SMILES efficiently using batch inference."
        ),
    )
    ensemble_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to generate and optimize conformers",
    )
    ensemble_parser.add_argument(
        "--conformers",
        type=int,
        default=None,
        help="Number of conformers to generate from SMILES (default: from config)",
    )
    ensemble_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output multi-XYZ file path for optimized ensemble",
    )
    ensemble_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # Batch optimization (structures from files)
    batch_parser = subparsers.add_parser(
        "batch",
        help="Optimize multiple structures from multi-XYZ file or directory",
        description=(
            "Optimize multiple structures using batch inference from a "
            "multi-XYZ file or a directory of XYZ files."
        ),
    )
    batch_input = batch_parser.add_mutually_exclusive_group(required=True)
    batch_input.add_argument(
        "--multi-xyz",
        type=str,
        help="Multi-structure XYZ file containing structures to optimize",
    )
    batch_input.add_argument(
        "--xyz-dir",
        type=str,
        help="Directory containing XYZ files of structures to optimize",
    )
    batch_parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total charge for XYZ inputs. Default: 0",
    )
    batch_parser.add_argument(
        "--multiplicity",
        type=int,
        default=1,
        help="Spin multiplicity for XYZ inputs. Default: 1",
    )
    batch_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output multi-XYZ file path for optimized structures",
    )
    batch_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Configuration file path (default: config.json if exists)",
    )

    # SMILES to XYZ conversion (utility)
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert SMILES to 3D coordinates (no optimization)",
        description="Convert SMILES strings to XYZ format with 3D coordinates.",
    )
    convert_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to convert to XYZ format",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output XYZ file path",
    )

    # Conformer generation (utility)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate conformer ensemble from SMILES (no optimization)",
        description="Generate multiple conformers from SMILES without optimization.",
    )
    generate_parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to generate conformers from",
    )
    generate_parser.add_argument(
        "--conformers",
        type=int,
        default=None,
        help="Number of conformers to generate (default: from config)",
    )
    generate_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output multi-XYZ file path for conformer ensemble",
    )

    # Configuration management
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration file management",
        description="Create default configuration files or validate existing ones.",
    )
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--create",
        type=str,
        help="Create default configuration file at specified path",
    )
    config_group.add_argument(
        "--validate",
        type=str,
        help="Validate existing configuration file",
    )

    # Global options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (overrides config setting)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output (overrides config setting)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help=(
            "Override compute device from config. "
            "Accepted values: 'cpu', 'cuda', 'cuda:N' (e.g. 'cuda:0')."
        ),
    )

    return parser


def cmd_optimize(args, config: Config) -> None:
    """Handle the single-structure optimization command.

    This command optimizes a single molecular structure from either a SMILES
    string or an XYZ file. The optimization uses the configured Fairchem UMA
    model.
    """
    try:
        if args.smiles:
            # charge is derived from SMILES; multiplicity can be overridden via config or CLI
            eff_mult = int(
                args.multiplicity
                if hasattr(args, "multiplicity") and args.multiplicity is not None
                else getattr(config.optimization, "multiplicity", 1)
            )
            # write back to config so optimize_single_smiles/smiles_to_xyz see it
            config.optimization.multiplicity = eff_mult
            logger.info(
                "Converting SMILES '%s' to 3D coordinates and optimizing (multiplicity=%d)...",
                args.smiles,
                eff_mult,
            )
            optimized = optimize_single_smiles(
                smiles=args.smiles,
                output_file=args.output,
                config=config,
            )
        else:
            eff_charge = int(
                args.charge
                if hasattr(args, "charge") and args.charge is not None
                else getattr(config.optimization, "charge", 0)
            )
            eff_mult = int(
                args.multiplicity
                if hasattr(args, "multiplicity") and args.multiplicity is not None
                else getattr(config.optimization, "multiplicity", 1)
            )
            logger.info(
                "Reading and optimizing structure from %s (charge=%d, multiplicity=%d)",
                args.xyz,
                eff_charge,
                eff_mult,
            )
            config.optimization.charge = eff_charge
            config.optimization.multiplicity = eff_mult
            optimized = optimize_single_xyz_file(
                input_file=args.xyz,
                output_file=args.output,
                config=config,
            )

        logger.info(
            "Optimization completed. Final energy: %.6f eV",
            optimized.energy if optimized.energy is not None else float("nan"),
        )
        logger.info("Optimized structure saved to %s", args.output)

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error during single structure optimization: %s", exc)
        sys.exit(1)


def cmd_ensemble(args, config: Config) -> None:
    """Handle conformer ensemble optimization from SMILES.

    This command optimizes multiple conformers of the same molecule using
    batch inference.
    """
    try:
        if getattr(args, "charge", None) not in (None, 0) or getattr(
            args, "multiplicity", None
        ) not in (None, 1):
            logger.warning(
                "Non-neutral charges or spin multiplicities are currently "
                "not supported in batch ensemble optimization; falling back "
                "to neutral singlet (charge=0, multiplicity=1).",
            )

        num_conf = args.conformers or config.optimization.max_num_conformers
        logger.info("Generating %d conformers for SMILES: %s", num_conf, args.smiles)
        config.optimization.max_num_conformers = num_conf

        optimized_conformers = optimize_ensemble_smiles(
            smiles=args.smiles,
            output_file=args.output,
            config=config,
        )

        if not optimized_conformers:
            logger.error("No conformers found")
            sys.exit(1)

        logger.info(
            "Ensemble optimization completed successfully! %d optimized conformers saved to %s",
            len(optimized_conformers),
            args.output,
        )

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error during ensemble optimization: %s", exc)
        sys.exit(1)


def cmd_batch(args, config: Config) -> None:
    """Handle batch optimization from files (multi-XYZ or directory)."""
    try:
        eff_charge = int(
            args.charge
            if hasattr(args, "charge") and args.charge is not None
            else getattr(config.optimization, "charge", 0)
        )
        config.optimization.charge = eff_charge
        eff_mult = int(
            args.multiplicity
            if hasattr(args, "multiplicity") and args.multiplicity is not None
            else getattr(config.optimization, "multiplicity", 1)
        )
        config.optimization.multiplicity = eff_mult

        if args.multi_xyz:
            logger.info(
                "Reading structures from multi-XYZ file: %s (charge=%d, multiplicity=%d)",
                args.multi_xyz,
                eff_charge,
                eff_mult,
            )

            structures = optimize_batch_multi_xyz_file(
                input_file=args.multi_xyz,
                output_file=args.output,
                config=config,
            )
        else:
            logger.info(
                "Reading structures from directory: %s (charge=%d, multiplicity=%d)",
                args.xyz_dir,
                eff_charge,
                eff_mult,
            )
            structures = optimize_batch_xyz_directory(
                input_directory=args.xyz_dir,
                output_file=args.output,
                config=config,
            )

        if not structures:
            logger.error("No structures found for batch optimization")
            sys.exit(1)

        logger.info("Batch optimization completed successfully!")
        logger.info("Optimized structures saved to %s", args.output)

    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error during batch optimization: %s", exc)
        sys.exit(1)


def cmd_convert(args, config: Config | None = None) -> None:  # pylint: disable=unused-argument
    """Handle the SMILES to XYZ conversion command.

    This command generates a single 3D structure from SMILES without running
    any optimization.
    """
    try:
        logger.info("Converting SMILES '%s' to XYZ without optimization", args.smiles)
        structure = smiles_to_xyz(args.smiles)
        save_xyz_file(structure, args.output)
        logger.info("Structure saved to %s", args.output)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error during SMILES to XYZ conversion: %s", exc)
        sys.exit(1)


def cmd_generate(args, config: Config) -> None:  # pylint: disable=unused-argument
    """Handle conformer generation from SMILES.

    This command generates a conformer ensemble from a SMILES string without
    optimization.
    """
    try:
        num_conf = args.conformers or config.optimization.max_num_conformers
        logger.info(
            "Generating %d conformers for SMILES (no optimization): %s",
            num_conf,
            args.smiles,
        )
        structures = smiles_to_ensemble(args.smiles, num_conf)
        comments = [
            f"Generated conformer {i + 1} from SMILES: {args.smiles}"
            for i in range(len(structures))
        ]
        save_multi_xyz(structures, args.output, comments)
        logger.info("Generated conformers saved to %s", args.output)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Error during conformer generation: %s", exc)
        sys.exit(1)


def cmd_config(args, config: Config) -> None:
    """Handle configuration file management commands."""
    if args.create:
        logger.info("Creating default configuration file at %s", args.create)
        save_config_to_file(config, args.create)
        logger.info("Configuration file created")
    elif args.validate:
        logger.info("Validating configuration file at %s", args.validate)
        _ = load_config_from_file(args.validate)
        logger.info("Configuration file is valid")


def _apply_global_verbosity_flags(config: Config, verbose: bool, quiet: bool) -> None:
    """Apply global verbosity CLI flags to the configuration in-place."""
    if verbose:
        config.optimization.logging_level = "DEBUG"
    elif quiet:
        config.optimization.logging_level = "ERROR"


def _apply_device_override(config: Config, device) -> None:
    """Apply a global device override if provided via CLI."""
    if device:
        config.optimization.device = device


def main(argv=None) -> int:
    """Entry point for the ``gpuma`` CLI.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments. If not provided, uses
        :data:`sys.argv`.

    Returns
    -------
    int
        Exit status code (``0`` for success).

    """
    parser = setup_parser()
    args = parser.parse_args(argv)

    # Load configuration from file if specified, otherwise try default
    cfg_path = getattr(args, "config", None) or "config.json"
    config = load_config_from_file(cfg_path)

    # Apply verbosity and device flags
    _apply_global_verbosity_flags(config, args.verbose, args.quiet)
    _apply_device_override(config, getattr(args, "device", None))

    # Configure logging
    logging_level = _level_from_string(config.optimization.logging_level)
    configure_logging(logging_level)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "optimize" or args.command == "smiles":
        cmd_optimize(args, config)
    elif args.command == "ensemble":
        cmd_ensemble(args, config)
    elif args.command == "batch":
        cmd_batch(args, config)
    elif args.command == "convert":
        cmd_convert(args, config)
    elif args.command == "generate":
        cmd_generate(args, config)
    elif args.command == "config":
        cmd_config(args, config)
    else:  # pragma: no cover - defensive guard
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
