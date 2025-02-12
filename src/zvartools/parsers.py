import argparse
import os

from zvartools.enums import ALLOWED_BANDS


def candidates_parser() -> argparse.ArgumentParser:
    """
    Create the parser for the candidate catalog creation script

    Returns
    -------
    argparse.ArgumentParser
        Parser for the candidate catalog creation script
    """
    parser = argparse.ArgumentParser(
        description="Run candidate detection on ZVAR results"
    )
    parser.add_argument(
        "--field_min",
        type=int,
        help="Field number on the low end of the range (inclusive)",
    )
    parser.add_argument(
        "--field_max",
        type=int,
        help="Field number on the high end of the range (inclusive)",
    )
    parser.add_argument(
        "--bands", type=str, default="g,r", help="Bands to process, separated by commas"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=2.0,
        help="Radius in arcseconds to search for xmatches in external catalogs",
    )
    parser.add_argument(
        "--periods_path",
        default="./data/periods",
        type=str,
        help="Path where to load the ZVAR files (e.g. periods for a field)",
    )
    parser.add_argument(
        "--output_path", type=str, help="Output directory for the CSV files"
    )
    parser.add_argument(
        "--credentials_path",
        type=str,
        help="Path to the JSON file with the Kowalski credentials",
    )

    return parser


def validate_candidates_args(args) -> argparse.Namespace:
    """
    Validate the arguments for the candidate catalog creation script

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed by the parser

    Returns
    -------
    argparse.Namespace
        Validated arguments

    Raises
    ------
    ValueError
        If any of the arguments is invalid
    """
    # FIELD_MIN
    try:
        args.field_min = int(args.field_min)
    except ValueError:
        raise ValueError("Field min must be an integer")

    # FIELD_MAX
    try:
        args.field_max = int(args.field_max)
    except ValueError:
        raise ValueError("Field max must be an integer")

    # BANDS
    args.bands = [band.strip() for band in args.bands.split(",")]
    if not all(band in ALLOWED_BANDS for band in args.bands):
        raise ValueError(f"Invalid band specified. Allowed bands are {ALLOWED_BANDS}")

    # RADIUS
    try:
        args.radius = float(args.radius)
    except ValueError:
        raise ValueError("Radius must be a number (integer or float, in arcseconds)")

    if args.radius < 0.0 or args.radius > 180.0:
        raise ValueError("Radius must be between 0 and 3600 arcseconds")

    # PERIODS_PATH
    if not os.path.exists(args.periods_path):
        raise ValueError(f"Periods path {args.periods_path} does not exist")

    # OUTPUT_PATH
    if not os.path.exists(args.output_path):
        raise ValueError(f"Output path {args.output_path} does not exist")

    # CREDENTIALS
    if args.credentials_path is None:
        raise ValueError("Credentials file must be specified")
    if not os.path.exists(args.credentials_path):
        raise ValueError(f"Credentials file {args.credentials_path} does not exist")

    return args
