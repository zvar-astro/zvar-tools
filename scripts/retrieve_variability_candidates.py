import numpy as np

from zvar_utils.candidate import (
    get_candidates,
    save_candidates_to_csv,
    add_gaia_xmatch_to_candidates,
)
from zvar_utils.kowalski import connect_to_kowalski
from zvar_utils.parsers import candidates_parser, validate_candidates_args
from zvar_utils.periodicity import load_field_periodicity_data

if __name__ == "__main__":
    parser = candidates_parser()
    args = validate_candidates_args(parser.parse_args())

    field_min = args.field_min
    field_max = args.field_max
    bands = args.bands
    radius = args.radius
    periods_path = args.periods_path
    output_path = args.output_path
    credentials_path = args.credentials_path

    k = connect_to_kowalski(credentials_path)

    fields = list(set(np.arange(field_min, field_max + 1)))

    for field in fields:
        for band in bands:
            (
                psids,
                ra,
                dec,
                ratio_valid,
                freqs,
                sigs_clean,
            ) = load_field_periodicity_data(
                field, band, periods_path
            )  # Load the data
            candidate_list = get_candidates(
                psids, ra, dec, ratio_valid, freqs, sigs_clean
            )  # Find the candidates
            candidate_list = add_gaia_xmatch_to_candidates(
                k, candidate_list, 5
            )  # Fill in the Gaia data
            save_candidates_to_csv(
                candidate_list, field, band, output_path
            )  # Write the candidates to a CSV file
