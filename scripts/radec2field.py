import argparse

from zvar_utils.spatial import get_field_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a position into ZTF field/ccd/quad/readout channel"
    )
    parser.add_argument("ra", type=float, help="Right ascension in degrees")
    parser.add_argument("dec", type=float, help="Declination in degrees")
    args = parser.parse_args()

    field_ccd_quads = get_field_id(args.ra, args.dec)
    for field, ccd, quad in field_ccd_quads:
        readout_channel = (ccd - 1) * 4 + quad
        print(
            f"Field: {field}, CCD: {ccd}, Quad: {quad} (Readout Channel: {readout_channel})"
        )
