import os

import numpy as np
from numba import njit

FIELDNO, RA_ALL, DEC_ALL = None, None, None

CCD_LAYOUT_X = [
    -3.646513,
    -3.647394,
    -1.920848,
    -1.920383,
    -1.790386,
    -1.790817,
    -0.064115,
    -0.064099,
    0.062113,
    0.062129,
    1.788830,
    1.788400,
    1.918441,
    1.918905,
    3.645452,
    3.644571,
    -3.646416,
    -3.646708,
    -1.919998,
    -1.919844,
    -1.789454,
    -1.789597,
    -0.062733,
    -0.062727,
    0.061814,
    0.061819,
    1.788683,
    1.788540,
    1.918871,
    1.919025,
    3.645736,
    3.645443,
    -3.646562,
    -3.646270,
    -1.919698,
    -1.919852,
    -1.789413,
    -1.789270,
    -0.062544,
    -0.062549,
    0.062876,
    0.062871,
    1.789598,
    1.789741,
    1.919874,
    1.919720,
    3.646292,
    3.646584,
    -3.645853,
    -3.644972,
    -1.918842,
    -1.919306,
    -1.789367,
    -1.788937,
    -0.062651,
    -0.062666,
    0.063143,
    0.063128,
    1.789415,
    1.789845,
    1.919878,
    1.919413,
    3.645543,
    3.646424,
]

CCD_LAYOUT_Y = [
    -3.727898,
    -2.001758,
    -2.004785,
    -3.731333,
    -3.729368,
    -2.002803,
    -2.003812,
    -3.730512,
    -3.730976,
    -2.004276,
    -2.003269,
    -3.729834,
    -3.731505,
    -2.004957,
    -2.001932,
    -3.728073,
    -1.816060,
    -0.089749,
    -0.090622,
    -1.817335,
    -1.816611,
    -0.089881,
    -0.090172,
    -1.817035,
    -1.817472,
    -0.090609,
    -0.090319,
    -1.817048,
    -1.817584,
    -0.090871,
    -0.089998,
    -1.816309,
    0.090679,
    1.816989,
    1.818266,
    0.091552,
    0.091155,
    1.817884,
    1.818309,
    0.091446,
    0.090876,
    1.817739,
    1.817315,
    0.090586,
    0.091290,
    1.818003,
    1.816728,
    0.090417,
    2.002667,
    3.728808,
    3.732241,
    2.005694,
    2.003694,
    3.730258,
    3.731401,
    2.004701,
    2.003834,
    3.730533,
    3.729391,
    2.002826,
    2.004674,
    3.731221,
    3.727789,
    2.001648,
]

# maximal angular distance between the center of the field and its edge
# in radians
ADIST_MAX = 5.66 * np.pi / 180.0


class ZTFFieldData:
    fieldno = None
    ra_all = None
    dec_all = None

    def __init__(self):
        self.fieldno, self.ra_all, self.dec_all = self._get_fields_data()

    def _get_fields_data(self):
        if self.fieldno is not None:
            return FIELDNO, RA_ALL, DEC_ALL

        field_path = os.path.join(os.path.dirname(__file__), "./data/ZTF_Fields.txt")
        _fieldno, _ra_all, _dec_all = np.loadtxt(
            field_path, unpack=True, usecols=(0, 1, 2), dtype="int,float,float"
        )
        # convert to radians
        _ra_all *= np.pi / 180.0
        _dec_all *= np.pi / 180.0
        return _fieldno, _ra_all, _dec_all


ZTFFieldData = ZTFFieldData()


@njit
def fit_line(x, x0, y0, x1, y1):
    """
    Fit a linear function connecting two points (x0,y0) and (x1,y1)
    and evaluate its value at point x.
    """

    return (y1 - y0) * (x - x0) / (x1 - x0) + y0


@njit
def ortographic_projection(ra, dec, ra0, dec0):
    """
    Calculate the ortographic projection onto the (x,y) tangent plane
    with the origin at (ra0,dec0)
    See: https://en.wikipedia.org/wiki/Orthographic_map_projection
    """

    x = -np.cos(dec) * np.sin(ra - ra0)
    y = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)

    x *= 180.0 / np.pi
    y *= 180.0 / np.pi

    return x, y


@njit
def inside_polygon(xp, yp, x, y):
    """
    Check if the given point (xp,yp) is located within the field of view
    of the ZTF camera.
    (xp,yp) are the coordinates of the given object in the ortographic
    projection with the origin at the center of the field of view.
    This code checks if the point is located within any of ZTF CCDs,
    if yes, it returns the ccd and quadrant number. If not, None is
    returned.
    (x,y) are arrays containing the coordinates of CCD vertices, see
    http://www.oir.caltech.edu/twiki_ptf/pub/ZTF/ZTFFieldGrid/ZTF_CCD_Layout.tbl
    """

    for i in range(16):
        idx = 4 * i

        y_test_1 = fit_line(xp, x[idx], y[idx], x[idx + 3], y[idx + 3])
        y_test_2 = fit_line(xp, x[idx + 1], y[idx + 1], x[idx + 2], y[idx + 2])
        if yp < y_test_1 or yp > y_test_2:
            continue

        x_test_1 = fit_line(yp, y[idx], x[idx], y[idx + 1], x[idx + 1])
        x_test_2 = fit_line(yp, y[idx + 3], x[idx + 3], y[idx + 2], x[idx + 2])
        if xp < x_test_1 or xp > x_test_2:
            continue

        ccd = i + 1

        y_test = fit_line(
            xp,
            0.5 * (x[idx + 2] + x[idx + 3]),
            0.5 * (y[idx + 2] + y[idx + 3]),
            0.5 * (x[idx] + x[idx + 1]),
            0.5 * (y[idx] + y[idx + 1]),
        )
        x_test = fit_line(
            yp,
            0.5 * (y[idx] + y[idx + 3]),
            0.5 * (x[idx] + x[idx + 3]),
            0.5 * (y[idx + 1] + y[idx + 2]),
            0.5 * (x[idx + 1] + x[idx + 2]),
        )

        if yp < y_test:
            if xp < x_test:
                quad = 4
            else:
                quad = 3
        else:
            if xp < x_test:
                quad = 1
            else:
                quad = 2

        return ccd, quad

    return None, None


@njit
def great_circle_distance_rad(ra1, dec1, ra2, dec2):
    """
        Distance between two points on the sphere
    :param ra1_deg:
    :param dec1_deg:
    :param ra2_deg:
    :param dec2_deg:
    :return: distance in radias
    """
    delta_ra = np.abs(ra2 - ra1)
    distance = np.arctan2(
        np.sqrt(
            (np.cos(dec2) * np.sin(delta_ra)) ** 2
            + (
                np.cos(dec1) * np.sin(dec2)
                - np.sin(dec1) * np.cos(dec2) * np.cos(delta_ra)
            )
            ** 2
        ),
        np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(delta_ra),
    )
    return distance


@njit
def great_circle_distance(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
        Distance between two points on the sphere
    :param ra1_deg:
    :param dec1_deg:
    :param ra2_deg:
    :param dec2_deg:
    :return: distance in degrees
    """
    # this is orders of magnitude faster than astropy.coordinates.Skycoord.separation
    DEGRA = np.pi / 180.0
    ra1, dec1, ra2, dec2 = (
        ra1_deg * DEGRA,
        dec1_deg * DEGRA,
        ra2_deg * DEGRA,
        dec2_deg * DEGRA,
    )
    return great_circle_distance_rad(ra1, dec1, ra2, dec2) / DEGRA


def get_field_id(ra, dec):
    """
    Find the field, ccd, and quadrant number for a given object.
    ra, dec are coordinates of the object (in degrees), fieldno,
    ra_all, and dec_all are arrays containing ZTF field numbers,
    and coordinates of their centers (ra_all,dec_all).
    These calculations are approximate and may fail if the object
    is located close to the edge of the field of view / edge of the
    reference image.
    """

    ra = ra * np.pi / 180.0  # convert to radians
    dec = dec * np.pi / 180.0  # convert to radians

    nf = len(ZTFFieldData.fieldno)

    res = []

    for i in range(nf):
        adist = great_circle_distance_rad(
            ra, dec, ZTFFieldData.ra_all[i], ZTFFieldData.dec_all[i]
        )
        if adist > ADIST_MAX:
            continue
        x, y = ortographic_projection(
            ra, dec, ZTFFieldData.ra_all[i], ZTFFieldData.dec_all[i]
        )
        ccd, quad = inside_polygon(x, y, CCD_LAYOUT_X, CCD_LAYOUT_Y)
        if ccd is not None and quad is not None:
            res.append([ZTFFieldData.fieldno[i], ccd, quad])

    return res
