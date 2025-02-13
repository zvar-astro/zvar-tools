from typing import Union

import numpy as np
from astropy.utils.data import download_file
from numba import njit

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

# buffer for the CCD layout, to avoid edge effects
CCD_BUFFER = 0.03

DEGRA = np.pi / 180.0


class ZTFFieldData:
    fieldno = None
    ra_all = None
    dec_all = None

    def __init__(self):
        self.fieldno, self.ra_all, self.dec_all = self._get_fields_data()

    def _get_fields_data(self):
        """
        Load the ZTF field data from the file ZTF_Fields.txt

        Returns
        -------
        tuple
            fieldno, ra_all, dec_all
        """
        if self.fieldno is not None:
            return self.fieldno, self.ra_all, self.dec_all

        url = "https://github.com/zvar-astro/zvar-tools/raw/refs/heads/main/data/ZTF_Fields.txt"
        filename = download_file(url, cache=True)

        _fieldno, _ra_all, _dec_all = np.loadtxt(
            filename, unpack=True, usecols=(0, 1, 2), dtype="int,float,float"
        )
        return _fieldno, _ra_all * DEGRA, _dec_all * DEGRA


ZTFFieldData = ZTFFieldData()


@njit
def fit_line(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Fit a linear function connecting two points (x0,y0) and (x1,y1)
    and evaluate its value at point x.

    Parameters
    ----------
    x : float
        x-coordinate of the point
    x0 : float
        x-coordinate of the first point
    y0 : float
        y-coordinate of the first point
    x1 : float
        x-coordinate of the second point
    y1 : float
        y-coordinate of the second point

    Returns
    -------
    float
        y-coordinate of the point on the line
    """

    return (y1 - y0) * (x - x0) / (x1 - x0) + y0


@njit
def ortographic_projection(ra: float, dec: float, ra0: float, dec0: float) -> tuple:
    """
    Calculate the ortographic projection onto the (x,y) tangent plane
    with the origin at (ra0,dec0)
    See: https://en.wikipedia.org/wiki/Orthographic_map_projection

    Parameters
    ----------
    ra : float
        Right ascension of the object
    dec : float
        Declination of the object
    ra0 : float
        Right ascension of the origin
    dec0 : float
        Declination of the origin

    Returns
    -------
    tuple
        x, y coordinates of the object
    """

    x = -np.cos(dec) * np.sin(ra - ra0)
    y = np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)

    x *= 180.0 / np.pi
    y *= 180.0 / np.pi

    return x, y


@njit
def inside_polygon(xp: float, yp: float, x: list, y: list) -> tuple:
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

    Parameters
    ----------
    xp : float
        x-coordinate of the object
    yp : float
        y-coordinate of the object
    x : list
        x-coordinates of CCD vertices
    y : list
        y-coordinates of CCD vertices

    Returns
    -------
    tuple
        ccd, quadrant number
    """

    ccd_quads = []

    for i in range(16):
        idx = 4 * i

        y_test_1 = fit_line(
            xp, x[idx], y[idx] - CCD_BUFFER, x[idx + 3], y[idx + 3] - CCD_BUFFER
        )
        y_test_2 = fit_line(
            xp, x[idx + 1], y[idx + 1] + CCD_BUFFER, x[idx + 2], y[idx + 2] + CCD_BUFFER
        )

        x_test_1 = fit_line(
            yp, y[idx], x[idx] - CCD_BUFFER, y[idx + 1], x[idx + 1] - CCD_BUFFER
        )
        x_test_2 = fit_line(
            yp, y[idx + 3], x[idx + 3] + CCD_BUFFER, y[idx + 2], x[idx + 2] + CCD_BUFFER
        )

        if yp < y_test_1 or yp > y_test_2:
            continue
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

        # also use a buffer here, and a point could be in multiple quadrants
        if yp < y_test or abs(yp - y_test) < CCD_BUFFER:
            if xp < x_test or abs(xp - x_test) < CCD_BUFFER:
                ccd_quads.append((ccd, 4))
            if xp > x_test or abs(xp - x_test) < CCD_BUFFER:
                ccd_quads.append((ccd, 3))

        if yp > y_test or abs(yp - y_test) < CCD_BUFFER:
            if xp < x_test or abs(xp - x_test) < CCD_BUFFER:
                ccd_quads.append((ccd, 1))
            if xp > x_test or abs(xp - x_test) < CCD_BUFFER:
                ccd_quads.append((ccd, 2))

    return ccd_quads


@njit
def great_circle_distance_rad(
    ra1: float, dec1: float, ra2: float, dec2: float
) -> float:
    """
    Distance between two points on the sphere

    Parameters
    ----------
    ra1 : float
        Right ascension of the first object
    dec1 : float
        Declination of the first object
    ra2 : float
        Right ascension of the second object
    dec2 : float
        Declination of the second object

    Returns
    -------
    float
        Distance between two points in radians
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
def great_circle_distance(
    ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float
) -> float:
    """
    Distance between two points on the sphere

    Parameters
    ----------
    ra1_deg : float
        Right ascension of the first object in degrees
    dec1_deg : float
        Declination of the first object in degrees
    ra2_deg : float
        Right ascension of the second object in degrees
    dec2_deg : float
        Declination of the second object in degrees

    Returns
    -------
    float
        Distance between two points in degrees
    """
    ra1, dec1, ra2, dec2 = (
        ra1_deg * DEGRA,
        dec1_deg * DEGRA,
        ra2_deg * DEGRA,
        dec2_deg * DEGRA,
    )
    return great_circle_distance_rad(ra1, dec1, ra2, dec2) / DEGRA


def get_field_id(ra: float, dec: float, radius: Union[float, int, None] = None) -> list:
    """
    Find the field, ccd, and quadrant number for a given object.
    ra, dec are coordinates of the object (in degrees), fieldno,
    ra_all, and dec_all are arrays containing ZTF field numbers,
    and coordinates of their centers (ra_all,dec_all).
    These calculations are approximate and may fail if the object
    is located close to the edge of the field of view / edge of the
    reference image.

    Parameters
    ----------
    ra : float
        Right ascension of the object in degrees
    dec : float
        Declination of the object in degrees
    radius : Union[float, int, None], optional
        Radius of the search in arcsec, by default None.

    Returns
    -------
    list
        List of field, ccd, and quadrant number
    """

    ra = ra * DEGRA
    dec = dec * DEGRA

    field_adist = ADIST_MAX
    if radius is not None:
        field_adist += (radius / 3600) * DEGRA

    nf = len(ZTFFieldData.fieldno)

    res = []

    for i in range(nf):
        adist = great_circle_distance_rad(
            ra, dec, ZTFFieldData.ra_all[i], ZTFFieldData.dec_all[i]
        )
        if adist > field_adist:
            continue
        x, y = ortographic_projection(
            ra, dec, ZTFFieldData.ra_all[i], ZTFFieldData.dec_all[i]
        )
        # TODO, use the radius (if provided) to increase the CCD buffer size
        ccd_quads = inside_polygon(x, y, CCD_LAYOUT_X, CCD_LAYOUT_Y)
        if len(ccd_quads) > 0:
            for ccd, quad in ccd_quads:
                res.append([ZTFFieldData.fieldno[i], ccd, quad])

    return res
