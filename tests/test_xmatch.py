from zvartools.candidate import VariabilityCandidate
from zvartools.external import (
    connect_to_kowalski,
    query_gaia,
    query_ps1,
    query_2mass,
    query_allwise,
    query_galex,
)

credentials_path = "./credentials.json"

k = connect_to_kowalski(credentials_path)

candidate = VariabilityCandidate(
    psid=149053599892210341,
    ra=359.9892272949219,
    dec=34.20810317993164,
    valid=1.0,
    freq=None,
    fap=None,
    best_M=5,
)


def test_gaia_xmatch():
    xmatch_per_candidate = query_gaia(
        k, [candidate.id], [candidate.ra], [candidate.dec], 5.0
    )
    assert 149053599892210341 in xmatch_per_candidate

    xmatch = xmatch_per_candidate[149053599892210341]

    assert xmatch["id"] == 2875126085188101504
    assert xmatch["ra"] == 359.98922725671724
    assert xmatch["dec"] == 34.208105082089176
    assert xmatch["parallax"] == 4.6704942599886765
    assert xmatch["parallax_error"] == 0.039022523999999996
    assert xmatch["pmra"] == 17.838292582751816
    assert xmatch["pmra_error"] == 0.03712507
    assert xmatch["pmdec"] == 2.6669618109416886
    assert xmatch["pmdec_error"] == 0.026505275
    assert xmatch["astrometric_n_good_obs_al"] == 400
    assert xmatch["astrometric_chi2_al"] == 436.64474
    assert xmatch["ruwe"] == 1.0316383
    assert xmatch["phot_g_mean_flux"] == 8759.786131134488
    assert xmatch["phot_g_mean_flux_error"] == 6.2475442999999995
    assert xmatch["phot_g_mean_mag"] == 15.831133
    assert xmatch["phot_bp_mean_flux"] == 2002.6571921151572
    assert xmatch["phot_bp_mean_flux_error"] == 11.913191000000001
    assert xmatch["phot_bp_mean_mag"] == 17.084526
    assert xmatch["phot_rp_mean_flux"] == 10197.112668226617
    assert xmatch["phot_rp_mean_flux_error"] == 18.354717
    assert xmatch["phot_rp_mean_mag"] == 14.726703
    assert xmatch["phot_bp_rp_excess_factor"] == 1.3927017


def test_ps1_xmatch():
    xmatch_per_candidate = query_ps1(k, [candidate.id])
    assert 149053599892210341 in xmatch_per_candidate

    xmatch = xmatch_per_candidate[149053599892210341]

    assert xmatch["id"] == 149053599892210341 == candidate.id
    assert xmatch["gMeanPSFMag"] == 17.488899
    assert xmatch["gMeanPSFMagErr"] == 0.0063439999
    assert xmatch["rMeanPSFMag"] == 16.256399
    assert xmatch["rMeanPSFMagErr"] == 0.0049950001
    assert xmatch["iMeanPSFMag"] == 15.1836
    assert xmatch["iMeanPSFMagErr"] == 0.0035959998999999998
    assert xmatch["zMeanPSFMag"] == 14.690999999999999
    assert xmatch["zMeanPSFMagErr"] == 0.0051589999
    assert xmatch["yMeanPSFMag"] == 14.4642
    assert xmatch["yMeanPSFMagErr"] == 0.003332


def test_2mass_xmatch():
    xmatch_per_candidate = query_2mass(
        k, [candidate.id], [candidate.ra], [candidate.dec], 5.0
    )
    assert 149053599892210341 in xmatch_per_candidate

    xmatch = xmatch_per_candidate[149053599892210341]

    assert xmatch["id"] == "23595739+3412291"
    assert xmatch["ra"] == 359.989154
    assert xmatch["j_m"] == 13.286
    assert xmatch["j_cmsig"] == 0.019
    assert xmatch["h_m"] == 12.595
    assert xmatch["h_cmsig"] == 0.023
    assert xmatch["k_m"] == 12.387
    assert xmatch["k_cmsig"] == 0.022


def test_allwise_xmatch():
    xmatch_per_candidate = query_allwise(
        k, [candidate.id], [candidate.ra], [candidate.dec], 5.0
    )
    assert 149053599892210341 in xmatch_per_candidate

    xmatch = xmatch_per_candidate[149053599892210341]

    assert xmatch["id"] == "J235957.40+341229.0"
    assert xmatch["ra"] == 359.98920110000006
    assert xmatch["dec"] == 34.2080776
    assert xmatch["w1mpro"] == 12.257
    assert xmatch["w1sigmpro"] == 0.023
    assert xmatch["w2mpro"] == 12.175999999999998
    assert xmatch["w2sigmpro"] == 0.023
    assert xmatch["w3mpro"] == 11.832
    assert xmatch["w3sigmpro"] == 0.258
    assert xmatch["w4mpro"] == 8.606
    assert xmatch["w4sigmpro"] == 0.389


def test_galex_xmatch():
    xmatch_per_candidate = query_galex(
        k, [candidate.id], [candidate.ra], [candidate.dec], 5.0
    )
    assert 149053599892210341 in xmatch_per_candidate

    xmatch = xmatch_per_candidate[149053599892210341]

    assert xmatch["id"] == "GALEX J235957.3+341227"
    assert xmatch["ra"] == 359.989152
    assert xmatch["dec"] == 34.207653
    assert xmatch["b"] == 1
    assert xmatch["NUVmag"] == 22.1035
    assert xmatch["e_NUVmag"] == 0.3238
