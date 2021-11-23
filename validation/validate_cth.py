import h5py
import numpy as np
import argparse
from atrain_plot.get_imager import get_imager_cph


ICE_TEMPERATURE_LIMIT = -20
CALIPSO_QUAL_VALUES = dict(none=0,
                           low=1,
                           medium=2,
                           high=3)

def bias(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    return np.nanmean(x_img - x_cal)


def mae(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    return np.nanmean(np.abs(x_cal - x_img))


def rmse(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    return np.sqrt(np.nanmean((x_cal - x_img) * (x_cal - x_img)))


def std(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    mu = np.nanmean(x_cal)
    return np.sqrt(np.nanmean((x_img - mu) * (x_img - mu)))


def std2(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    mu = np.nanmean(x_img)
    return np.sqrt(np.nanmean((x_cal - mu) * (x_cal - mu)))

def get_imager_cth(ds):
    """ Get imager CTH. """
    alti = np.array(ds['ctth_height'])
    # set FillValue to NaN
    alti = np.where(alti < 0, np.nan, alti)
    # alti = np.where(alti>45000, np.nan, alti)
    return alti

def get_caliop_cth(ds):
    """Get CALIOP CTH."""
    cth = np.array(ds['layer_top_altitude'])
    #elev = np.array(ds['elevation'])
    # set FillValue to NaN, convert to m
    cth = np.where(cth == -9999, np.nan, cth * 1000.)
    cth = np.where(cth < 0, np.nan, cth)
    # compute height above surface
    #cth_surf = cth - elev
    #return cth_surf
    return cth


def _get_cflag(flag, idx1, idx2):
    """ Get cloud flag value from binary flag. """
    flag_int = int(flag)
    flag_len = flag_int.bit_length()
    flag_bin = _intToBin(flag)
    return int(flag_bin[flag_len - idx1:flag_len - idx2], 2)


def get_phase_from_temperature(mlt):
    if mlt < ICE_TEMPERATURE_LIMIT:
        phase = 1
    else:
        phase = 2

    if mlt <= -100:
        phase = 0
    return phase


def _intToBin(x):
    """ Convert integer to binary. """

    return str(bin(x)[2:])


def run_validation(matchup_file, cot_limits, satz_lim):
    if matchup_file is not None:
        # read data
        file = h5py.File(matchup_file, 'r')
        caliop_cth = get_caliop_cth(file['calipso'])
        imager_cth = get_imager_cth(file['cci'])
        imager_cma = file['cci']['cloudmask'][:]
        calipso_cfc = file['calipso']['cloud_fraction'][:]
        imager_satz = file['cci']['satz'][:]
        imager_cph = get_imager_cph(file['cci'])
        laser_energy = file['calipso']['minimum_laser_energy_532'][:]

        from CythonExtension import parse_cal_profiles as pcp
        results = pcp.parse_profile_cth(file['calipso']['feature_classification_flags'][:],
                                        laser_energy.astype(np.float32),
                                        np.array(cot_limits, dtype=np.float32),
                                        file['calipso']['feature_optical_depth_532'][:].astype(np.float32),
                                        caliop_cth.astype(np.float32),
                                        imager_satz.astype(np.float32),
                                        calipso_cfc.astype(np.float32),
                                        imager_cma.astype(np.float32),
                                        imager_cth.astype(np.float32),
                                        imager_cph.astype(np.float32),
                                        satz_lim)

        cth_cal_all, cth_cal_ice, cth_cal_wat, cth_img_all, cth_img_ice, cth_img_wat = results
        cth_cal_all = np.where(cth_cal_all < 0, np.nan, cth_cal_all)
        cth_cal_ice = np.where(cth_cal_ice < 0, np.nan, cth_cal_ice)
        cth_cal_wat = np.where(cth_cal_wat < 0, np.nan, cth_cal_wat)
        cth_img_all = np.where(cth_img_all < 0, np.nan, cth_img_all)
        cth_img_ice = np.where(cth_img_ice < 0, np.nan, cth_img_ice)
        cth_img_wat = np.where(cth_img_wat < 0, np.nan, cth_img_wat)

        vals_all = {}
        vals_liq = {}
        vals_ice = {}

        res = '\nSATZ_LIM: {:.1f}\n'.format(satz_lim)

        # calculate measures and print results
        for cnt, l in enumerate(cot_limits):
            n = 'COT > {}'.format(l)
            nn = 'LIM' + str(cnt+1)

            vals_all[nn] = {'IMG': cth_img_all[cnt, :],
                            'CAL': cth_cal_all[cnt, :]}

            vals_ice[nn] = {'IMG': cth_img_ice[cnt, :],
                            'CAL': cth_cal_ice[cnt, :]}

            vals_liq[nn] = {'IMG': cth_img_wat[cnt, :],
                            'CAL': cth_cal_wat[cnt, :]}

            res += '\n------------ COT = {:.2f}------------\n'.format(l)
            res += '    BIAS\n'
            res += '        ALL {:.3f}\n'.format(bias(vals_all[nn]['IMG'], vals_all[nn]['CAL'])/1000)
            res += '        LIQ {:.3f}\n'.format(bias(vals_liq[nn]['IMG'], vals_liq[nn]['CAL'])/1000)
            res += '        ICE {:.3f}\n'.format(bias(vals_ice[nn]['IMG'], vals_ice[nn]['CAL'])/1000)

            res += '    MAE\n'
            res += '        ALL {:.3f}\n'.format(mae(vals_all[nn]['IMG'], vals_all[nn]['CAL'])/1000)
            res += '        LIQ {:.3f}\n'.format(mae(vals_liq[nn]['IMG'], vals_liq[nn]['CAL'])/1000)
            res += '        ICE {:.3f}\n'.format(mae(vals_ice[nn]['IMG'], vals_ice[nn]['CAL'])/1000)

            res += '    RMSE\n'
            res += '        ALL {:.3f}\n'.format(rmse(vals_all[nn]['IMG'], vals_all[nn]['CAL'])/1000)
            res += '        LIQ {:.3f}\n'.format(rmse(vals_liq[nn]['IMG'], vals_liq[nn]['CAL'])/1000)
            res += '        ICE {:.3f}\n'.format(rmse(vals_ice[nn]['IMG'], vals_ice[nn]['CAL'])/1000)

            res += '    STD (img - mean(cal))\n'
            res += '        ALL {:.3f}\n'.format(std(vals_all[nn]['IMG'], vals_all[nn]['CAL'])/1000)
            res += '        LIQ {:.3f}\n'.format(std(vals_liq[nn]['IMG'], vals_liq[nn]['CAL'])/1000)
            res += '        ICE {:.3f}\n'.format(std(vals_ice[nn]['IMG'], vals_ice[nn]['CAL'])/1000)

            res += '    STD2 TEST (cal - mean(img)\n'
            res += '        ALL {:.3f}\n'.format(std2(vals_all[nn]['IMG'], vals_all[nn]['CAL'])/1000)
            res += '        LIQ {:.3f}\n'.format(std2(vals_liq[nn]['IMG'], vals_liq[nn]['CAL'])/1000)
            res += '        ICE {:.3f}\n'.format(std2(vals_ice[nn]['IMG'], vals_ice[nn]['CAL'])/1000)

            res += '    N\n'
            res += '        ALL {:.3f}\n'.format(vals_all[nn]['IMG'].shape[0])
            res += '        LIQ {:.3f}\n'.format(vals_liq[nn]['IMG'].shape[0])
            res += '        ICE {:.3f}\n'.format(vals_ice[nn]['IMG'].shape[0])

        return res
    else:
        return None