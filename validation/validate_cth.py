import h5py
import numpy as np
import argparse


ICE_TEMPERATURE_LIMIT = -20
CALIPSO_QUAL_VALUES = dict(none=0,
                           low=1,
                           medium=2,
                           high=3)

def bias(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    return np.mean(x_img - x_cal)


def mae(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    return np.mean(np.abs(x_cal - x_img))


def rmse(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    return np.sqrt(np.mean((x_cal - x_img) * (x_cal - x_img)))


def std(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    mu = np.mean(x_cal)
    return np.sqrt(np.mean((x_img - mu) * (x_img - mu)))


def std2(x_img, x_cal):
    assert x_img.shape[0] == x_cal.shape[0]
    mu = np.mean(x_img)
    return np.sqrt(np.mean((x_cal - mu) * (x_cal - mu)))

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


def get_imager_cph(ds):
    """ Get imager CPH. """
    phase = np.array(ds['cpp_phase'])
    phase = np.where(phase < 0, np.nan, phase)
    phase = np.where(phase > 10, np.nan, phase)
    phase = np.where(phase == 0, np.nan, phase)
    phase = np.where(phase == 1, 0, phase)
    phase = np.where(phase == 2, 1, phase)
    return phase


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


def setup_results_dict(n_limits):
    vals_all = dict()
    vals_ice = dict()
    vals_liq = dict()
    for i in range(n_limits):
        lname = 'LIM' + str(i + 1)
        vals_all[lname] = {'IMG': list(), 'CAL': list()}
        vals_ice[lname] = {'IMG': list(), 'CAL': list()}
        vals_liq[lname] = {'IMG': list(), 'CAL': list()}
    return vals_all, vals_ice, vals_liq


def list_to_numpy(vals_all, vals_ice, vals_liq, n_limits):
    # convert lists to numpy arrays
    for k in range(n_limits):
        lname = 'LIM' + str(k+1)
        vals_all[lname]['IMG'] = np.array(vals_all[lname]['IMG'])
        vals_all[lname]['CAL'] = np.array(vals_all[lname]['CAL'])

        vals_ice[lname]['IMG'] = np.array(vals_ice[lname]['IMG'])
        vals_ice[lname]['CAL'] = np.array(vals_ice[lname]['CAL'])

        vals_liq[lname]['IMG'] = np.array(vals_liq[lname]['IMG'])
        vals_liq[lname]['CAL'] = np.array(vals_liq[lname]['CAL'])

    return vals_all, vals_ice, vals_liq


def run_validation(matchup_file, cot_limits, satz_lim):

    # read data
    file = h5py.File(matchup_file, 'r')
    caliop_cth = get_caliop_cth(file['calipso'])
    imager_cth = get_imager_cth(file['cci'])
    imager_cma = file['cci']['cloudmask'][:]
    calipso_cfc = file['calipso']['cloud_fraction'][:]
    imager_satz = file['cci']['satz'][:]
    laser_energy = file['calipso']['minimum_laser_energy_532'][:]

    # setup variables
    n_profiles = imager_cth.shape[0]
    n_levels = caliop_cth.shape[1]
    n_limits = len(cot_limits)

    # setup results dictionaries
    vals_all, vals_ice, vals_liq = setup_results_dict(n_limits)

    # iterate over profiles
    for idx in range(n_profiles):
        if laser_energy[idx] > 0.08:
            if imager_satz[idx] < satz_lim:
                # if imager has cloud
                if ~np.isnan(imager_cth[idx]) and imager_cma[idx] > 0.5 and calipso_cfc[idx] > 0.5:
                    # iterate over COT limits
                    for lim in range(n_limits):
                        cot_sum = 0
                        lname = 'LIM' + str(lim+1)
                        # iterate over vertical levels
                        for lev in range(n_levels):
                            cal_fflag = file['calipso']['feature_classification_flags'][idx, lev]
                            if cal_fflag != 1:
                                cflag = _get_cflag(cal_fflag, 3, 0)
                                phase = _get_cflag(cal_fflag, 7, 5)
                                phase_qual = _get_cflag(cal_fflag, 9, 7)
                            else:
                                cflag = -999
                                phase = -999
                                phase_qual = -999

                            # if cloud
                            if cflag == 2:
                                #if phase == 0:
                                    #break
                                    #phase = get_phase_from_temperature(caliop_mlt[idx, lev])

                                cot_sum += file['calipso']['feature_optical_depth_532'][idx, lev]

                                # if integrated COT exceeds limit
                                if cot_sum >= cot_limits[lim]:
                                    cal_cth = caliop_cth[idx, lev]
                                    vals_all[lname]['IMG'].append(imager_cth[idx])
                                    vals_all[lname]['CAL'].append(cal_cth)


                                    if phase_qual >= CALIPSO_QUAL_VALUES['medium']:
                                        # if ice phase
                                        if phase == 1 or phase == 3:
                                            vals_ice[lname]['IMG'].append(imager_cth[idx])
                                            vals_ice[lname]['CAL'].append(cal_cth)
                                        # if liquid phase
                                        elif phase == 2:
                                            vals_liq[lname]['IMG'].append(imager_cth[idx])
                                            vals_liq[lname]['CAL'].append(cal_cth)
                                    break

    vals_all, vals_ice, vals_liq = list_to_numpy(vals_all, vals_ice,
                                                 vals_liq, n_limits)

    res = '\nSATZ_LIM: {:.1f}\n'.format(satz_lim)

    # calculate measures and print results
    for cnt, l in enumerate(cot_limits):
        n = 'COT > {}'.format(l)
        nn = 'LIM' + str(cnt+1)

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