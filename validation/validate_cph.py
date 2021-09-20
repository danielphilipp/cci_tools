import h5py
import numpy as np
import argparse
from atrain_match.statistics.scores import ScoreUtils


CALIPSO_QUAL_VALUES = dict(none=0,
                           low=1,
                           medium=2,
                           high=3)

def get_imager_cma(ds):
    """ Get imager CMA. """
    data = np.array(ds['cloudmask'])
    return data.astype(bool)


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


def _intToBin(x):
    """ Convert integer to binary. """

    return str(bin(x)[2:])


def setup_results_dict(n_limits):
    p = dict()
    for i in range(n_limits):
        lname = 'LIM' + str(i + 1)
        p[lname] = {'IMG': list(), 'CAL': list()}
    return p


def list_to_numpy(d, n_limits):
    # convert lists to numpy arrays
    for k in range(n_limits):
        lname = 'LIM' + str(k+1)
        d[lname]['IMG'] = np.array(d[lname]['IMG'])
        d[lname]['CAL'] = np.array(d[lname]['CAL'])

    return d


def run_validation(matchup_file, cot_limits, satz_lim):

    # read data
    file = h5py.File(matchup_file, 'r')
    imager_cph = get_imager_cph(file['cci'])
    imager_satz = file['cci']['satz'][:]
    imager_cma = get_imager_cma(file['cci'])
    calipso_cfc = file['calipso']['cloud_fraction'][:]
    laser_energy = file['calipso']['minimum_laser_energy_532'][:]

    # setup variables
    n_profiles = imager_cph.shape[0]
    n_levels = 10
    n_limits = len(cot_limits)

    # setup results dictionaries
    phasedict = setup_results_dict(n_limits)

    # iterate over profiles
    for idx in range(n_profiles):
        if laser_energy[idx] > 0.08:
            if imager_satz[idx] < satz_lim:
                # if imager has cloud
                if ~np.isnan(imager_cph[idx]) and imager_cma[idx] > 0.5 and calipso_cfc[idx] > 0.5:
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
                                cot_sum += file['calipso']['feature_optical_depth_532'][idx, lev]
                                # if integrated COT exceeds limit
                                if cot_sum >= cot_limits[lim] and phase_qual >= CALIPSO_QUAL_VALUES['medium']:
                                    phasedict[lname]['IMG'].append(imager_cph[idx])
                                    phasedict[lname]['CAL'].append(phase)
                                    break

    phasedict = list_to_numpy(phasedict, n_limits)

    res = '\nSATZ_LIM: {sza_lim:.1f}\n'

    # calculate measures and print results
    for cnt, l in enumerate(cot_limits):
        nn = 'LIM' + str(cnt + 1)
        x = phasedict[nn]['CAL']
        y = phasedict[nn]['IMG']

        xx = np.ones(x.size) * np.nan
        xx[np.logical_or(x == 1, x == 3)] = 1
        xx[x == 2] = 0

        mask = np.logical_or(np.isnan(xx), np.isnan(y))
        xx = xx[~mask]
        y = y[~mask]

        hits = np.sum(np.logical_and(xx == 1, y == 1))
        misses = np.sum(np.logical_and(xx == 1, y == 0))
        false_alarms = np.sum(np.logical_and(xx == 0, y == 1))
        correct_negatives = np.sum(np.logical_and(xx == 0, y == 0))

        scu = ScoreUtils(hits, false_alarms, misses, correct_negatives)

        res += '\n------------ COT = {limit:.2f}------------\n' + \
               'Hitrate: {hitrate:.2f}\n' + \
               'POD liq: {pod_clr:.2f}\n' + \
               'POD ice: {pod_cld:.2f}\n' + \
               'FAR liq: {far_clr:.2f}\n' + \
               'FAR ice: {far_cld:.2f}\n' + \
               'Heidke: {heidke:.2f}\n' + \
               'Kuiper: {kuiper:.2f}\n' + \
               'Bias: {bias:.3f}\n' + \
               'Num: {N:d}\n\n' + \
               'Hits: {hits:d}\n' + \
               'False Alarms: {fa:d}\n' + \
               'Misses: {misses:d}\n' + \
               'Correct Negatives: {cn:d}\n' +\
               'Imager ice cloud fraction: {img_icf:.2f}\n' + \
               'Reference ice cloud fraction: {ref_icf:.2f}\n'

        res = res.format(hitrate=scu.hitrate(), pod_clr=scu.pod_0(), pod_cld=scu.pod_1(),
                         far_clr=scu.far_0(), far_cld=scu.far_1(), heidke=scu.heidke(),
                         kuiper=scu.kuiper(), bias=scu.bias(), N=scu.n, hits=scu.a, fa=scu.b,
                         misses=scu.c, cn=scu.d, limit=l, sza_lim=satz_lim,
                         img_icf=(hits+false_alarms)/(hits+false_alarms+misses+correct_negatives),
                         ref_icf=(hits+misses)/(hits+false_alarms+misses+correct_negatives))

    return res
    #print(res)
