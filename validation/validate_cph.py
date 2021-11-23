import h5py
import numpy as np
import argparse
from atrain_match.statistics.scores import ScoreUtils
from atrain_plot.get_imager import get_imager_cph


CALIPSO_QUAL_VALUES = dict(none=0,
                           low=1,
                           medium=2,
                           high=3)

def get_imager_cma(ds):
    """ Get imager CMA. """
    data = np.array(ds['cloudmask'])
    return data.astype(bool)


def _get_cflag(flag, idx1, idx2):
    """ Get cloud flag value from binary flag. """

    flag_int = int(flag)
    flag_len = flag_int.bit_length()
    flag_bin = _intToBin(flag)
    return int(flag_bin[flag_len - idx1:flag_len - idx2], 2)


def _intToBin(x):
    """ Convert integer to binary. """

    return str(bin(x)[2:])


def run_validation(matchup_file, cot_limits, satz_lim):
    if matchup_file is not None:
        # read data
        file = h5py.File(matchup_file, 'r')
        imager_cph = get_imager_cph(file['cci'])
        imager_satz = file['cci']['satz'][:]
        imager_cma = get_imager_cma(file['cci'])
        calipso_cfc = file['calipso']['cloud_fraction'][:]
        laser_energy = file['calipso']['minimum_laser_energy_532'][:]

        from CythonExtension import parse_cal_profiles as pcp
        results = pcp.parse_profile_cph(file['calipso']['feature_classification_flags'][:],
                                        laser_energy.astype(np.float32),
                                        np.array(cot_limits, dtype=np.float32),
                                        file['calipso']['feature_optical_depth_532'][:].astype(np.float32),
                                        imager_satz.astype(np.float32),
                                        calipso_cfc.astype(np.float32),
                                        imager_cma.astype(np.float32),
                                        imager_cph.astype(np.float32),
                                        satz_lim)

        cph_cal, cph_img = results
        cph_cal = np.where(cph_cal < 0, np.nan, cph_cal)
        cph_img = np.where(cph_img < 0, np.nan, cph_img)

        phasedict = {}

        res = '\nSATZ_LIM: {sza_lim:.1f}\n'

        # calculate measures and print results
        for cnt, l in enumerate(cot_limits):
            nn = 'LIM' + str(cnt + 1)

            phasedict[nn] = {'IMG': cph_img[cnt, :],
                             'CAL': cph_cal[cnt, :]}
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
    else:
        return None
