import h5py
import numpy as np
from atrain_match.statistics.scores import ScoreUtils


def get_imager_cma(ds):
    """ Get imager CMA. """
    data = np.array(ds['cloudmask'])
    return data.astype(bool)


def run_validation(matchup_file, cot_limits, satz_lim):

    # read data
    file = h5py.File(matchup_file, 'r')
    caliop_column_cot = file['calipso']['column_optical_depth_cloud_532'][:]
    caliop_cfc =  file['calipso']['cloud_fraction'][:]
    imager_cma = get_imager_cma(file['cci'])
    imager_satz = file['cci']['satz'][:]
    laser_energy = file['calipso']['minimum_laser_energy_532'][:]

    res  = '\nSATZ_LIM: {sza_lim:.1f}\n'

    for cnt, limit in enumerate(cot_limits):

        cal_cma_tmp = np.where(np.logical_and(caliop_cfc >= 0.5, caliop_column_cot >= limit), 1, 0)

        satz_mask = imager_satz > satz_lim
        cal_cma_tmp = np.where(satz_mask, np.nan, cal_cma_tmp)
        imager_cma_tmp = np.where(satz_mask, np.nan, imager_cma)

        laser_energy_mask = laser_energy < 0.08
        cal_cma_tmp = np.where(laser_energy_mask, np.nan, cal_cma_tmp)
        imager_cma_tmp = np.where(laser_energy_mask, np.nan, imager_cma_tmp)

        mask = np.logical_or(np.isnan(cal_cma_tmp), np.isnan(imager_cma_tmp))
        cal_cma_tmp = cal_cma_tmp[~mask]
        imager_cma_tmp = imager_cma_tmp[~mask]

        assert cal_cma_tmp.shape[0] == imager_cma_tmp.shape[0]

        x = cal_cma_tmp
        y = imager_cma_tmp

        hits = np.sum(np.logical_and(x == 1, y == 1))
        misses = np.sum(np.logical_and(x == 1, y == 0))
        false_alarms = np.sum(np.logical_and(x == 0, y == 1))
        correct_negatives = np.sum(np.logical_and(x == 0, y == 0))

        scu = ScoreUtils(hits, false_alarms, misses, correct_negatives)

        res += '\n------------ COT = {limit:.2f}------------\n' +\
              'Hitrate: {hitrate:.2f}\n' +\
              'POD clr: {pod_clr:.2f}\n' +\
              'POD cld: {pod_cld:.2f}\n' +\
              'FAR clr: {far_clr:.2f}\n' +\
              'FAR cld: {far_cld:.2f}\n' +\
              'Heidke: {heidke:.2f}\n' +\
              'Kuiper: {kuiper:.2f}\n' +\
              'Bias: {bias:.3f}\n' +\
              'Num: {N:d}\n\n' +\
              'Hits: {hits:d}\n' +\
              'False Alarms: {fa:d}\n' +\
              'Misses: {misses:d}\n' +\
              'Correct Negatives: {cn:d}\n' + \
              'Imager cloud fraction: {img_cf:.2f}\n' + \
              'Reference cloud fraction: {ref_cf:.2f}\n'

        res = res.format(hitrate=scu.hitrate(), pod_clr=scu.pod_0(), pod_cld=scu.pod_1(),
                         far_clr=scu.far_0(), far_cld=scu.far_1(), heidke=scu.heidke(),
                         kuiper=scu.kuiper(), bias=scu.bias(), N=scu.n, hits=scu.a, fa=scu.b,
                         misses=scu.c, cn=scu.d, limit=limit, sza_lim=satz_lim,
                         img_cf=(hits+false_alarms)/(hits+false_alarms+misses+correct_negatives),
                         ref_cf=(hits+misses)/(hits+false_alarms+misses+correct_negatives))
        
    return res

    #print(res)

