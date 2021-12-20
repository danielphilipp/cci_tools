import h5py
import matplotlib.pyplot as plt
import numpy as np
import dask.array as da
from cci_tools.helperfuncs import Statistics
from atrain_plot.get_amsr2 import read_amsr_lwp
import time


def run_validation(ifile, ofile):
    if ifile is not None:
        X, Y, _, _ = read_amsr_lwp(ifile, 'CCI')

        # print('Mean CFC: ', np.sum(cma)/cma.shape[0])
        N = X.shape[0]  # lwp_amsr.shape[0]
        bias = np.mean(Y - X)
        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        std = np.sqrt(1 / N * np.sum((Y - X - bias) ** 2))
        rms = np.sqrt(1 / N * np.sum((Y - X) ** 2))

        r, p = Statistics.correlation(X, Y)
        ann = 'Corr    : {:.2f}\n' + \
              'RMSE    : {:.2f}\n' + \
              'bc-RMSE : {:.2f}\n' + \
              'Mean AMSR2: {:.2f}\n' + \
              'Mean Imager: {:.2f}\n' + \
              'Bias    : {:.2f}\n' + \
              'N       : {:d}'
        ann = ann.format(r, rms, std, mean_X, mean_Y, bias, N)
        return ann
    else:
        return None

if __name__ == '__main__':

    #ifile = '/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/' \
    #        'Reshaped_Files/proc9/msg4/5km/2019/07/' \
    #        '5km_msg4_20190731_1430_99999_amsr_avhrr__match.h5'
    ifile = '/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files_merged_amsr/proc9/2019/07/' \
            '5km_msg4_201907_0000_99999_amsr_avhrr_match_merged.h5'
    run_validation(ifile)
