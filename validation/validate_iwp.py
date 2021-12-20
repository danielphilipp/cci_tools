import h5py
import numpy as np
from cci_tools.helperfuncs import Statistics
from atrain_plot.get_dardar import read_dardar_iwp

def run_validation(ifile, ofile):
    if ifile is not None:
        X, Y, _, _ = read_dardar_iwp(ifile, 'CCI')

        N = X.shape[0]
        bias = np.mean(Y - X)
        mean_X = np.mean(X)
        mean_Y = np.mean(Y)
        std = np.sqrt(1 / N * np.sum((Y - X - bias) ** 2))
        rms = np.sqrt(1 / N * np.sum((Y - X) ** 2))

        r, p = Statistics.correlation(X, Y)
        ann = 'Corr    : {:.2f}\n' + \
              'RMSE    : {:.2f}\n' + \
              'bc-RMSE : {:.2f}\n' + \
              'Mean DARDAR: {:.2f}\n' + \
              'Mean Imager: {:.2f}\n' + \
              'Bias    : {:.2f}\n' + \
              'N       : {:d}'
        ann = ann.format(r, rms, std, mean_X, mean_Y, bias, N)
        print(ann)
        return ann
    else:
        return None


if __name__ == '__main__':

    #filename = '/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files/proc1/msg2' \
    #           '/5km/2019/02/dardar/5km_msg2_20190201_113000_99999_dardar_avhrr__match.h5'

    filename='/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files_merged_dardar/' \
             'proc1/2019/02/5km_msg_201902_0000_99999_dardar_avhrr_match_merged_final.h5'
    run_validation(filename, None)
