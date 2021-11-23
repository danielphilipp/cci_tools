import h5py
import numpy as np
from cci_tools.helperfuncs import Statistics
from atrain_plot.get_dardar import read_dardar_iwp

def run_validation(ifile, ofile):
    if ifile is not None:
        X, Y, _, _ = read_dardar_iwp(ifile, 'CCI')

        # print('Mean CFC: ', np.sum(cma)/cma.shape[0])
        N = X.shape[0]  # lwp_amsr.shape[0]
        bias = np.mean(Y - X)
        std = np.sqrt(1 / N * np.sum((Y - X - bias) ** 2))
        rms = np.sqrt(1 / N * np.sum((Y - X) ** 2))

        r, p = Statistics.correlation(X, Y)
        ann = 'Corr    : {:.2f}\n' + \
              'RMSE    : {:.2f}\n' + \
              'bc-RMSE : {:.2f}\n' + \
              'Bias    : {:.2f}\n' + \
              'N       : {:d}'
        ann = ann.format(r, rms, std, bias, N)
        print(ann)
        return ann
    else:
        return None


if __name__ == '__main__':

    filename = '/cmsaf/cmsaf-cld1/dphilipp/atrain_match_results/Reshaped_Files/proc1/msg2' \
               '/5km/2019/02/dardar/5km_msg2_20190201_113000_99999_dardar_avhrr__match.h5'

    run_validation(filename, None)
