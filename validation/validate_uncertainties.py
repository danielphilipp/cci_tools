import argparse
import numpy as np
from CythonExtension import parse_cal_profiles as pcp
from atrain_plot.get_imager import get_imager_cph
from atrain_plot.get_dardar import read_dardar_iwp
from atrain_plot.get_amsr2 import read_amsr_lwp
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import h5py
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr
import os

class GetImagerCaliop:
    @staticmethod
    def get_imager_ctp(ds):
        """ Get imager CTP. """
        ctp = np.array(ds['ctth_pressure'])
        # set FillValue to NaN
        ctp = np.where(ctp < 0, np.nan, ctp)
        ctp = np.where(ctp > 1100, np.nan, ctp)
        return ctp

    @staticmethod
    def get_caliop_ctp(ds):
        """Get CALIOP CTP."""
        ctp = np.array(ds['layer_top_pressure'])
        ctp = np.where(ctp < 0, np.nan, ctp)
        return ctp

    @staticmethod
    def get_imager_cth(ds):
        """ Get imager CTP. """
        cth = np.array(ds['ctth_height'])
        # set FillValue to NaN
        cth = np.where(cth < 0, np.nan, cth/1000)
        #cth = np.where(cth > 40, np.nan, cth)
        return cth

    @staticmethod
    def get_caliop_cth(ds):
        """Get CALIOP CTP."""
        cth = np.array(ds['layer_top_altitude'])
        cth = np.where(cth < 0, np.nan, cth)
        return cth

    @staticmethod
    def get_imager_ctt(ds):
        """ Get imager CTT. """
        temp = np.array(ds['ctth_temperature'])
        # set FillValue to NaN
        temp = np.where(temp < 0, np.nan, temp)
        return temp

    @staticmethod
    def get_caliop_ctt(ds):
        """Get CALIOP CTT."""
        temp = np.array(ds['layer_top_temperature'])
        temp = np.where(temp < -9000, np.nan, temp+273.15)
        temp = np.where(temp < 0, np.nan, temp)
        return temp


class BetaRatio:
    def __init__(self, data, uncertainty):
        self.data = data
        self.uncertainty = uncertainty

        self.phases =value.keys()
        self.cot_lims = value[list(self.phases)[0]].keys()

        self.beta = {}
        self.d = {}
        self.u = {}

    def calculate(self):
        """ Calculate d_i, u_i and beta_i for all phases and COT limits. """
        for p in self.phases:
            self.beta[p] = {}
            self.u[p] = {}
            self.d[p] = {}
            for c in self.cot_lims:
                data_img = self.data[p][c]['IMG']
                data_ref = self.data[p][c]['REF']
                unc_img = self.uncertainty[p][c]['IMG']
                unc_ref = self.uncertainty[p][c]['REF']
                self.d[p][c] = self.get_d_i(data_img, data_ref)
                self.u[p][c] = self.get_u_i(unc_img, unc_ref)
                self.beta[p][c] = self.get_b_i(self.d[p][c], self.u[p][c])

    def get_d_i(self, img, ref):
        """ Calculate d_i as CCI - REF """
        return img - ref

    def get_u_i(self, img_unc, ref_unc):
        """ Calculate u_i as SQRT(CCI_UNC^2 + REF_UNC^2) """
        return np.sqrt(img_unc**2 + ref_unc**2)

    def get_b_i(self, d, u):
        """" Calculate beta_i as |d_i| / u_i """
        return np.abs(d) / u


class ProcessorCalipso:
    def __init__(self, matchup_file_calipso, satz_limit=70, cot_list=[0, 0.1, 0.15, 0.2, 1.0]):
        self.filename = matchup_file_calipso
        self.satz_limit = satz_limit
        self.cot_list = np.array(cot_list, dtype=np.float32)

        # constants
        self.CTP_UNC = 10    # hPa
        self.CTH_UNC = 0.05  # km
        self.CTT_UNC = 3     # K
        self.truth = 'CALIPSO'

        # read data
        self.file = h5py.File(self.filename, 'r')

        self.ctp_results = None
        self.ctp_unc_results = None

        self.cth_results = None
        self.cth_unc_results = None

        self.ctt_results = None
        self.ctt_unc_results = None

    def read_ctp(self):
        d, u = self._read_ctx('CTP')
        self.ctp_results = d
        self.ctp_unc_results = u

    def read_ctt(self):
        d, u = self._read_ctx('CTT')
        self.ctt_results = d
        self.ctt_unc_results = u

    def read_cth(self):
        d, u = self._read_ctx('CTH')
        self.cth_results = d
        self.cth_unc_results = u

    def _read_ctx(self, ctx_var):

        if ctx_var == 'CTP':
            calipso_ctx = GetImagerCaliop.get_caliop_ctp(self.file['calipso']).astype(np.float32)
            imager_ctx = GetImagerCaliop.get_imager_ctp(self.file['cci']).astype(np.float32)
            imager_ctx_unc = self.file['cci']['ctp_unc'][:].astype(np.float32)
            imager_ctx_unc[imager_ctx_unc > 1000] = np.nan
            CTX_UNC = self.CTP_UNC
        elif ctx_var == 'CTH':
            calipso_ctx = GetImagerCaliop.get_caliop_cth(self.file['calipso']).astype(np.float32)
            imager_ctx = GetImagerCaliop.get_imager_cth(self.file['cci']).astype(np.float32)
            imager_ctx_unc = self.file['cci']['cth_unc'][:].astype(np.float32)
            CTX_UNC = self.CTH_UNC
        elif ctx_var == 'CTT':
            calipso_ctx = GetImagerCaliop.get_caliop_ctt(self.file['calipso']).astype(np.float32)
            imager_ctx = GetImagerCaliop.get_imager_ctt(self.file['cci']).astype(np.float32)
            imager_ctx_unc = self.file['cci']['ctt_unc'][:].astype(np.float32)
            CTX_UNC = self.CTT_UNC

        # calipso data
        calipso_cfc = self.file['calipso']['cloud_fraction'][:].astype(np.float32)
        calipso_laser_energy = self.file['calipso']['minimum_laser_energy_532'][:].astype(np.float32)
        calipso_flags = self.file['calipso']['feature_classification_flags'][:]
        calipso_cot = self.file['calipso']['feature_optical_depth_532'][:].astype(np.float32)

        calipso_ctx_unc = np.ones(calipso_ctx.shape, dtype=np.float32) * CTX_UNC
        calipso_ctx_unc = np.where(np.isnan(calipso_ctx), np.nan, calipso_ctx_unc)

        # imager data
        imager_qual = self.file['cci']['cpp_quality'][:].astype(int)
        imager_cma = self.file['cci']['cloudmask'][:].astype(np.float32)
        imager_satz = self.file['cci']['satz'][:].astype(np.float32)
        imager_cph = get_imager_cph(self.file['cci']).astype(np.float32)

        bad_quality_imager = np.bitwise_and(imager_qual, 3) > 0
        imager_ctx[bad_quality_imager] = np.nan

        col_var_img = imager_ctx
        col_var_ref = calipso_ctx

        results = pcp.parse_profile_ctx(calipso_flags,
                                        calipso_laser_energy,
                                        self.cot_list,
                                        calipso_cot,
                                        col_var_ref,
                                        imager_satz,
                                        calipso_cfc,
                                        imager_cma,
                                        col_var_img,
                                        imager_cph,
                                        self.satz_limit)

        ctx_cal_all, ctx_cal_ice, ctx_cal_wat, ctx_img_all, ctx_img_ice, ctx_img_wat = results
        ctx_cal_all = np.where(ctx_cal_all < 0, np.nan, ctx_cal_all)
        ctx_cal_ice = np.where(ctx_cal_ice < 0, np.nan, ctx_cal_ice)
        ctx_cal_wat = np.where(ctx_cal_wat < 0, np.nan, ctx_cal_wat)
        ctx_img_all = np.where(ctx_img_all < 0, np.nan, ctx_img_all)
        ctx_img_ice = np.where(ctx_img_ice < 0, np.nan, ctx_img_ice)
        ctx_img_wat = np.where(ctx_img_wat < 0, np.nan, ctx_img_wat)

        ctx_results_all = {}
        ctx_results_liq = {}
        ctx_results_ice = {}

        for cnt, l in enumerate(self.cot_list):
            nn = 'COT-{:.2f}'.format(l)

            ctx_results_all[nn] = {'IMG': ctx_img_all[cnt, :],
                                   'REF': ctx_cal_all[cnt, :]}

            ctx_results_ice[nn] = {'IMG': ctx_img_ice[cnt, :],
                                   'REF': ctx_cal_ice[cnt, :]}

            ctx_results_liq[nn] = {'IMG': ctx_img_wat[cnt, :],
                                   'REF': ctx_cal_wat[cnt, :]}

        ctx_results = {'ALL': ctx_results_all, 'ICE': ctx_results_ice, 'LIQ': ctx_results_liq}


        col_var_img = imager_ctx_unc
        col_var_ref = calipso_ctx_unc

        results = pcp.parse_profile_ctx(calipso_flags,
                                        calipso_laser_energy,
                                        self.cot_list,
                                        calipso_cot,
                                        col_var_ref,
                                        imager_satz,
                                        calipso_cfc,
                                        imager_cma,
                                        col_var_img,
                                        imager_cph,
                                        self.satz_limit)

        ctx_cal_all, ctx_cal_ice, ctx_cal_wat, ctx_img_all, ctx_img_ice, ctx_img_wat = results
        ctx_cal_all = np.where(ctx_cal_all < 0, np.nan, ctx_cal_all)
        ctx_cal_ice = np.where(ctx_cal_ice < 0, np.nan, ctx_cal_ice)
        ctx_cal_wat = np.where(ctx_cal_wat < 0, np.nan, ctx_cal_wat)
        ctx_img_all = np.where(ctx_img_all < 0, np.nan, ctx_img_all)
        ctx_img_ice = np.where(ctx_img_ice < 0, np.nan, ctx_img_ice)
        ctx_img_wat = np.where(ctx_img_wat < 0, np.nan, ctx_img_wat)

        ctx_results_all = {}
        ctx_results_liq = {}
        ctx_results_ice = {}

        for cnt, l in enumerate(self.cot_list):
            nn = 'COT-{:.2f}'.format(l)

            ctx_results_all[nn] = {'IMG': ctx_img_all[cnt, :],
                                   'REF': ctx_cal_all[cnt, :]}

            ctx_results_ice[nn] = {'IMG': ctx_img_ice[cnt, :],
                                   'REF': ctx_cal_ice[cnt, :]}

            ctx_results_liq[nn] = {'IMG': ctx_img_wat[cnt, :],
                                   'REF': ctx_cal_wat[cnt, :]}

        ctx_unc_results = {'ALL': ctx_results_all, 'ICE': ctx_results_ice, 'LIQ': ctx_results_liq}
        return ctx_results, ctx_unc_results


class ProcessorAMSR2:
    def __init__(self, matchup_file_amsr2):
        self.filename = matchup_file_amsr2


class ProcessorDARDAR:
    def __init__(self, matchup_file_dardar):
        self.filename = matchup_file_dardar

    def read_iwp(self):
        data = read_dardar_iwp(self.filename, 'CCI')


class Plotting:
    def __init__(self, values, uncertainty, beta, output_dir=None, variable='VAR', units='', truth='truth'):
        self.values = values
        self.uncertainty = uncertainty
        self.beta = beta
        self.units = units
        self.output_dir = output_dir
        self.variable = variable

        self.phases = values.keys()
        self.cot_lims = values[list(self.phases)[0]].keys()
        self.truth = truth

    def _remove_common_nans(self, a, b):
        """ Remove indices where a or b is NaN from a and b. """
        mask = np.logical_and(~np.isnan(a), ~np.isnan(b))
        return a[mask], b[mask]

    def _get_beta_portion_leq_1(self, data):
        """ For a single bin get fraction of elements satisfying beta <= 1. """
        N = data.shape[0]
        N_leq_1 = np.sum(data <= 1)
        return 100 * N_leq_1/N

    def _bin_data(self, X, Y, NBINS):
        """ Bin data to get list of arrays [bin1(elements), bin2(elements)] """
        bins = np.arange(0, 1050, NBINS)#np.linspace(0, 1050, NBINS)
        digitized = np.digitize(Y, bins)
        bin_collection = [Y[digitized == i] for i in range(1, len(bins))]
        return bin_collection, bins

    def plot_all(self):
        """ Make all plots. """
        self.plot_value_unc_box()
        self.plot_value_2d_hist()
        self.plot_beta_frequency()
        self.plot_beta_portion_distribution()

    def plot_value_unc_box(self):
        NBINS = 30
        for c in self.cot_lims:
            for p in self.phases:

                figname = '{}_{}_uncertainty_value_boxplot_{}_phase-{}.png'.format(self.variable, self.truth, c, p)
                figname = os.path.join(self.output_dir, figname)

                fig = plt.figure(figsize=(12, 3))
                
                ax = fig.add_subplot(1, 2, 1)
                X, Y = self._remove_common_nans(self.values[p][c]['IMG'],
                                                self.uncertainty[p][c]['IMG'])
                mean, edges, _ = binned_statistic(X, Y, 'mean', bins=NBINS)
                min, edges, _ = binned_statistic(X, Y, 'min', bins=NBINS)
                max, edges, _ = binned_statistic(X, Y, 'max', bins=NBINS)
                std, edges, _ = binned_statistic(X, Y, 'std', bins=NBINS)
                width=edges[1]-edges[0]
                ax.errorbar(x=edges[:-1]+width/2, y=mean, yerr=(min, max), ecolor='red', fmt='o', markersize=3, capsize=3)
                ax.errorbar(x=edges[:-1]+width/2, y=mean, yerr=std, ecolor='black', capsize=3)

                ax.set_xlabel('Cloud_cci {} [{}]'.format(self.variable, self.units))
                ax.set_ylabel('Cloud_cci {}_unc [{}]'.format(self.variable, self.units))
                ax.grid(ls='--', color='grey')
                
                ax = fig.add_subplot(1, 2, 2)
                X, Y = self._remove_common_nans(self.values[p][c]['REF'],
                                                self.uncertainty[p][c]['REF'])
                mean, edges, _ = binned_statistic(X, Y, 'mean', bins=NBINS)
                min, edges, _ = binned_statistic(X, Y, 'min', bins=NBINS)
                max, edges, _ = binned_statistic(X, Y, 'max', bins=NBINS)
                std, edges, _ = binned_statistic(X, Y, 'std', bins=NBINS)

                width = edges[1]-edges[0]
                ax.errorbar(x=edges[:-1]+width/2, y=mean, yerr=(min, max), ecolor='red', fmt='o', markersize=3, capsize=3)
                ax.errorbar(x=edges[:-1]+width/2, y=mean, yerr=std, ecolor='black', capsize=3)

                ax.set_xlabel('{} {} [{}]'.format(self.truth, self.variable, self.units))
                ax.set_ylabel('{} {}_unc [{}]'.format(self.truth, self.variable, self.units))
                ax.grid(ls='--', color='grey')
                plt.tight_layout()
                plt.savefig(figname)
                print(figname, '  saved')
                plt.close()

    def plot_value_2d_hist(self):
        BINS = (80, 80)
        for c in self.cot_lims:
            for p in self.phases:

                figname = '{}_{}_2d_hist_{}_phase-{}.png'.format(self.variable, self.truth, c, p)
                figname = os.path.join(self.output_dir, figname)

                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111)
                X, Y = self._remove_common_nans(self.values[p][c]['REF'],
                                                self.values[p][c]['IMG'])
                h = ax.hist2d(X, Y, bins=BINS, cmap=plt.get_cmap('jet'), norm=LogNorm(), vmin=1, vmax=1E3)
                ax.set_xlabel('{} {} [{}]'.format(self.truth, self.variable, self.units))
                ax.set_ylabel('Cloud_cci {} [{}]'.format(self.variable, self.units))

                plt.colorbar(h[3], ax=ax)
                plt.tight_layout()
                plt.savefig(figname)
                print(figname, '  saved')
                plt.close()

    def plot_beta_frequency(self):
        for c in self.cot_lims:
            for p in self.phases:

                figname = '{}_{}_beta_frequency_{}_phase-{}.png'.format(self.variable, self.truth, c, p)
                figname = os.path.join(self.output_dir, figname)

                tmp_beta = self.beta[p][c]

                fig = plt.figure(figsize=(8, 3))
                ax = fig.add_subplot(111)

                tmp_beta = tmp_beta[~np.isnan(tmp_beta)]
                N = tmp_beta.shape[0]
                h, edges, = np.histogram(tmp_beta, bins=40, weights=100*np.ones(tmp_beta.shape)/N, range=(0, 40))
                width = edges[1] - edges[0]
                ax.plot(edges[:-1]+width/2, h, marker='.', color='black')
                ax.grid(ls='--', color='grey')
                ax.set_xlabel(r'$\beta$-Ratio')
                ax.set_ylabel('Relative Frequency [%]')

                plt.tight_layout()
                plt.savefig(figname)
                print(figname, '  saved')
                plt.close()

    def plot_beta_portion_distribution(self):
        NBINS=15
        for c in self.cot_lims:
            for p in self.phases:

                figname = '{}_{}_beta_portion_{}_phase-{}.png'.format(self.variable, self.truth, c, p)
                figname = os.path.join(self.output_dir, figname)

                fig = plt.figure(figsize=(16, 6))

                # Cloud_cci ------------------------
                tmp_beta = self.beta[p][c]
                tmp_value = self.values[p][c]['IMG']

                ax = fig.add_subplot(221)
                X, Y = self._remove_common_nans(tmp_value, tmp_beta)
                beta_portion_leq_1, edges, _ = binned_statistic(X, Y, self._get_beta_portion_leq_1, bins=NBINS)
                width = edges[1]-edges[0]
                ax.plot(edges[:-1]+width/2, beta_portion_leq_1, marker='.', color='black')
                ax.grid(ls='--', color='grey')
                ax.set_xlabel('Cloud_cci {} [{}]'.format(self.variable, self.units))
                ax.set_ylabel(r'Portion of $\beta \leq$ 1 [%] ')

                ax = fig.add_subplot(222)
                tmp_value = tmp_value[~np.isnan(tmp_value)]
                h, edges, = np.histogram(tmp_value, bins=30)
                width = edges[1]-edges[0]
                ax.plot(edges[:-1]+width/2, h, marker='.', color='black')
                ax.grid(ls='--', color='grey')
                ax.set_xlabel('Cloud_cci {} [{}]'.format(self.variable, self.units))
                ax.set_ylabel('Number')
                ax.set_yscale('log')

                # REFERENCE ----------------------
                tmp_beta = self.beta[p][c]
                tmp_value = self.values[p][c]['REF']

                ax = fig.add_subplot(223)
                X, Y = self._remove_common_nans(tmp_value, tmp_beta)
                beta_portion_leq_1, edges, _ = binned_statistic(X, Y, self._get_beta_portion_leq_1, bins=NBINS)
                width = edges[1]-edges[0]
                ax.plot(edges[:-1]+width/2, beta_portion_leq_1, marker='.', color='black')
                ax.grid(ls='--', color='grey')
                ax.set_xlabel('{} {} [{}]'.format(self.truth, self.variable, self.units))
                ax.set_ylabel(r'Portion of $\beta \leq$ 1 [%] ')

                ax = fig.add_subplot(224)
                tmp_value = tmp_value[~np.isnan(tmp_value)]
                h, edges, = np.histogram(tmp_value, bins=30)
                width = edges[1]-edges[0]
                ax.plot(edges[:-1]+width/2, h, marker='.', color='black')
                ax.grid(ls='--', color='grey')
                ax.set_xlabel('{} {} [{}]'.format(self.truth, self.variable, self.units))
                ax.set_ylabel('Number')
                ax.set_yscale('log')

                plt.tight_layout()
                plt.savefig(figname)
                print(figname, '  saved')
                plt.close()


def _print_beta_corr(beta, unc, savepath):
    phases_list = ['ALL', 'LIQ', 'ICE']
    cot_lims = unc['ALL'].keys()
    s = ''
    for c in cot_lims:
        s += 'COT LIMIT: {}\n'.format(c)
        for p in phases_list:
            s += '   PHASE: {}\n'.format(p)

            X = beta.d[p][c]
            Y = unc[p][c]['IMG']
            mask = np.logical_and(~np.isnan(X), ~np.isnan(Y))
            X = X[mask]
            Y = Y[mask]
            r, _ = pearsonr(X, Y)
            s += '      cor(|d|, unc_cci) = {:.3f}\n'.format(r)

            X = beta.d[p][c]
            Y = beta.u[p][c]
            mask = np.logical_and(~np.isnan(X), ~np.isnan(Y))
            X = X[mask]
            Y = Y[mask]
            r, _ = pearsonr(X, Y)
            s += '      cor(|d|, u) = {:.3f}\n'.format(r)

            X = beta.d[p][c]
            Y = beta.beta[p][c]
            mask = np.logical_and(~np.isnan(X), ~np.isnan(Y))
            X = X[mask]
            Y = Y[mask]
            r, _ = pearsonr(X, Y)
            s += '      cor(|d|, beta) = {:.3f}\n'.format(r)

    filepath = os.path.join(savepath, 'correlations.txt')
    with open(filepath, 'w') as fh:
        fh.write(s)

    print(filepath, ' saved')


if __name__ == '__main__':

    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-mc', '--matchup_file_calipso', type=str, default=None)
    parser.add_argument('-mf', '--matchup_file_amsr2', type=str, default=None)
    parser.add_argument('-md', '--matchup_file_dardar', type=str, default=None)
    parser.add_argument('-cl', '--cot_list',  nargs='+')
    parser.add_argument('-sl', '--satz_list', default=70, type=float)
    parser.add_argument('-od', '--output_dir', type=str, required=True)
    args = parser.parse_args()

    # CALIPSO
    if args.matchup_file_calipso is not None:
        calipso = ProcessorCalipso(args.matchup_file_calipso)
        calipso_available = True
    else:
        calipso_available = False

    # AMSR2
    if args.matchup_file_amsr2 is not None:
        calipso = ProcessorAMR2(args.matchup_file_amsr2)
        amsr2_available = True
    else:
        amsr2_available = False

    # DARDAR
    if args.matchup_file_dardar is not None:
        calipso = ProcessorDARDAR(args.matchup_file_dardar)
        dardar_available = True
    else:
        dardar_available = False


    if calipso_available:
        # ---------------------- CTP UNCERTAINTY --------------------------
        ctp_output_dir = os.path.join(args.output_dir, 'CTP')
        if not os.ispath(ctp_output_dir):
            os.makedirs(ctp_output_dir)
        calipso.read_ctp()

        beta_ctp = BetaRatio(calipso.ctp_results, calipso.ctp_unc_results)
        beta_ctp.calculate()

        _print_beta_corr(beta_ctp,
                         calipso.ctp_unc_results,
                         ctp_output_dir)

        p = Plotting(calipso.ctp_results,
                 calipso.ctp_unc_results,
                 beta_ctp.beta,
                 output_dir=ctp_output_dir,
                 variable='CTP',
                 units='hPa',
                 truth=calipso.truth)
        p.plot_all()

        # ---------------------- CTH UNCERTAINTY --------------------------
        cth_output_dir = os.path.join(args.output_dir, 'CTH')
        if not os.ispath(cth_output_dir):
            os.makedirs(cth_output_dir)
        calipso.read_cth()

        beta_cth = BetaRatio(calipso.cth_results, calipso.cth_unc_results)
        beta_cth.calculate()

        _print_beta_corr(beta_cth,
                         calipso.cth_unc_results,
                         cth_output_dir )

        p = Plotting(calipso.cth_results,
                 calipso.cth_unc_results,
                 beta_cth.beta,
                 output_dir=cth_output_dir,
                 variable='CTH',
                 units='km',
                 truth=calipso.truth)
        p.plot_all()

        # ---------------------- CTT UNCERTAINTY --------------------------
        ctt_output_dir = os.path.join(args.output_dir, 'CTT')
        if not os.ispath(ctt_output_dir):
            os.makedirs(ctt_output_dir)
        calipso.read_ctt()

        beta_ctt = BetaRatio(calipso.ctt_results, calipso.ctt_unc_results)
        beta_ctt.calculate()

        _print_beta_corr(beta_ctt, calipso.ctt_unc_results, ctt_output_dir)

        p = Plotting(calipso.ctt_results,
                 calipso.ctt_unc_results,
                 beta_ctt.beta,
                 output_dir=ctt_output_dir,
                 variable='CTT',
                 units='K',
                 truth=calipso.truth)
        p.plot_all()

    if amsr2_available:
        pass

    if dardar_available:
        pass
