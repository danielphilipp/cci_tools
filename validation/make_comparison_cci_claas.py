from os.path import join as ojoin
import xarray as xr
from cci_tools.helperfuncs import GridDefinitions as gd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import dask.array as da
import numpy as np
from scipy.stats import gaussian_kde, linregress
from matplotlib.colors import LogNorm
import argparse

CCI_EXTENT = [-179.95, 179.95, -89.95, 89.95]
CCI_RESOLUTION = 0.5

def weighted_spatial_average(data, extent=[-89.975, 89.975, -89.975, 89.975], res=0.05):

    if isinstance(data, xr.DataArray):
        data = data.data

    if isinstance(data, np.ndarray):
        data = da.from_array(data, chunks=(100, 100))

    lon = np.arange(extent[0], extent[1], res)
    lat = np.arange(extent[2], extent[3], res)
    latcos = np.abs(np.cos(lat * np.pi / 180))

    cosfield = np.empty([lat.shape[0], lon.shape[0]])
    cosfield[...] = latcos[:, None]
    cosfield = da.from_array(cosfield, chunks=(100, 100))#[mask]

    return da.nansum(data * cosfield) / da.nansum(cosfield)


def get_claas3_mm(basepath, year, month):
    cfc_file = ojoin(basepath, 'cfc', year, month, 'CFCmm{}{}01000000423SVMSG01MA.nc'.format(year, month))
    cph_file = ojoin(basepath, 'cph', year, month, 'CPHmm{}{}01000000423SVMSG01MA.nc'.format(year, month))
    ctx_file = ojoin(basepath, 'cto', year, month, 'CTOmm{}{}01000000423SVMSG01MA.nc'.format(year, month))
    iwp_file = ojoin(basepath, 'cwp', year, month, 'IWPmm{}{}01000000423SVMSG01MA.nc'.format(year, month))
    lwp_file = ojoin(basepath, 'cwp', year, month, 'LWPmm{}{}01000000423SVMSG01MA.nc'.format(year, month))

    ds_iwp = xr.open_dataset(iwp_file, chunks={'lat': 1000, 'lon': 1000})[['iwp', 'cot_ice', 'cre_ice', 'SZA']]
    ds_lwp = xr.open_dataset(lwp_file, chunks={'lat': 1000, 'lon': 1000})[['lwp', 'cot_liq', 'cre_liq']]

    ds_cfc = xr.open_dataset(cfc_file, chunks={'lat': 1000, 'lon': 1000})[['cfc', 'cfc_low', 'cfc_middle', 'cfc_high']]
    ds_cph = xr.open_dataset(cph_file, chunks={'lat': 1000, 'lon': 1000})[['cph']]
    ds_ctx = xr.open_dataset(ctx_file, chunks={'lat': 1000, 'lon': 1000})[['ctp', 'ctt', 'cth']]

    #cond_lwp = np.logical_and(np.isnan(ds_lwp['lwp']), ~np.isnan(ds_iwp['iwp']))
    #cond_iwp = np.logical_and(np.isnan(ds_iwp['iwp']), ~np.isnan(ds_lwp['lwp']))

    #ds_lwp['lwp'] = xr.where(cond_lwp, 0, ds_lwp['lwp'])
    #ds_iwp['iwp'] = xr.where(cond_iwp, 0, ds_iwp['iwp'])

    cwp = ds_lwp['lwp'] * ds_cph['cph']*0.01 + (1-ds_cph['cph']*0.01) * ds_iwp['iwp']
    cot = ds_lwp['cot_liq'] * ds_cph['cph']*0.01 + (1-ds_cph['cph']*0.01) * ds_iwp['cot_ice']
    cer = ds_lwp['cre_liq'] * ds_cph['cph']*0.01 + (1-ds_cph['cph']*0.01) * ds_iwp['cre_ice']

    ds = ds_cfc.merge(ds_cph).merge(ds_ctx)
    ds['cth'] = ds['cth'] / 1000
    ds['cwp'] = cwp * 1000
    ds['iwp'] = ds_iwp['iwp'] * 1000
    ds['lwp'] = ds_lwp['lwp'] * 1000
    ds['cot'] = cot
    ds['cer'] = cer * 1E6

    ds = ds.rename({'cfc_middle': 'cfc_mid', 'cfc_high': 'cfc_high'})
    ds = ds.squeeze()

    from pyresample.bucket import BucketResampler
    from cci_tools.helperfuncs import GridDefinitions
    import dask.array as da

    ilat = ds['lat'].data
    ilon = ds['lon'].data
    ilon, ilat = np.meshgrid(ilon, ilat)
    ogrid = GridDefinitions.regular_grid2(0.5)
    olon, olat = ogrid.get_lonlats()
    resampler = BucketResampler(ogrid, da.from_array(ilon, chunks=(1000, 1000)), da.from_array(ilat, chunks=(1000, 1000)))
    ds_lowres = xr.Dataset(coords={'lon': ('lon', olon[0,:]), 'lat': ('lat', olat[:,0])})
    for v in ds.variables:
        if v not in ['time', 'lat', 'lon']:
            idata = ds[v].data
            ds_lowres[v] = (('lat', 'lon'), resampler.get_average(idata))

    return ds_lowres


def get_cci_mm_official(ifile):

    ds = xr.open_dataset(ifile).squeeze()
    ds = ds.squeeze()

    ds['cfc'] = ds['cfc'] * 100
    ds['cfc_low'] = ds['cfc_low'] * 100#* ds['cfc']
    ds['cfc_mid'] = ds['cfc_mid'] * 100#* ds['cfc']
    ds['cfc_high'] = ds['cfc_high'] * 100#* ds['cfc']
    ds['cph'] = ds['cph'] * 100

    cond_lwp = np.logical_and(np.isnan(ds['lwp']), ~np.isnan(ds['iwp']))
    cond_iwp = np.logical_and(np.isnan(ds['iwp']), ~np.isnan(ds['lwp']))

    ds['lwp'] = xr.where(cond_lwp, 0, ds['lwp'])
    ds['iwp'] = xr.where(cond_iwp, 0, ds['iwp'])
    ds['cwp'] = ds['lwp'] * ds['cph'] * 0.01 + (1 - ds['cph'] * 0.01) * ds['iwp']

    return ds


def get_cci_mm_manual(ifile):
    
    ds = xr.open_dataset(ifile).squeeze()
    ds = ds.squeeze()

    ds = ds.rename({'cfc_hig': 'cfc_high', 'ann_phase': 'cph'})

    ds['cfc'] = ds['cfc'] * 100
    ds['cfc_low'] = ds['cfc_low'] * ds['cfc']
    ds['cfc_mid'] = ds['cfc_mid'] * ds['cfc']
    ds['cfc_high'] = ds['cfc_high'] * ds['cfc']
    ds['cph'] = ds['cph'] * 100


    ds['iwp'] = xr.where(np.isnan(ds['cot_day']), np.nan, ds['iwp'])
    ds['lwp'] = xr.where(np.isnan(ds['cot_day']), np.nan, ds['lwp'])

    #ds['iwp'] = (1-ds['cph']/100) * ds['cwp_day']
    #ds['lwp'] = ds['cph'] / 100 * ds['cwp_day']

    #cond_lwp = np.logical_and(np.isnan(ds['lwp']), ~np.isnan(ds['iwp']))
    #cond_iwp = np.logical_and(np.isnan(ds['iwp']), ~np.isnan(ds['lwp']))

    #ds['lwp'] = xr.where(cond_lwp, 0, ds['lwp'])
    #ds['iwp'] = xr.where(cond_iwp, 0, ds['iwp'])

    #cond_lwp = np.logical_and(np.isnan(ds['lwp']), ~np.isnan(ds['iwp']))
    #cond_iwp = np.logical_and(np.isnan(ds['iwp']), ~np.isnan(ds['lwp']))

    #ds['lwp'] = xr.where(cond_lwp, 0, ds['lwp'])
    #ds['iwp'] = xr.where(cond_iwp, 0, ds['iwp'])
    #ds['cwp'] = ds['lwp'] * ds['cph'] * 0.01 + (1 - ds['cph'] * 0.01) * ds['iwp']

    return ds


def plot_cfc_comparison(cci_data, claas_data, figname, slstr):
    regular = ccrs.PlateCarree()

    extent = [-180, 180, -90, 90]
    plot_data = {'CCI CFC': [np.flipud(cci_data['cfc']), 0, 100, 'CFC [%]', 'viridis'],
                 'CCI CFC_low': [np.flipud(cci_data['cfc_low']), 0, 60, 'CFC_low [%]', 'viridis'],
                 'CCI CFC_mid': [np.flipud(cci_data['cfc_mid']), 0, 40, 'CFC_mid [%]', 'viridis'],
                 'CCI CFC_high': [np.flipud(cci_data['cfc_high']), 0, 40, 'CFC_high [%]', 'viridis'],
                 '{} CFC'.format(slstr): [claas_data['cfc'], 0, 100, 'CFC [%]', 'viridis'],
                 '{} CFC_low'.format(slstr): [claas_data['cfc_low'], 0, 60, 'CFC_low [%]', 'viridis'],
                 '{} CFC_mid'.format(slstr): [claas_data['cfc_mid'], 0, 40, 'CFC_mid [%]', 'viridis'],
                 '{} CFC_high'.format(slstr): [claas_data['cfc_high'], 0, 40, 'CFC_high [%]', 'viridis'],
                 '(CCI - {}) CFC'.format(slstr): [np.flipud(cci_data['cfc'].data) - claas_data['cfc'], -30, 30, 'Diff [%]', 'bwr'],
                 '(CCI - {}) CFC_low'.format(slstr): [np.flipud(cci_data['cfc_low'].data) - claas_data['cfc_low'], -30, 30, 'Diff [%]', 'bwr'],
                 '(CCI - {}) CFC_mid'.format(slstr): [np.flipud(cci_data['cfc_mid'].data) - claas_data['cfc_mid'], -30, 30, 'Diff [%]', 'bwr'],
                 '(CCI - {}) CFC_high'.format(slstr): [np.flipud(cci_data['cfc_high'].data) - claas_data['cfc_high'], -30, 30, 'Diff [%]', 'bwr']}

    ctp_mask = np.isnan(np.flipud(cci_data['ctp']))
    s = ''
    fig = plt.figure(figsize=(14, 7))
    for cnt, key in enumerate(plot_data.keys()):
        ax = fig.add_subplot(3, 4, cnt+1, projection=regular)
        ax.coastlines()
        ax.set_extent(extent, crs=regular)
        ax.set_title(key)
        data = plot_data[key][0]

        ims = ax.imshow(data, extent=extent, transform=regular, origin='upper',
                        vmin=plot_data[key][1], vmax=plot_data[key][2], cmap=plt.get_cmap(plot_data[key][4]))
        #av = weighted_spatial_average(data, CCI_EXTENT, CCI_RESOLUTION).compute()
        #ax.annotate(xy=(0.0, 0.0), s='MEAN={:.2f}'.format(av), xycoords='axes fraction', color='red', fontweight='bold',
        #            backgroundcolor='lightgrey')

        s += '{}: {:.3f}\n'.format(key,  np.nanmean(np.where(ctp_mask, np.nan, data)))
        ax.set_extent([-90,90,-90, 90])
        cbar = plt.colorbar(ims)
        cbar.set_label(plot_data[key][3])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig(figname)
    print('SAVED ', figname)
    return s


def plot_ctx_comparison(cci_data, claas_data, figname, slstr):
    regular = ccrs.PlateCarree()
    extent = [-180, 180, -90, 90]

    plot_data = {'CCI CTP': [np.flipud(cci_data['ctp']), 200, 1000, 'CTP [hPa]', 'viridis'],
                 'CCI CTT': [np.flipud(cci_data['ctt']), 180, 300, 'CTT [K]', 'viridis'],
                 'CCI CTH': [np.flipud(cci_data['cth']), 0, 12, 'CTH [km]', 'viridis'],
                 '{} CTP'.format(slstr): [claas_data['ctp'], 200, 1000, 'CTP [hPa]', 'viridis'],
                 '{} CTT'.format(slstr): [claas_data['ctt'], 180, 300, 'CTT [K]', 'viridis'],
                 '{} CTH'.format(slstr): [claas_data['cth'], 0, 12, 'CTH [km]', 'viridis'],
                 '(CCI - {}) CTP'.format(slstr): [np.flipud(cci_data['ctp'].data) - claas_data['ctp'], -300, 300, 'Diff [hPa]', 'bwr'],
                 '(CCI - {}) CTT'.format(slstr): [np.flipud(cci_data['ctt'].data) - claas_data['ctt'], -50, 50, 'Diff [K]', 'bwr'],
                 '(CCI - {}) CTH'.format(slstr): [np.flipud(cci_data['cth'].data) - claas_data['cth'], -5, 5, 'Diff [km]', 'bwr']}

    ctp_mask = np.isnan(np.flipud(cci_data['ctp']))
    s = ''
    fig = plt.figure(figsize=(14, 9))
    for cnt, key in enumerate(plot_data.keys()):
        ax = fig.add_subplot(3, 3, cnt + 1, projection=regular)
        ax.coastlines()
        ax.set_extent(extent, crs=regular)
        ax.set_title(key)
        data = plot_data[key][0]

        ims = ax.imshow(data, extent=extent, transform=regular, origin='upper',
                        vmin=plot_data[key][1], vmax=plot_data[key][2], cmap=plt.get_cmap(plot_data[key][4]))
        #av = weighted_spatial_average(data, CCI_EXTENT, CCI_RESOLUTION).compute()
        #ax.annotate(xy=(0.0, 0.0), s='MEAN={:.2f}'.format(av), xycoords='axes fraction', color='red', fontweight='bold',
        #            backgroundcolor='lightgrey')

        s += '{}: {:.3f}\n'.format(key,  np.nanmean(np.where(ctp_mask, np.nan, data)))

        ax.set_extent([-90, 90, -90, 90])
        cbar = plt.colorbar(ims)
        cbar.set_label(plot_data[key][3])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig(figname)
    print('SAVED ', figname)
    return s


def plot_cld_comparison(cci_data, claas_data, figname, slstr):
    regular = ccrs.PlateCarree()

    plot_data = {'CCI COT': [np.flipud(cci_data['cot']), 0, 30, 'COT [1]', 'viridis'],
                 'CCI CER': [np.flipud(cci_data['cer']), 10, 40, r'CER [$\mu$m]', 'viridis'],
                 'CCI IWP': [np.flipud(cci_data['iwp']), 0, 220, r'WP [g m$^{-2}$]', 'viridis'],
                 'CCI LWP': [np.flipud(cci_data['lwp']), 0, 220, r'IWP [g m$^{-2}$]', 'viridis'],
                 'CCI LCF': [np.flipud(cci_data['cph']), 0, 100, 'LCF [%]', 'viridis'],
                 '{} COT'.format(slstr): [claas_data['cot'], 0, 30, 'COT [1]', 'viridis'],
                 '{} CER'.format(slstr): [claas_data['cer'], 10, 40, r'CER [$\mu$m]', 'viridis'],
                 '{} IWP'.format(slstr): [claas_data['iwp'], 0, 220, r'IWP [gm$^{-2}$]', 'viridis'],
                 '{} LWP'.format(slstr): [claas_data['lwp'], 0, 220, r'LWP [gm$^{-2}$]', 'viridis'],
                 '{} LCF'.format(slstr): [claas_data['cph'], 0, 100, 'LCF [%]', 'viridis'],
                 '(CCI - {}) COT'.format(slstr): [np.flipud(cci_data['cot'].data) - claas_data['cot'], -10, 10, 'Diff [1]', 'bwr'],
                 '(CCI - {}) CER'.format(slstr): [np.flipud(cci_data['cer'].data) - claas_data['cer'], -30, 30, r'Diff [$\mu$m]', 'bwr'],
                 '(CCI - {}) IWP'.format(slstr): [np.flipud(cci_data['iwp'].data) - claas_data['iwp'], -100, 100, r'Diff [gm$^{-2}$]', 'bwr'],
                 '(CCI - {}) LWP'.format(slstr): [np.flipud(cci_data['lwp'].data) - claas_data['lwp'], -100, 100, r'Diff [gm$^{-2}$]', 'bwr'],
                 '(CCI - {}) LCF'.format(slstr): [np.flipud(cci_data['cph'].data) - claas_data['cph'], -30, 30, 'Diff [%]', 'bwr']}

    extent = [-180, 180, -90, 90]

    ctp_mask = np.isnan(np.flipud(cci_data['ctp']))
    s = ''
    #CLD
    fig = plt.figure(figsize=(16, 7))
    for cnt, key in enumerate(plot_data.keys()):
        ax = fig.add_subplot(3, 5, cnt+1, projection=regular)
        ax.coastlines()
        ax.set_extent(extent, crs=regular)
        ax.set_title(key)
        data = plot_data[key][0]

        ims = ax.imshow(data, extent=extent, transform=regular, origin='upper',
                        vmin=plot_data[key][1], vmax=plot_data[key][2], cmap=plt.get_cmap(plot_data[key][4]))
        #av = weighted_spatial_average(data, CCI_EXTENT, CCI_RESOLUTION).compute()
        #ax.annotate(xy=(0.0, 0.0), s='MEAN={:.2f}'.format(av), xycoords='axes fraction', color='red', fontweight='bold',
        #            backgroundcolor='lightgrey')

        s += '{}: {:.3f}\n'.format(key,  np.nanmean(np.where(ctp_mask, np.nan, data)))
        ax.set_extent([-90,90,-90, 90])
        cbar = plt.colorbar(ims)
        cbar.set_label(plot_data[key][3])

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig(figname)
    print('SAVED ', figname)
    return s


def plot_scatter(cci_data, claas_data, figname, slstr):

    fig = plt.figure(figsize=(12,8))
    vars = ['ctp', 'cth', 'ctt', 'cot', 'cer', 'iwp', 'lwp']
    ranges = {'ctp': (200, 1000), 'ctt': (225, 320), 'cth': (0, 13),
              'cot': (0, 100), 'cer': (0, 60), 'iwp': (0, 800), 'lwp': (0, 500)}

    dummy = np.arange(0,2000)
    for cnt, v in enumerate(vars):
        ax = fig.add_subplot(3, 3, cnt+1)

        x = cci_data[v].data.ravel()
        y = np.flipud(claas_data[v].data).ravel()

        mask = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[~mask]
        y = y[~mask]

        h = ax.hist2d(x, y, bins=(100, 100), cmap=plt.get_cmap('YlOrRd'), norm=LogNorm())
        ax.set_xlabel('CCI {}'.format(v))
        ax.set_ylabel('{} {}'.format(slstr, v))
        ax.set_xlim(ranges[v][0], ranges[v][1])
        ax.set_ylim(ranges[v][0], ranges[v][1])
        plt.colorbar(h[3], ax=ax)
        ax.plot(dummy, dummy, color='black')
        reg = linregress(x, y)
        ax.annotate(xy=(0.05, 0.9), s='r={:.2f}\nr**2={:.2f}'.format(reg[2], reg[2] * reg[2]),
                    xycoords='axes fraction', color='blue', fontweight='bold',
                    backgroundcolor='lightgrey')
        ax.plot(reg[0] * dummy + reg[1], color='blue')
        ax.set_title(v.upper(), fontweight='bold')

    plt.tight_layout()
    plt.savefig(figname)
    print('SAVED ', figname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seviri_l3c', type=str, required=True)
    parser.add_argument('--claas3_l3', type=str, required=True)
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--month', type=str, required=True)
    parser.add_argument('--opath', type=str, required=True)
    args = parser.parse_args()
    S3 = 'CLAAS3'

    cci_data = get_cci_mm_official(args.seviri_l3c)
    claas_data = get_claas3_mm(args.claas3_l3, args.year, args.month)

    ofile_cfc = ojoin(args.opath, '{}{}_{}_SEVIRI_CFC_MM_comparison.png'.format(args.year, args.month, S3))
    ofile_ctx = ojoin(args.opath, '{}{}_{}_SEVIRI_CTX_MM_comparison.png'.format(args.year, args.month, S3))
    ofile_cld = ojoin(args.opath, '{}{}_{}_SEVIRI_CLD_MM_comparison.png'.format(args.year, args.month, S3))
    ofile_sct = ojoin(args.opath, '{}{}_{}_SEVIRI_CTX_MM_scatter.png'.format(args.year, args.month, S3))
    ofile_sta = ojoin(args.opath, '{}{}_{}_stats.txt'.format(args.year, args.month, S3))    


    s_cfc = plot_cfc_comparison(cci_data, claas_data, ofile_cfc, S3)
    s_ctx = plot_ctx_comparison(cci_data, claas_data, ofile_ctx, S3)
    s_cld = plot_cld_comparison(cci_data, claas_data, ofile_cld, S3)
    plot_scatter(cci_data, claas_data, ofile_sct, S3)

    s = s_cfc + '\n\n' + s_ctx + '\n\n' + s_cld

    with open(ofile_sta, 'w') as fh:
        fh.write(s)
