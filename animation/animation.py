import glob
import argparse
import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from gridDefs import seviri_grid
from matplotlib import colors
import numpy as np

fore_color = 'white'
back_color = 'black'
plt.rcParams["text.color"] = fore_color
plt.rcParams["axes.labelcolor"] = fore_color
plt.rcParams["xtick.color"] =  fore_color
plt.rcParams["ytick.color"] = fore_color


def _set_binary_cmap(c1, c2):
    cmap = colors.ListedColormap([c1, c2])
    bounds = [0, 0.5, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _set_plot_config(ax, date, cbar, cbar_label, coastline_color):
    ax.coastlines(color=coastline_color)
    ax.set_facecolor(back_color)
    ax.background_patch.set_facecolor(back_color)
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)


def _plot_animation(binary, regression, mode, odir, sev, date):

    if mode == 'CMA':
        ticks = ['CLR', 'CLD']
        c1 = plt.cm.viridis(0)
        c2 = plt.cm.viridis(255)
        suptitle = 'Cloud mask NN'
    if mode == 'CPH':
        binary = np.where(binary == 0, np.nan, binary)
        binary -= 1
        c1 = plt.cm.viridis(0)
        c2 = plt.cm.viridis(255)
        ticks = ['LIQ', 'ICE']
        suptitle = 'Cloud phase NN'

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(back_color)

    ax = fig.add_subplot(121, projection=sev)

    img = ax.imshow(regression, extent=sev.bounds, transform=sev, vmin=0, vmax=1)
    cbar = plt.colorbar(img, orientation='horizontal')
    _set_plot_config(ax, date, cbar, 'NN {} output'.format(mode), 'orange')

    ax = fig.add_subplot(122, projection=sev)
    cmap, norm = _set_binary_cmap(c1, c2)
    img = ax.imshow(binary, extent=sev.bounds, transform=sev, cmap=cmap, norm=norm, vmin=0, vmax=1)
    cbar = plt.colorbar(img, orientation='horizontal')
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(ticks)
    _set_plot_config(ax, date, cbar, 'Binary {}'.format(mode), 'orange')

    plt.suptitle(suptitle + ' ' + date[8:] + 'UTC',
                 fontweight='bold',
                 color=fore_color,
                 fontsize=18)
    plt.tight_layout()
    figname = mode + '_' + date + '.png'
    plt.savefig(os.path.join(args.odir, figname),
                facecolor=back_color)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--primary_file_pattern', type=str, required=True)
    parser.add_argument('--odir', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True,
                        choices=['CMA', 'CPH'])
    args = parser.parse_args()

    primary_files = glob.glob(args.primary_file_pattern, recursive=True)
    nfiles = len(primary_files)

    seviri_crs = seviri_grid(3712, 3712).to_cartopy_crs()

    for cnt, file in enumerate(primary_files):
        print('Reading ({:02d}/{}): {}'.format(cnt+1, nfiles, os.path.basename(file)))
        date = os.path.basename(file)[:12]
        ds = xr.open_dataset(file, decode_times=False).squeeze()

        if args.mode == 'CMA':
            bin = np.fliplr(ds['cldmask'])
            reg = np.fliplr(ds['cccot_pre'])
        if args.mode == 'CPH':
            bin = np.fliplr(ds['ann_phase'])
            reg = np.fliplr(ds['cphcot'])

        _plot_animation(bin, reg, args.mode, args.odir, seviri_crs, date)


