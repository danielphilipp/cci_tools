import numpy as np
import numpy.ma as ma
from scipy import stats
import matplotlib.path as mpath
import pyresample.geometry


def global_mean_regular(data, lat, lon):
    """
    Weighted global arithmetic mean of data on regular grid.
    """
    assert len(lat.shape) == len(lon.shape)

    if len(lat.shape) == 2:
        lat = lat[:,0]
        lon = lon[0,:]

    xdim = lon.shape[0]
    ydim = lat.shape[0]
    cos_field = np.empty([ydim, xdim])

    for i in range(0, ydim):
        cos_field[i][:] = abs(np.cos(lat[i] * np.pi / 180))

    return ma.average(data, weights=cos_field)


def cre_toa(toa_lwup, toa_lwup_clr, toa_swup, toa_swup_clr):
    """
    Calculate CRE at TOA.
    """
    lw = toa_lwup_clr - toa_lwup
    sw = toa_swup_clr - toa_swup
    tot = lw + sw
    return lw, sw, tot


def cre_boa(boa_lwdn, boa_lwdn_clr, boa_swdn, boa_swdn_clr):
    """
    Calculate CRE at BOA.
    """
    lw = boa_lwdn - boa_lwdn_clr
    sw = boa_swdn - boa_swdn_clr
    tot = lw + sw
    return lw, sw, tot


def cfclow_sic_detection_efficiency_correction(cfc, sic):
    """
    Correct cfc low for detection efficiency over melting sea ice.
    Derived from CALIOP collocations.
    """
    x = np.asarray([5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    y_JFM = np.asarray([6.51608, 10.6481, 12.8007, 7.54005, 11.5188, 14.7673,
                        11.9644, 10.2299, 1.41286, -2.53112])
    y_AMJ = np.asarray([7.11481, 8.24505, 8.59341, 11.0957, 8.73511, 10.1767,
                        9.46830, 8.12593, 7.85213, 5.81272])
    y_JJS = np.asarray([2.97948, 3.71151, 6.46204, 6.61056, 5.51932, 6.30836,
                        6.50598, 8.54252, 6.78840, 9.29544])
    y_OND = np.asarray([5.52121, 6.18386, 5.94645, 7.55352, 7.36561, 5.38008,
                        7.05116, 7.37463, 0.499668, -3.13082])

    m_JFM1, b_JFM1, r_JFM1, p_JFM1, std_err_JFM1 = Statistics.regression(x[:-3],
                                                              y_JFM[:-3])
    m_JFM2, b_JFM2, r_JFM2, p_JFM2, std_err_JFM2 = Statistics.regression(x[-3:],
                                                              y_JFM[-3:])

    m_AMJ, b_AMJ, r_AMJ, p_AMJ, std_err_AMJ = Statistics.regression(x[:],
                                                                    y_AMJ[:])

    m_JJS, b_JJS, r_JJS, p_JJS, std_err_JJS = Statistics.regression(x[:],
                                                                    y_JJS[:])

    m_OND1, b_OND1, r_OND1, p_OND1, std_err_OND1 = Statistics.regression(x[:-3],
                                                              y_OND[:-3])
    m_OND2, b_OND2, r_OND2, p_OND2, std_err_OND2 = Statistics.regression(x[-3:],
                                                              y_OND[-3:])

    cfc_out = ma.empty(cfc.shape)

    for f in range(0, 12):
        cfc_tmp = cfc[f, :, :]
        sic_tmp = sic[f, :, :]

        if f in [0, 1, 2]:
            m1 = m_JFM1
            b1 = b_JFM1
            m2 = m_JFM2
            b2 = b_JFM2

        if f in [3, 4, 5]:
            m1 = m_AMJ
            b1 = b_AMJ

        if f in [6, 7, 8]:
            m1 = m_JJS
            b1 = b_JJS

        if f in [9, 10, 11]:
            m1 = m_OND1
            b1 = b_OND1
            m2 = m_OND2
            b2 = b_OND2

        if f in [0, 1, 2, 9, 10, 11]:
            cfc_tmp[sic_tmp < 75.] -= (m1*sic_tmp[sic_tmp < 75.] + b1) / 100.
            cfc_tmp[sic_tmp >= 75.] -= (m2*sic_tmp[sic_tmp >= 75.] + b2) / 100.
        else:
            cfc_tmp -= (m1 * sic_tmp + b1) / 100.

        cfc_out[f, :, :] = cfc_tmp

    return cfc_out


def get_latlon_box_indices_regular(extent, resolution):
    minlon = extent[0]
    maxlon = extent[1]
    minlat = extent[2]
    maxlat = extent[3]


class Definitions:
    def __init__(self):
        self.months = ['January', 'February', 'March', 'April',
                       'May', 'June', 'July', 'August', 'September',
                       'October', 'November', 'December']
        

class Plotting:
    @staticmethod
    def bounds():
        """
        Get circular boundaries for ccrs.NorthPolarStereo.
        """
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        return circle

    @staticmethod
    def create_scat(p, lat, lon):
        """
        Create p-value scattering.
        """
        idy, idx = np.nonzero(p > 0.05)
        for k in range(0, idx.shape[0]):
            idx[k] = lon[idx[k]]
            idy[k] = lat[idy[k]]
        return idx, idy


class Statistics:
    @staticmethod
    def correlation(x, y):
        """
        Calculate pearson R and corresponding p-value
        """
        r, p = stats.pearsonr(x, y)
        return r, p
    
    @staticmethod
    def linregress_multidimensional(x, y, timedim=0):
        """ Calculate linear slope, intercept, Pearson correlation coefficient 
            and p-value even on multidimensional arrays. """
        from scipy import special

        # find NaN elements
        mask = np.logical_or(np.isnan(x), np.isnan(y))
        # mask pixel-wise where at least one element in time
        # series is invalid
        pixel_any_nan_mask = np.any(mask, axis=timedim)
        # replace invalid values with a FillValue to run correlation
        # Fill value should not be constant as otherwise the calculation
        # of r divides by zero (std=0). Thus we replace invalid values
        # with random values and mask those pixels afterwards.
        rand = np.random.random(x.shape)
        x[mask] = rand[mask]
        y[mask] = rand[mask]

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        Xm = x - np.mean(x, axis=timedim)
        Ym = y - np.mean(y, axis=timedim)

        N = Xm.shape[timedim]

        # Average sums of square differences from the mean
        # mean( (x-mean(x))^2 )
        ssxm = (Xm * Xm).sum(axis=timedim) / N
        # mean( (y-mean(y))^2 )
        ssym = (Ym * Ym).sum(axis=timedim) / N
        # mean( (x-mean(x)) * (y-mean(y)) )
        ssxym = (Xm * Ym).sum(axis=timedim) / N

        slope = ssxym / ssxm
        intercept = np.mean(y, axis=timedim) - slope*np.mean(x, axis=timedim)

        X_std = np.std(Xm, axis=timedim)
        Y_std = np.std(Ym, axis=timedim)
        r = ssxym / (X_std * Y_std)

        # calculate corresponding p-value
        #the p-value can be computed as p = 2*dist.cdf(-abs(r))
        # where dist is the beta distribution on [-1, 1] with shape parameters
        # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
        # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
        # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
        # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
        # to avoid a TypeError raised by btdtr when r is higher precision.)
        ab = N/2 - 1
        pval = 2*special.btdtr(ab, ab, 0.5*(1 - np.abs(np.float64(r))))

        # mask pixels which's time series has at least 1 invalid element
        slope = np.where(pixel_any_nan_mask, np.nan, slope)
        intercept = np.where(pixel_any_nan_mask, np.nan, intercept)
        r = np.where(pixel_any_nan_mask, np.nan, r)
        pval = np.where(pixel_any_nan_mask, np.nan, pval)

        return slope, intercept, r, pval
    
    @staticmethod
    def correlate_multidimensional(x, y, timedim=0):
        """ Calculate Pearson correlation coefficient and p-value even on
            multidimensional arrays. """

        from scipy import special

        # find NaN elements
        mask = np.logical_or(np.isnan(x), np.isnan(y))
        # mask pixel-wise where at least one element in time
        # series is invalid
        pixel_any_nan_mask = np.any(mask, axis=timedim)
        # replace invalid values with a FillValue to run correlation
        # Fill value should not be constant as otherwise the calculation
        # of r divides by zero (std=0). Thus we replace invalid values
        # with random values and mask those pixels afterwards.
        rand = np.random.random(x.shape)
        x[mask] = rand[mask]
        y[mask] = rand[mask]

        x = x.astype(np.float64)
        y = y.astype(np.float64)

        # calculate covariance
        Xm = x - np.mean(x, axis=timedim)
        Ym = y - np.mean(y, axis=timedim)
        N = Xm.shape[timedim]
        cov = (Xm * Ym).sum(axis=timedim) / N

        # calculate Pearson correlation coefficient
        X_std = np.std(Xm, axis=timedim)
        Y_std = np.std(Ym, axis=timedim)
        r = cov / (X_std *  Y_std)

        # calculate corresponding p-value
        #the p-value can be computed as p = 2*dist.cdf(-abs(r))
        # where dist is the beta distribution on [-1, 1] with shape parameters
        # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
        # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
        # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
        # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
        # to avoid a TypeError raised by btdtr when r is higher precision.)
        ab = N/2 - 1
        pval = 2*special.btdtr(ab, ab, 0.5*(1 - np.abs(np.float64(r))))

        # mask pixels which's time series has at least 1 invalid element
        r = np.where(pixel_any_nan_mask, np.nan, r)
        pval = np.where(pixel_any_nan_mask, np.nan, pval)

        return r, pval

    @staticmethod
    def regression(x, y):
        """
        Linear least squares regression.
        """
        m, b, r, p, std_err = stats.linregress(x, y)
        return m, b, r, p, std_err

    @staticmethod
    def standardize(x):
        """
        Standardize timeseries for zero mean and unit variance.
        """
        return (x - ma.mean(x)) / ma.std(x)

    @staticmethod
    def deseasonalize(x):
        """
        Deseasonalize monthly data using the seasonal mean.
        """
        xvals = ma.empty(x.shape)

        if len(x.shape) == 3:
            is_2d = True
        elif len(x.shape) == 1:
            is_2d = False
        else:
            raise Exception('Array must be 1D (time) or 2D (time, lat, lon)')

        for m in range(0, 12):
            if is_2d:
                xmean = ma.mean(x[m::12, :, :], axis=0)
                xvals[m::12, :, :] = x[m::12, :, :] - xmean
            else:
                xmean = ma.mean(x[m::12], axis=0)
                xvals[m::12] = x[m::12] - xmean
        return xvals


class GridDefinitions:

    @staticmethod
    def regular_grid2(resolution):
        from pyresample import load_area
        if resolution == 0.05:
            return load_area('areas.yaml', 'pc_world_005')
        elif resolution == 0.5:
            return load_area('/cmsaf/nfshome/dphilipp/software/cci_tools/areas.yaml', 'pc_world_05')
        elif resolution == 1.0:
            return load_area('areas.yaml', 'pc_world_1')
        else:
            raise Exception('Resolution {} not available in file ' \
                            'areas.yaml'.format(resolution))

    @staticmethod
    def regular_grid_latlon(resolution, extent):
        from pyresample.geometry import create_area_def
        import numpy as np
        dimx = int(np.rint(360 / resolution))
        dimy = int(np.rint(180 / resolution))
        area = create_area_def(
                        area_id='global',
                        proj_id='regular grid 0.05 deg',
                        description='Global regular grid 0.05 degrees res',
                        projection={'proj': 'latlong'},
                        width=dimx,
                        height=dimy,
                        area_extent=extent,
                        units='degrees'
        )
        return area

    @staticmethod
    def ease2_250_nh(width, height):
        """
        Grid for 25 km NH EASE2 grid. width/height for OSI-SAF is 432.
        Projection is Lambert Azimuthal Equal Area (laea).
        """

        llx, lly, urc, ury = (
        -5387.5 * 1000, -5387.5 * 1000, 5387.5 * 1000, 5387.5 * 1000)

        ogrid = pyresample.geometry.AreaDefinition(
                    area_id='ease2_250_nh',
                    name='EASE 2 Grid, 25km, Northern Hemisphere',
                    proj_id='dummy',
                    proj_dict=dict(proj='laea', lon_0=0, datum='WGS84',
                                   ellps='WGS84', lat_0=90.0),
                    x_size=width,
                    y_size=height,
                    area_extent=(llx, lly, urx, ury)
        )
        return ogrid

    @staticmethod
    def seviri_grid(width,height):
        """
        Grid for SEVIRI geostationary CGMS projection. Before 6 Dec 2017 L1.5
        SEVIRI data were shifted by 1.5km SSP to North and West.
        This is the old grid before this date.
        """
        llx, lly, urx, ury = (-5570248.686685662,
                              -5567248.28340708,
                               5567248.28340708,
                               5570248.686685662)

        ogrid = pyresample.geometry.AreaDefinition(
                    area_id='cgms',
                    description='CGMS SEVIRI Grid',
                    proj_id='geos',
                    projection={'a': 6378169.0, 'b': 6356583.8, 'lon_0': 0.0,
                        'h': 35785831.0, 'proj': 'geos', 'units': 'm'},
                    width=width,
                    height=height,
                    area_extent=(llx, lly, urx, ury)
        )

        return ogrid

    @staticmethod
    def seviri_grid_shifted(width,height):
        """
        Grid for SEVIRI geostationary shifted projection. Before 6 Dec 2017 L1.5
        SEVIRI data were shifted by 1.5km SSP to North and West.
        This is the new grid after this date
        """
        shift = 1500
        llx, lly, urx, ury = (-5570248.686685662 + shift,
                              -5567248.28340708 - shift,
                               5567248.28340708 + shift,
                               5570248.686685662 - shift)

        ogrid = pyresample.geometry.AreaDefinition(
                    area_id='seviri_shifted',
                    description='SEVIRI Grid shifted',
                    proj_id='geos',
                    projection={'a': 6378169.0, 'b': 6356583.8, 'lon_0': 0.0,
                                'h': 35785831.0, 'proj': 'geos', 'units': 'm'},
                    width=width,
                    height=height,
                    area_extent=(llx, lly, urx, ury)
        )
        return ogrid
