cimport cython
import numpy as np
cimport numpy as np


ctypedef np.uint16_t DTYPEU16_t
ctypedef np.int8_t DTYPE8_t
ctypedef np.int DTYPE32_t
ctypedef np.float32_t FTYPE32_t
ctypedef np.float64_t FTYPE64_t

def _get_cflag(flag, idx1, idx2):
    """ Get cloud flag value from binary flag. """

    flag_int = int(flag)
    flag_len = flag_int.bit_length()
    flag_bin = _intToBin(flag)
    return int(flag_bin[flag_len - idx1:flag_len - idx2], 2)


def _intToBin(x):
    """ Convert integer to binary. """

    return str(bin(x)[2:])


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def parse_profile_cph(np.ndarray[DTYPEU16_t, ndim=2] cal_flags,
                      np.ndarray[FTYPE32_t, ndim=1] cal_laser_energy,
                      np.ndarray[FTYPE32_t, ndim=1] cot_lims,
                      np.ndarray[FTYPE32_t, ndim=2] cal_cot,
                      np.ndarray[FTYPE32_t, ndim=1] img_satz,
                      np.ndarray[FTYPE32_t, ndim=1] cal_cfc,
                      np.ndarray[FTYPE32_t, ndim=1] img_cma,
                      np.ndarray[FTYPE32_t, ndim=1] img_cph,
                      double satz_lim):

    # setup variables
    cdef int n_profiles = cal_cot.shape[0]
    cdef int n_levels = cal_cot.shape[1]
    cdef int n_limits = cot_lims.shape[0]
    cdef int idx, lim, lev
    cdef int cflag, phase, phase_qual
    cdef double cot_sum
    cdef unsigned short cal_fflag
    cdef int FILLVALUE = -999

    cdef np.ndarray col_cph_cal = np.ones([n_limits, n_profiles], dtype=cal_cot.dtype) * FILLVALUE
    cdef np.ndarray col_cph_img = np.ones([n_limits, n_profiles], dtype=cal_cot.dtype) * FILLVALUE

    for idx in range(n_profiles):
        if cal_laser_energy[idx] > 0.08:
            if img_satz[idx] < satz_lim:
                if ~np.isnan(img_cph[idx]) and img_cma[idx] > 0.5 and cal_cfc[idx] > 0.5:
                    for lim in range(n_limits):
                        cot_sum = 0.0
                        for lev in range(n_levels):
                            cal_fflag = cal_flags[idx, lev]
                            if cal_fflag != 1:
                                cflag = _get_cflag(cal_fflag, 3, 0)
                                phase = _get_cflag(cal_fflag, 7, 5)
                                phase_qual = _get_cflag(cal_fflag, 9, 7)
                            else:
                                cflag = FILLVALUE
                                phase = FILLVALUE
                                phase_qual = FILLVALUE

                            if cflag == 2:
                                cot_sum += cal_cot[idx, lev]
                                if cot_sum >= cot_lims[lim] and phase_qual >= 2:
                                    col_cph_cal[lim, idx] = phase
                                    col_cph_img[lim, idx] = img_cph[idx]
                                    #if phase == 1 or phase == 3:
                                    #    col_cph_cal[lim, idx] = 1
                                    #    col_cph_img[lim, idx] = 1
                                    #if phase == 2:
                                    #    col_cph_cal[lim, idx] = 0
                                    #    col_cph_img[lim, idx] = 0
                                    break

    return col_cph_cal, col_cph_img


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def parse_profile_cth(np.ndarray[DTYPEU16_t, ndim=2] cal_flags,
                      np.ndarray[FTYPE32_t, ndim=1] cal_laser_energy,
                      np.ndarray[FTYPE32_t, ndim=1] cot_lims,
                      np.ndarray[FTYPE32_t, ndim=2] cal_cot,
                      np.ndarray[FTYPE32_t, ndim=2] cal_cth,
                      np.ndarray[FTYPE32_t, ndim=1] img_satz,
                      np.ndarray[FTYPE32_t, ndim=1] cal_cfc,
                      np.ndarray[FTYPE32_t, ndim=1] img_cma,
                      np.ndarray[FTYPE32_t, ndim=1] img_cth,
                      np.ndarray[FTYPE32_t, ndim=1] img_cph,
                      double satz_lim):

    # setup variables
    cdef int n_profiles = cal_cot.shape[0]
    cdef int n_levels = cal_cot.shape[1]
    cdef int n_limits = cot_lims.shape[0]
    cdef int idx, lim, lev
    cdef int cflag, phase, phase_qual
    cdef double cot_sum
    cdef unsigned short cal_fflag
    cdef int FILLVALUE = -999

    cdef np.ndarray col_cth_cal_all = np.ones([n_limits, n_profiles], dtype=cal_cth.dtype) * FILLVALUE
    cdef np.ndarray col_cth_cal_ice = np.ones([n_limits, n_profiles], dtype=cal_cth.dtype) * FILLVALUE
    cdef np.ndarray col_cth_cal_wat = np.ones([n_limits, n_profiles], dtype=cal_cth.dtype) * FILLVALUE
    cdef np.ndarray col_cth_img_all = np.ones([n_limits, n_profiles], dtype=cal_cth.dtype) * FILLVALUE
    cdef np.ndarray col_cth_img_ice = np.ones([n_limits, n_profiles], dtype=cal_cth.dtype) * FILLVALUE
    cdef np.ndarray col_cth_img_wat = np.ones([n_limits, n_profiles], dtype=cal_cth.dtype) * FILLVALUE

    for idx in range(n_profiles):
        if cal_laser_energy[idx] > 0.08:
            if img_satz[idx] < satz_lim:
                if ~np.isnan(img_cth[idx]) and img_cma[idx] > 0.5 and cal_cfc[idx] > 0.5:
                    for lim in range(n_limits):
                        cot_sum = 0.0
                        for lev in range(n_levels):
                            cal_fflag = cal_flags[idx, lev]
                            if cal_fflag != 1:
                                cflag = _get_cflag(cal_fflag, 3, 0)
                                phase = _get_cflag(cal_fflag, 7, 5)
                                phase_qual = _get_cflag(cal_fflag, 9, 7)
                            else:
                                cflag = FILLVALUE
                                phase = FILLVALUE
                                phase_qual = FILLVALUE

                            if cflag == 2:
                                cot_sum += cal_cot[idx, lev]

                                if cot_sum >= cot_lims[lim] and phase_qual >= 2:
                                    col_cth_cal_all[lim, idx] = cal_cth[idx, lev]
                                    col_cth_img_all[lim, idx] = img_cth[idx]
                                    if (phase == 1 or phase == 3) and img_cph[idx] == 1:
                                        col_cth_cal_ice[lim, idx] = cal_cth[idx, lev]
                                        col_cth_img_ice[lim, idx] = img_cth[idx]
                                    if phase == 2 and img_cph[idx] == 0:
                                        col_cth_cal_wat[lim, idx] = cal_cth[idx, lev]
                                        col_cth_img_wat[lim, idx] = img_cth[idx]
                                    break

    return col_cth_cal_all, col_cth_cal_ice, col_cth_cal_wat, col_cth_img_all, col_cth_img_ice, col_cth_img_wat