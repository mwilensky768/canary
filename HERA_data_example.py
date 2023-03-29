from canary import canary
from argparse import ArgumentParser
from canary.utils import make_aff_means_plot, make_cov_plots, make_data_plots, make_cov_hists
import numpy as np
from hera_cal.vis_clean import VisClean
from scipy.linalg import toeplitz
import warnings

parser = ArgumentParser()
parser.add_argument("--infile", type=str, action="store", required=True,
                    help="The data file on which to operate. Should be a uvh5 "
                    "file readable by hera_pspec.")
parser.add_argument("--outdir", type=str, action="store", required=False,
                    default='./',
                    help="The output directory. Default is the working directory.")
parser.add_argument("--seed", type=int, action="store", required=False,
                    default=3849205, help="Set the random seed.")
parser.add_argument("--noise_file", type=str, action="store", required=True,
                    help="Corresponding file that contains noise covariance "
                    "info for the infile arg")
parser.add_argument("--time_inds", type=int, nargs=2, action="store", required=False,
                    default=[0, 5], help="Lower and upper time indices to process.")
parser.add_argument("--blgroup", type=str, action="store", required=True,
                    help="An argument to pick out which baseline group to do (TBD).")
parser.add_argument("--delay_ind", type=int, action="store", required=True,
                    help="Which delay index to run the test on.")
parser.add_argument("--Niter", type=int, action="store", required=False,
                    default=int(1e4), help="Number of posterior samples to draw.")
parser.add_argument("--burn_ind", type=int, action="store", required=False,
                    default=int(2e3),
                    help="Number of samples to burn when calculating statistics,"
                    " plotting, etc. All samples are always written out "
                    " regardless of this argument, it just changes what is "
                    " calculated/plotted.")
parser.add_argument("--draws", type=int, action="store", required=False,
                    default=[10, 20, 30], nargs=3, help="Which blps to plot.")
parser.add_argument("--freqs", type=float, nargs=2, action="store", required=True,
                    help="lower and upper frequencies to use in the data")
parser.add_argument("--pol", type=str, action="store", required=False,
                    default='nn', help="Which polarization to use.")
parser.add_argument("--max_lag", type=int, action="store", required=False,
                    default=2, help="Max lag to use for noise cov. estimation.")
args = parser.parse_args()

def extract_file_nickname(infile):
    left_ind = infile.rfind("/") + 1
    right_ind = infile.rfind(".")

    nickname = infile[left_ind:right_ind]

    return nickname

def read_data(infile, freqs, delay_ind, time_inds, red_group_list):
    """
    Read data from input file using desired frequencies, delay ind, times, and
    redundant baselines.

    Parameters:
        infile (str): Path to a file to read.
        freqs (array_like): Upper and lower frequency bounds for desired data.
        time_inds (array_like): Upper and lower time indices to grab from data.
        delay_ind (int): Index of the delay mode to analyze.
        red_group_list (array_like): list of tuples with two integers and a string
            specifying the two antennas and the desired visibility polarization.

    Returns:
        data (array_like): Complex array of shape (Nbls, Ntimes) containing the
            delay transformed visibilities in the infile.
    """
    # This is how data are FFT'd to delay space in the tutorial notebook, so copying that
    hd = VisClean(infile)
    freq_cond = np.logical_and(hd.freqs >= min(freqs), hd.freqs <= max(freqs))
    frequencies = hd.freqs[freq_cond]
    hd.read(frequencies=frequencies)
    hd.fft_data(ax="freq", window="bh", ifft=True, edgecut_low=0, edgecut_hi=0,
                overwrite=True)

    data = np.array([hd.dfft[bl][min(time_inds):max(time_inds) + 1, delay_ind]
                     for bl in red_group_list])

    return data

# FIXME: Ad-hoc way to estimate the noise cov. from a single long realization per bl.
def read_noise(noise_file, freqs, delay_ind, time_inds, red_group_list,
               max_lag=2):
    """
    Estimate noise covariance from input file using desired frequencies, delay
    ind, times, and redundant baselines. Assumes stationarity and linearity of
    all transforms applied to data.

    Parameters:
        infile (str): Path to a file to read.
        freqs (array_like): Upper and lower frequency bounds for desired data.
        time_inds (array_like): Upper and lower time indices to grab from data.
        delay_ind (int): Index of the delay mode to analyze.
        red_group_list (array_like): list of tuples with two integers and a string
            specifying the two antennas and the desired visibility polarization.
        max_lag (int): Maximum lag to estimate the covariance for. Lags beyond
            this are assumed to be 0. Includes this number, so default value
            does three lags: (0, 1, 2).

    Returns:
        noise_cov (array_like): Complex array of shape (Nbls, Ntimes, Ntimes)
            containing the noise covariance based on the noise realization in
            the noise_file.
    """
    noise_data = read_data(noise_file, freqs, delay_ind, time_inds, red_group_list)

    # Assumes the noise is zero-mean and stationary after FRF
    # Stationarity will not be true if there is significant variation in nsamples
    # as a fn. of time (zero-mean is fine if all preceeding transformations are linear).
    # Given this, we can estimate a few of the covariance entries using one noise realization.
    Nbls = noise_data.shape[0]
    Ntimes = noise_data.shape[1]
    if max_lag >= Ntimes:
        max_lag = Ntimes
        warnings.warn("Using max_lag that is greater than or equal to Ntimes. "
                      "Long lags will have large estimator noise.")
    noise_cov_shape = (Nbls, Ntimes, Ntimes)
    noise_cov = np.zeros(noise_cov_shape, dtype=complex)

    # Calculate a topeplitz matrix per baseline from the noise realization
    for blind in range(Nbls):
        cov_row = np.zeros(Ntimes, dtype=complex)
        for lag in range(max_lag + 1): # Assume 0 beyond a certain lag
            if lag == 0:
                cov_row[lag] = np.mean(np.abs(noise_data[blind])**2)
            else:
                dat_use = noise_data[blind, :-lag]
                dat_conj = noise_data[blind, lag:].conj()
                cov_row[lag] = np.mean(dat_use * dat_conj)
        noise_cov[blind] = toeplitz(cov_row)

    return noise_cov


if __name__ == "__main__":

    #red_group = [(176, 178), (177, 179), (181, 183), (183, 185), (160, 162),
                 #(162, 164), (163, 165), (187, 189), (189, 191), (166, 168),
                 #(167, 169), (168, 170), (141, 143), (143, 145), (117, 119),
                 #(118, 120), (122, 124), (127, 129), (128, 130), (98, 100),
                 #(102, 104), (103, 105), (109, 111), (81, 83), (83, 85),
                 #(85, 87), (92, 94), (66, 68), (50, 52), (44, 46),
                 #(11, 13), (13, 15), (15, 17), (3, 5)]
    # FIXME: Hardcoded 28m E-W group with more than a couple nights in H4C IDR2.2
    red_group = [(92, 130), (103, 143), (81, 119), (3, 29)]
    red_group_list = [bl + ("nn",) for bl in red_group]

    print("Reading in data.")
    data_use = read_data(args.infile, args.freqs, args.delay_ind, args.time_inds,
                         red_group_list)

    print("Making noise matrices.")
    noise_use = read_noise(args.noise_file, args.freqs, args.delay_ind,
                           args.time_inds, red_group_list, args.max_lag)

    print("Making differences.")
    # Must have even number of baselines
    diffs_use = data_use[1::2] - data_use[::2] # even minus odd
    diff_noise_use = noise_use[1::2] + noise_use[::2] # subtract data so add covs.
    Nblp = diffs_use.shape[0]

    print("Inverting noise matrices.")
    # get inverse noise covariances
    diff_noise_conds = np.linalg.cond(diff_noise_use)
    if np.any(diff_noise_conds > 1e9):
        diff_ninv = np.zeros_like(diff_noise_use)
        for blp_ind in range(Nblp):
            if diff_noise_conds[blp_ind] < 1e9:
                diff_ninv[blp_ind] = np.linalg.inv(diff_noise_use[blp_ind])
            else: # do pseudo-inverse, like in other pspec code
                diff_ninv[blp_ind] = np.linalg.pinv(diff_noise_use[blp_ind])
    else: # no need for fancy tricks
        diff_ninv = np.linalg.inv(diff_noise_use)

    print("Splitting by complex component.")
    # split into real/imag
    Ntimes = diffs_use.shape[1]
    diff_block = np.zeros((Nblp, 2 * Ntimes), dtype=float)
    diff_block[:, :Ntimes] = diffs_use.real
    diff_block[:, Ntimes:] = diffs_use.imag

    # Factor of 2 because we inverted before blocking
    ninv_block = 2 * np.block([[diff_ninv.real, -diff_ninv.imag], [diff_ninv.imag, diff_ninv.real]])

    print("Setting up canary object.")
    can = canary(diff_block, ninv_block, diag_noise=False)
    np.random.seed(args.seed)

    aff0 = np.random.randint(2, size=Nblp)
    s0 = np.random.normal(size=(Nblp, 2 * Ntimes))
    Caff0 = np.eye(2 * Ntimes)
    var0 = 1e-2

    print("Started sampling.")
    sys_samps, Caff_samps, var_samps, aff_samps = can.get_post_samps(s0, Caff0,
                                                                     var0, aff0,
                                                                     do_affsamps=True,
                                                                     Niter=args.Niter)

    nickname = extract_file_nickname(args.infile)
    tag = f"{nickname}_times_{min(args.time_inds)}_{max(args.time_inds)}_blgroup"
    f"_{args.blgroup}_delay_ind_{args.delay_ind}_seed_{args.seed}"
    print("Plotting.")
    make_aff_means_plot(args.outdir, tag, aff_samps[args.burn_ind:],
                        "Baseline-Pair Index", args.draws)

    make_cov_plots(args.outdir, tag, Caff_samps[args.burn_ind:], vmax=None,)
    make_cov_hists(args.outdir, tag, Caff_samps[args.burn_ind:],
                   time_inds_tup=((0, 0), (2, 2), (0, 1), (0, 2)))
    make_data_plots(args.outdir, tag, diff_block, sys_samps[args.burn_ind:],
                    args.draws, "Blp", mode="constant_phase")
