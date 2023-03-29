from canary import canary
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()
parser.add_argument("--infile", type=str, action="store", required=True,
                    help="The data file on which to operate. Should be a uvh5 "
                    "file readable by hera_pspec.")
parser.add_argument("--outdir", type=str, action="store", required=False,
                    default='./'
                    help="The output directory. Default is the working directory.")
parser.add_argument("--seed", type=int, action="store", required=False,
                    default=3849205, help="Set the random seed.")
parser.add_argument("--noise_file", type=str, action="store", required=True,
                    help="Corresponding file that contains noise covariance "
                    "info for the infile arg")
parser.add_argument("--times", type=int, nargs=2, action="store", required=False,
                    default=[0, 10], help="Lower and upper time indices to process.")
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
parser.add_argument("--draws", type-int, action="store", required=False,
                    default=[10, 20, 30], nargs=3, help="Which blps to plot.")
args = parser.parse_args()

def extract_file_nickname(infile):
    left_ind = infile.rfind("/") + 1
    right_ind = infile.rfind(".")

    nickname = infile[left_ind:right_ind]

    return nickname


if __name__ == "__main__":
    # FIXME: need a function that slices the data array to a shape (Nbls, Ntimes)
    data_use = get_data_func(args.infile, args.delay_ind, args.times)
    # FIXME: need a function that gets the noise covariance per baseline and time (Nbls, Ntimes, Ntimes)
    noise_use = get_noise_func(args.noise_file, args.times)

    # Must have even number of baselines
    diffs_use = data_use[1::2] - data_use[::2] # even minus odd
    diff_noise_use = noise_use[1::2] + noise_use[::2] # subtract data so add covs.
    Nblp = diffs_use.shape[0]

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


    # split into real/imag
    Ntimes = diffs_use.shape[1]
    diff_block = np.zeros((Nblp, 2 * Ntimes), dtype=float)
    diff_block[:, :Ntimes] = diffs_use.real
    diff_block[:, Ntimes:] = diffs_use.imag

    # Factor of 2 because we inverted before blocking
    ninv_block = 2 * np.block([[diff_ninv.real, -diff_ninv.imag], [diff_ninv.imag, diff_ninv.real]])

    can = canary(diff_block, ninv_block, diag_noise=False)
    np.random.seed(args.seed)

    aff0 = np.random.randint(2, size=Nblp)
    s0 = np.random.normal(size=(Nblp, Ntimes))
    Caff0 = np.eye(2 * Ntimes)
    var0 = 1e-2

    sys_samps, Caff_samps, var_samps, aff_samps = can.get_post_samps(s0, Caff0,
                                                                     var0, aff0,
                                                                     do_affsamps=True,
                                                                     Niter=args.Niter)

    nickname = extract_file_nickname(args.infile)
    tag = f"{nickname}_times_{min(times)}_{max(times)}_blgroup_{blgroup}_delay_ind_{delay_ind}_seed_{seed}"
    make_aff_means_plot(args.outdir, tag, aff_samps[args.burn_ind:],
                        "Baseline-Pair Index", args.draws)

    make_cov_plots(args.outdir, tag, Caff_samps[args.burn_ind:], vmax=sys_var,)
    make_cov_hists(args.outdir, tag, Caff_samps[args.burn_ind:])
    make_data_plots(args.outdir, tag, data, sys_samps[args.burn_ind:],
                    args.draws, "Blp", mode=mode)
