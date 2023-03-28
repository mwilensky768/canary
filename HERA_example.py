import matplotlib.pyplot as plt
import os
from canary import canary
import argparse
import warnings
from scipy.stats import norm
from scipy.linalg import toeplitz, block_diag
import numpy as np

"""
An example script that generates some data and draws samples according to some
user-set parameters.
"""

def sim_wrapper(Ntimes=20, Nblps=100, noise_cov=1, corr_scale=10, cov_func=norm, sys_var=1, scale_prior_aff=None,
                Niter=int(1e5), df_prior_aff=None, frac_sys=1, do_affsamps=True, s0=np.full((100, 20), 1.0),
                aff0=np.ones(100), var0=1e-2, Caff0=np.eye(20), save=True, load=False,
                outdir="inv_wishart", mode="real"):

    prefix = f"{outdir}/sys_jackknife_"
    tag = f"Ntimes_{Ntimes}_Nblps_{Nblps}_noise_cov_{noise_cov}_corr_scale_{corr_scale}_"
    tag += f"sys_var_{sys_var}_df_prior_aff_{df_prior_aff}_frac_sys_{frac_sys}_do_affsamps_{do_affsamps}_var0_{var0}"

    num_no_sys = int((1 - frac_sys) * Nblps)
    noise_cov_val = noise_cov
    noise_cov = np.full((Nblps, Ntimes), noise_cov)
    if scale_prior_aff is None:
        scale_prior_aff = np.eye(Ntimes)

    times = np.arange(Ntimes)
    if mode == "real":
        cov_row = cov_func(scale=corr_scale).pdf(times)
        Csys = toeplitz(sys_var * cov_row / np.amax(cov_row))
        true_sys = np.random.multivariate_normal(mean=np.zeros(Ntimes), cov=Csys, size=Nblps)

    elif mode == "constant_phase":
        tag += "_constant_phase"
        # No sqrt(2) because the noise is defined with std = 1 and already split over real/imag
        stan_bl = (np.random.normal(size=Nblps) + 1.j * np.random.normal(size=Nblps))
        sys_bl = np.sqrt(sys_var) * stan_bl
        time_shape = (Ntimes // 2, Ntimes // 2)
        Csys = sys_var * np.block([[np.ones(time_shape), np.zeros(time_shape)],
                                  [np.zeros(time_shape), np.ones(time_shape)]])

        true_sys = np.zeros((Nblps, Ntimes), dtype=float)
        true_sys[:, :(Ntimes // 2)] = np.repeat(sys_bl[:, np.newaxis].real, Ntimes // 2, axis=1)
        true_sys[:, (Ntimes // 2):] = np.repeat(sys_bl[:, np.newaxis].imag, Ntimes // 2, axis=1)
    else:
        raise ValueError(f"{mode} is an invalid choice of mode. Valid choices are 'real' ir 'constant_phase'")
    true_sys[:num_no_sys, :] = 0
    noise = np.random.multivariate_normal(mean=np.zeros(Ntimes),
                                          cov=noise_cov_val * np.eye(Ntimes),
                                          size=Nblps)
    data = true_sys + noise


    if load:
        sys_samps = np.load(f"{prefix}_sys_samps_{tag}.npy")
        Caff_samps = np.load(f"{prefix}_Caff_samps_{tag}.npy")
        var_samps = np.load(f"{prefix}_var_samps_{tag}.npy")
        aff_samps = np.load(f"{prefix}_aff_samps_{tag}.npy")
        data = np.load(f"{prefix}_data_{tag}.npy")
        true_sys = np.load(f"{prefix}_true_sys_{tag}.npy")
    else:
        can = canary(data, 1/noise_cov, df_prior_aff=df_prior_aff, scale_prior_aff=scale_prior_aff)
        sys_samps, Caff_samps, var_samps, aff_samps = can.get_post_samps(s0, Caff0, var0, aff0,
                                                                         Niter=Niter, do_affsamps=do_affsamps)

        if save:
            np.save(f"{prefix}_sys_samps_{tag}.npy", sys_samps)
            np.save(f"{prefix}_Caff_samps_{tag}.npy", Caff_samps)
            np.save(f"{prefix}_var_samps_{tag}.npy", var_samps)
            np.save(f"{prefix}_aff_samps_{tag}.npy", aff_samps)
            np.save(f"{prefix}_data_{tag}.npy", data)
            np.save(f"{prefix}_true_sys_{tag}.npy", true_sys)

    return sys_samps, Caff_samps, var_samps, aff_samps, Csys, data, true_sys, tag

def make_data_plots(outdir, tag, data, true_sys, sys_samps, mode='real'):
    Ntimes = data.shape[1]
    Nblps = data.shape[0]
    if mode == "real":
        fig, ax = plt.subplots(ncols=3, figsize=(24, 12))
    else:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(24, 24))
    blind = 0
    bls = [0, int(0.5 * Nblps), Nblps - 1]
    for blind_ind, blind in enumerate(bls):

        if mode == "real":
            ax[blind_ind].set_title(f"Blp {blind}", fontsize=20)
            ax[blind_ind].plot(true_sys.T[:, blind], label="Injected systematic")
            ax[blind_ind].plot(data.T[:, blind], label="Simulated Data (sys + noise)")
            ax[blind_ind].plot(np.mean(sys_samps, axis=0)[blind], label="Mean (Posterior)")
            quants = np.quantile(sys_samps[:, blind], [norm.cdf(-2), norm.cdf(2)], axis=0)
            ax[blind_ind].fill_between(np.arange(Ntimes), *quants, color="tab:green", alpha=0.25,
                                       label="2$\sigma$ confidence")
            ax[blind_ind].set_xlabel("Time Step", fontsize=20)
            ax[blind_ind].set_ylabel("Data Value (arbs)", fontsize=20)
            ax[blind_ind].tick_params(labelsize=16)
            ax[blind_ind].legend(fontsize=16)
        else:

            ax[0, blind_ind].plot(true_sys.T[:(Ntimes // 2), blind], label="Re(Injected systematic)")
            ax[0, blind_ind].plot(data.T[:(Ntimes // 2), blind], label="Re(Simulated Data) (sys + noise)")
            ax[0, blind_ind].plot(np.mean(sys_samps, axis=0)[blind, :(Ntimes // 2)], label="Re(Mean(Posterior))")
            quants = np.quantile(sys_samps[:, blind, :(Ntimes // 2)], [norm.cdf(-2), norm.cdf(2)], axis=0)
            ax[0, blind_ind].fill_between(np.arange(Ntimes // 2), *quants, color="tab:green", alpha=0.25,
                                       label="2$\sigma$ confidence, Re")

            ax[1, blind_ind].plot(true_sys.T[(Ntimes // 2):, blind], label="Im(Injected systematic)")
            ax[1, blind_ind].plot(data.T[(Ntimes // 2):, blind], label="Im(Simulated Data) (sys + noise)")
            ax[1, blind_ind].plot(np.mean(sys_samps, axis=0)[blind, (Ntimes // 2):], label="Im(Mean(Posterior))")
            quants = np.quantile(sys_samps[:, blind, (Ntimes // 2):], [norm.cdf(-2), norm.cdf(2)], axis=0)
            ax[1, blind_ind].fill_between(np.arange(Ntimes // 2), *quants, color="tab:green", alpha=0.25,
                                       label="2$\sigma$ confidence, Im")

            for row_ind in [0, 1]:
                ax[row_ind, blind_ind].set_title(f"Blp {blind}", fontsize=20)
                ax[row_ind, blind_ind].set_xlabel("Time Step", fontsize=20)
                ax[row_ind, blind_ind].set_ylabel("Data Value (arbs)", fontsize=20)
                ax[row_ind, blind_ind].tick_params(labelsize=16)
                ax[row_ind, blind_ind].legend(fontsize=16)

    fig.savefig(f"{outdir}/sys_realization_plots_{tag}.png")
    plt.close(fig)

def make_cov_plots(outdir, tag, Csys, sys_cov_samps, vmin=0, vmax=1, cmap='inferno', mode='real'):
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    ax[0].imshow(Csys, vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title("Injected Systematic Cov.")

    im = ax[1].imshow(np.mean(sys_cov_samps, axis=0), vmin=vmin, vmax=vmax, cmap=cmap)
    ax[1].set_title("Mean Covariance Sample")
    fig.colorbar(im, ax=ax.ravel().tolist())
    for ax_ob in ax:
        if mode == 'real':
            ax_ob.set_ylabel("Time Step")
            ax_ob.set_xlabel("Time Step")
        else:
            ax_ob.set_ylabel("Block Time Step")
            ax_ob.set_xlabel("Block Time Step")

    fig.savefig(f"{outdir}/cov_matr_plots_{tag}.png")
    plt.close(fig)

def make_corr_samps(sys_cov_samps):
    Ntimes = sys_cov_samps.shape[-1]
    diag_samps = sys_cov_samps[:, np.arange(Ntimes), np.arange(Ntimes)]
    denom = np.sqrt(np.einsum('ij,ik->ijk', diag_samps, diag_samps))
    corr_samps = sys_cov_samps / denom

    return corr_samps

def make_cov_hists(outdir, tag, sys_cov_samps, Csys, time_inds_tup=((0, 0), (5, 5), (0, 1), (0, 10))):
    fig, ax = plt.subplots(ncols=3, figsize=(24, 8))
    ax = ax.ravel()


    corr_samps = make_corr_samps(sys_cov_samps)

    ax[0].hist([sys_cov_samps[:, time_inds_tup[0][0], time_inds_tup[0][1]],
                sys_cov_samps[:, time_inds_tup[1][0], time_inds_tup[1][1]]],
               bins="auto", histtype="step",
               label=[f"{time_inds_tup[0]}", f"{time_inds_tup[1]}"])
    ax[0].set_title("Diagonal Covariance Entries", fontsize=20)
    ax[0].set_xlabel("Matrix Value (arbs)", fontsize=20)
    ax[0].axvline(Csys[time_inds_tup[0][0], time_inds_tup[0][1]], color='black', linestyle='--')


    for samp_ind, samps in enumerate([sys_cov_samps, corr_samps]):

        ax[samp_ind + 1].hist([samps[:, time_inds_tup[2][0], time_inds_tup[2][1]],
                               samps[:, time_inds_tup[3][0], time_inds_tup[3][1]]],
                              bins="auto", histtype="step",
                              label=[f"{time_inds_tup[2]}", f"{time_inds_tup[3]}"])

    ax[1].axvline(Csys[time_inds_tup[2][0], time_inds_tup[2][1]], color='tab:blue', linestyle='--')
    ax[1].axvline(Csys[time_inds_tup[3][0], time_inds_tup[3][1]], color='tab:orange', linestyle='--')

    ax[2].axvline(Csys[time_inds_tup[2][0], time_inds_tup[2][1]] / Csys[0,0], color='tab:blue', linestyle='--')
    ax[2].axvline(Csys[time_inds_tup[3][0], time_inds_tup[3][1]] / Csys[0,0], color='tab:orange', linestyle='--')

    ax[1].set_title("Off-Diagonal Covariance Entries", fontsize=20)
    ax[2].set_title("Off-Diagonal Corrleation Coefficients", fontsize=20)
    for ax_ob in ax[:2]:
        ax_ob.set_xlabel("Matrix Value (arbs)", fontsize=20)
    ax[2].set_xlabel("Correlation Coefficient", fontsize=20)

    for ax_ob in ax:
        ax_ob.set_ylabel("Counts", fontsize=20)
        ax_ob.tick_params(labelsize=16)
        ax_ob.legend(fontsize=20)
    fig.savefig(f"{outdir}/cov_hists_{tag}.png")
    plt.close(fig)

def make_aff_means_plot(outdir, tag, aff_samps, frac_sys):
    aff_means = np.mean(aff_samps, axis=0)
    Nblps = len(aff_means)

    plt.figure(figsize=(6, 3))

    print(f"Number of baslines with p > 0.5: {np.count_nonzero(aff_means > 0.5)}")
    print(f"Number of baslines with p > 0.09: {np.count_nonzero(aff_means > 1/11)}")
    bls = [0, int(0.5 * Nblps), Nblps - 1]
    plt.plot(aff_means, marker='o', linestyle="None")
    plt.plot(bls, aff_means[bls], marker='o', linestyle="None", color="tab:red")
    plt.axhline(0.5, linestyle='--', color='black')
    plt.axvline((1 - frac_sys) * Nblps, linestyle='--', color='black')
    plt.ylabel("Probability of systematic presence")
    plt.xlabel("Baseline-pair index")

    plt.savefig(f"{outdir}/aff_means_{tag}.png")
    plt.close()


def full_wrapper(Ntimes=20, Nblps=100, noise_cov=1, corr_scale=10, cov_func=norm, sys_var=1, scale_prior_aff=None,
                 Niter=int(1e5), df_prior_aff=None, frac_sys=1, do_affsamps=True, s0=np.full((100, 20), 1.0),
                 aff0=np.ones(100), var0=1e-2, Caff0=np.eye(20), save=True,
                 outdir="inv_wishart", load=False, burn_ind=int(1e2), mode="real"):

    sys_samps, Caff_samps, var_samps, aff_samps, Csys, data, true_sys, tag = sim_wrapper(Ntimes=Ntimes, Nblps=Nblps,
                                                                                         noise_cov=noise_cov,
                                                                                         corr_scale=corr_scale,
                                                                                         cov_func=cov_func,
                                                                                         sys_var=sys_var,
                                                                                         scale_prior_aff=scale_prior_aff,
                                                                                         Niter=Niter,
                                                                                         df_prior_aff=df_prior_aff,
                                                                                         frac_sys=frac_sys,
                                                                                         do_affsamps=do_affsamps,
                                                                                         s0=s0,
                                                                                         aff0=aff0,
                                                                                         var0=var0, Caff0=Caff0,
                                                                                         save=save, outdir=outdir,
                                                                                         load=load, mode=mode)
    make_aff_means_plot(outdir, tag, aff_samps[burn_ind:], frac_sys)
    make_cov_plots(outdir, tag, Csys, Caff_samps[burn_ind:], vmax=sys_var)
    make_cov_hists(outdir, tag, Caff_samps[burn_ind:], Csys)
    make_data_plots(outdir, tag, data, true_sys, sys_samps, mode=mode)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, action="store", default=6027431,
                     required=False, help="Set the random seed.")
parser.add_argument("--Ntimes", type=int, action="store", default=20,
                    required=False,
                    help="Number of times in the simulation. "
                    "If doing complex mode, half the times are used for each complex component.")
parser.add_argument("--Nblps", type=int, action="store", default=100,
                    required=False, help="Number of baseline-pairs to simulate.")
parser.add_argument("--noise_var", type=float, action="store", default=1.,
                    required=False,
                    help="Noise variance. Assumed same for every blp, although "
                    "this is not necessary in the base code.")
parser.add_argument("--corr_scale", type=float, action="store", default=10.,
                    required=False,
                    help="Correlation length of simulated systematic.")
parser.add_argument("--sys_var", type=float, action="store", default=1.,
                    required=False, help="Systematic variance.")
parser.add_argument("--Niter", type=int, action="store", default=int(1e5),
                    required=False, help="Number of joint samples to generate.")
parser.add_argument("--frac_sys", type=float, action="store", default=1.,
                    required=False, help="Fraction of affected baseline-pairs.")
parser.add_argument("--outdir", type=str, action="store", default="./",
                    required=False, help="Directory for stored outputs.")
parser.add_argument("--save", action="store_true",
                    required=False, help="Whether to save the outputs.")
parser.add_argument("--load", action="store_true", required=False,
                    help="Whether to load the outputs rather than generate them.")
parser.add_argument("--comp", action="store_true",  required=False,
                    help="Whether to do a complex simulation. Default does a real-valued one.")
parser.add_argument("--burn_ind", action="store", type=int, required=False,
                    default=int(5e4), help="How many samples to burn.")
args = parser.parse_args()

if __name__ == "__main__":
    if (not args.save) and (not args.load):
        warnings.warn("Outputs are being simulated without being saved."
                      "Use --save at the command line to save outputs.")
    if args.burn_ind > args.Niter:
        warnings.warn("burn_ind is larger than Niter, so nothing will be plotted. "
                      "Check arguments for HERA_example script.")

    np.random.seed(args.seed)
    aff0 = np.random.randint(2, size=args.Nblps)
    s0 = np.random.normal(size=(args.Nblps, args.Ntimes))
    mode = "constant_phase" if args.comp else "real"
    full_wrapper(Ntimes=args.Ntimes, Nblps=args.Nblps, noise_cov=args.noise_var,
                 corr_scale=args.corr_scale, sys_var=args.sys_var, Niter=args.Niter,
                 frac_sys=args.frac_sys, outdir=args.outdir, save=args.save,
                 load=args.load, burn_ind=args.burn_ind, mode=mode, s0=s0, aff0=aff0)
