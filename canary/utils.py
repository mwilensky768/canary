import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def make_aff_means_plot(outdir, tag, aff_samps, xlabel, draws, frac_sys=None):
    aff_means = np.mean(aff_samps, axis=0)
    num_draw = len(aff_means)

    plt.figure(figsize=(6, 3))
    plt.plot(aff_means, marker='o', linestyle="None")
    if draws is not None:
        plt.plot(draws, aff_means[draws], marker='o', linestyle="None", color="tab:red")
    plt.axhline(0.5, linestyle='--', color='black')
    if frac_sys is not None:
        plt.axvline((1 - frac_sys) * num_draw, linestyle='--', color='black')
    plt.ylabel("Probability of systematic presence")
    plt.xlabel(xlabel)

    plt.tight_layout()
    plt.savefig(f"{outdir}/aff_means_{tag}.png")
    plt.close()

    return


def make_cov_plots(outdir, tag, sys_cov_samps, Csys=None, vmin=0, vmax=1,
                   cmap='inferno', mode='real'):
    fig, ax = plt.subplots(ncols=2, figsize=(16, 8))

    im = ax[0].imshow(np.mean(sys_cov_samps, axis=0), vmin=vmin, vmax=vmax, cmap=cmap)
    ax[0].set_title("Mean Covariance Sample")

    Csys_not_None = (Csys is not None)
    if Csys is not None:
        ax[1].imshow(Csys, vmin=vmin, vmax=vmax, cmap=cmap)
        ax[1].set_title("Injected Systematic Cov.")

        fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.02125, pad=0.04)
    else:
        fig.colorbar(im, ax=ax[0])
    for ax_ob in ax[:(1 + Csys_not_None)]:
        if mode == 'real':
            ax_ob.set_ylabel("Time Step")
            ax_ob.set_xlabel("Time Step")
        else:
            ax_ob.set_ylabel("Block Time Step")
            ax_ob.set_xlabel("Block Time Step")
            Ntimes = Csys.shape[0] // 2
            ticks = np.arange(0, 2 * Ntimes, 5)
            ticklabels = ticks % Ntimes

            ax_ob.set_xticks(ticks)
            ax_ob.set_yticks(ticks)
            ax_ob.set_xticklabels(ticklabels)
            ax_ob.set_yticklabels(ticklabels)

    fig.savefig(f"{outdir}/cov_matr_plots_{tag}.png")
    plt.close(fig)

    return


def make_data_plots(outdir, tag, data, sys_samps, draws, title, true_sys=None,
                    mode='real', units="arbs"):
    Ntimes = data.shape[1]
    num_draw = data.shape[0]
    if mode == "real":
        fig, ax = plt.subplots(ncols=3, figsize=(24, 12))
    else:
        fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(24, 24))
    blind = 0
    for blind_ind, blind in enumerate(draws):

        if mode == "real":
            ax[blind_ind].set_title(f"{title} {blind}", fontsize=20)


            ax[blind_ind].plot(data.T[:, blind], label="Simulated Data (sys + noise)")
            ax[blind_ind].plot(np.mean(sys_samps, axis=0)[blind], label="Mean (Posterior)")
            quants = np.quantile(sys_samps[:, blind], [norm.cdf(-2), norm.cdf(2)], axis=0)
            ax[blind_ind].fill_between(np.arange(Ntimes), *quants, color="tab:blue", alpha=0.25,
                                       label="2$\sigma$ confidence")
            if true_sys is not None:
                ax[blind_ind].plot(true_sys.T[:, blind], label="Injected systematic")
            ax[blind_ind].set_xlabel("Time Step", fontsize=20)
            ax[blind_ind].set_ylabel(f"Data Value ({units})", fontsize=20)
            ax[blind_ind].tick_params(labelsize=16)
            ax[blind_ind].legend(fontsize=16)
        else:



            ax[0, blind_ind].plot(np.mean(sys_samps, axis=0)[blind, :(Ntimes // 2)], label="Re(Mean(Posterior))")
            quants = np.quantile(sys_samps[:, blind, :(Ntimes // 2)], [norm.cdf(-2), norm.cdf(2)], axis=0)
            ax[0, blind_ind].fill_between(np.arange(Ntimes // 2), *quants, color="tab:blue", alpha=0.25,
                                       label="2$\sigma$ confidence, Re")
            ax[0, blind_ind].plot(data.T[:(Ntimes // 2), blind], label="Re(Simulated Data) (sys + noise)")
            if true_sys is not None:
                ax[0, blind_ind].plot(true_sys.T[:(Ntimes // 2), blind], label="Re(Injected systematic)")



            ax[1, blind_ind].plot(np.mean(sys_samps, axis=0)[blind, (Ntimes // 2):], label="Im(Mean(Posterior))")
            quants = np.quantile(sys_samps[:, blind, (Ntimes // 2):], [norm.cdf(-2), norm.cdf(2)], axis=0)
            ax[1, blind_ind].fill_between(np.arange(Ntimes // 2), *quants, color="tab:blue", alpha=0.25,
                                       label="2$\sigma$ confidence, Im")
            ax[1, blind_ind].plot(data.T[(Ntimes // 2):, blind], label="Im(Simulated Data) (sys + noise)")
            if true_sys is not None:
                ax[1, blind_ind].plot(true_sys.T[(Ntimes // 2):, blind], label="Im(Injected systematic)")

            for row_ind in [0, 1]:
                ax[row_ind, blind_ind].set_title(f"Blp {blind}", fontsize=20)
                ax[row_ind, blind_ind].set_xlabel("Time Step", fontsize=20)
                ax[row_ind, blind_ind].set_ylabel(f"Data Value ({units})", fontsize=20)
                ax[row_ind, blind_ind].tick_params(labelsize=16)
                ax[row_ind, blind_ind].legend(fontsize=16)

    plt.tight_layout()
    fig.savefig(f"{outdir}/sys_realization_plots_{tag}.png")
    plt.close(fig)

    return


def make_corr_samps(sys_cov_samps):
    Ntimes = sys_cov_samps.shape[-1]
    diag_samps = sys_cov_samps[:, np.arange(Ntimes), np.arange(Ntimes)]
    denom = np.sqrt(np.einsum('ij,ik->ijk', diag_samps, diag_samps))
    corr_samps = sys_cov_samps / denom

    return corr_samps


def make_cov_hists(outdir, tag, sys_cov_samps, Csys=None,
                   time_inds_tup=((0, 0), (5, 5), (0, 1), (0, 10)), units="arbs"):
    fig, ax = plt.subplots(ncols=3, figsize=(24, 8))
    ax = ax.ravel()


    corr_samps = make_corr_samps(sys_cov_samps)

    ax[0].hist([sys_cov_samps[:, time_inds_tup[0][0], time_inds_tup[0][1]],
                sys_cov_samps[:, time_inds_tup[1][0], time_inds_tup[1][1]]],
               bins="auto", histtype="step",
               label=[f"{time_inds_tup[0]}", f"{time_inds_tup[1]}"])
    ax[0].set_title("Diagonal Covariance Entries", fontsize=20)
    ax[0].set_xlabel(f"Matrix Value ({units})", fontsize=20)

    for samp_ind, samps in enumerate([sys_cov_samps, corr_samps]):

        ax[samp_ind + 1].hist([samps[:, time_inds_tup[2][0], time_inds_tup[2][1]],
                               samps[:, time_inds_tup[3][0], time_inds_tup[3][1]]],
                              bins="auto", histtype="step",
                              label=[f"{time_inds_tup[2]}", f"{time_inds_tup[3]}"])



    ax[1].set_title("Off-Diagonal Covariance Entries", fontsize=20)
    ax[2].set_title("Off-Diagonal Corrleation Coefficients", fontsize=20)
    if Csys is not None:
        ax[0].axvline(Csys[time_inds_tup[0][0], time_inds_tup[0][1]], color='black', linestyle='--')
        ax[1].axvline(Csys[time_inds_tup[2][0], time_inds_tup[2][1]], color='tab:blue', linestyle='--')
        ax[1].axvline(Csys[time_inds_tup[3][0], time_inds_tup[3][1]], color='tab:orange', linestyle='--')

        ax[2].axvline(Csys[time_inds_tup[2][0], time_inds_tup[2][1]] / Csys[0,0], color='tab:blue', linestyle='--')
        ax[2].axvline(Csys[time_inds_tup[3][0], time_inds_tup[3][1]] / Csys[0,0], color='tab:orange', linestyle='--')

    for ax_ob in ax[:2]:
        ax_ob.set_xlabel(f"Matrix Value ({units})", fontsize=20)
    ax[2].set_xlabel("Correlation Coefficient", fontsize=20)

    for ax_ob in ax:
        ax_ob.set_ylabel("Counts", fontsize=20)
        ax_ob.tick_params(labelsize=16)
        ax_ob.legend(fontsize=20)
    fig.savefig(f"{outdir}/cov_hists_{tag}.png")
    plt.close(fig)

    return

def save_samps(outdir, tag, sys_samps, Caff_samps, var_samps, aff_samps):

    np.save(f"{outdir}/canary_sys_samps_{tag}.npy", sys_samps)
    np.save(f"{outdir}/canary_Caff_samps_{tag}.npy", Caff_samps)
    np.save(f"{outdir}/canary_var_samps_{tag}.npy", var_samps)
    np.save(f"{outdir}/canary_aff_samps_{tag}.npy", aff_samps)

    return
