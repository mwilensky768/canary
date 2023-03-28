import numpy as np
from scipy.stats import invwishart, multivariate_normal, invgamma
from scipy.linalg import cholesky

class canary:

    def __init__(self, data, ninv, diag_noise=True,
                 scale_prior_aff=None, df_prior_aff=None, scale_prior_unaff=None,
                 df_prior_unaff=None):

        """
        Class that holds the data and does the Gibbs sampling.

        Parameters:
            data (array_like):
                Array of data from which the inference is drawn.
                Should have shape (num_draw, dat_dim), where num_draw is the
                number of multivariate draws in question and dat_dim is the
                dimension of each multivariate draw.

            ninv (array_like):
                Inverse of the noise covariance matrix. If diag_noise is True
                then this may have shape (num_draw, dat_dim), otherwise this should
                have shape (num_draw, dat_dim, dat_dim). If diag_noise is True and a
                shape of (num_draw, dat_dim, dat_dim) is supplied, will extract diagonal entries.

            diag_noise (bool):
                Whether the noise covariance matrix is diagonal or not. Will
                make the calculations much faster if True.

            scale_prior_aff (array_like):
                Scale matrix for the inverse Wishart prior on the systematic
                covariance. Default is identity.

            df_prior_aff (int):
                Degrees of freedom parameter for the inverse Wishart prior on
                the systematic covariance. Default is dat_dim + 1, which is the
                lowest allowed value while still being a proper prior. Higher
                value creates a sharper prior.

            df_prior_unaff (int):
                Degrees of freedom parameter for the inverse Gamma prior on the
                unaffected data. A higher number creates a sharper prior. This
                prior should be fairly sharp around a small value so that it
                represents a very weak systematic (i.e. no systematic).

            scale_prior_unaff (float):
                Scale parameter for the inverse Gamma prior on the unaffected
                data. This should be set in correspondence with df_prior unaff
                such that the mode of the prior is on a small value relative to
                the noise variance.
        """
        self.data = np.copy(data)
        if len(data.shape) != 2:
            raise ValueError("Supplied data array must have dimension of 2.")
        self.num_draw, self.dat_dim = data.shape

        self.diag_noise = diag_noise
        self.ninv, self.ninv_cho = self._setup_noise(ninv, diag_noise)


        if scale_prior_aff is None: # uninformative
            self.scale_prior_aff = np.eye(self.dat_dim)
        else:
            self.scale_prior_aff = scale_prior_aff
        if df_prior_aff is None: # uninformative
            self.df_prior_aff = self.dat_dim + 1
        else:
            self.df_prior_aff = df_prior_aff


        if df_prior_unaff is None:
            self.df_prior_unaff = 1
        else:
            self.df_prior_unaff = df_prior_unaff
        if scale_prior_unaff is None:
            self.scale_prior_unaff = 1e-4 * self.df_prior_unaff
        else:
            self.scale_prior_unaff = scale_prior_unaff

    def _setup_noise(self, ninv, diag_noise):
        """
        Set up the noise attributes during initialization.

        Parameters:
            ninv (array_like):
                Inverse of the noise covariance matrix. If diag_noise is True
                then this may have shape (dat_dim), otherwise this should
                have shape (dat_dim, dat_dim). If diag_noise is True and a
                shape of (dat_dim, dat_dim) is supplied, will extract diagonal entries.

            diag_noise (bool):
                Whether the noise covariance matrix is diagonal or not. Will
                make the calculations much faster if True.

        Returns:
            ninv (array_like):
                Reshaped noise inverse according to diag_noise argument.
            ninv_cho (array_like:
                Cholesky decomposition of the inverse noise covariance.
                Calculated with elementwise square root if diag_noise,
                else uses scipy.linalg.cholesky.
            ninvd (array_like:
                Inverse-noise-covariance weighted data.
        """
        valid_shapes = ((self.num_draw, self.dat_dim,), (self.num_draw, self.dat_dim, self.dat_dim))
        if ninv.shape not in valid_shapes:
            raise ValueError(f"ninv shape, {ninv.shape}, does not match data shape, "
                             f"{self.data.shape}. Check inputs.")

        if diag_noise:
            if len(ninv.shape) == 3:
                ninv_ret = np.zeros(self.num_draw, self.dat_dim)
                for drind in range(self.num_draw):
                    ninv_ret[drind] = np.diag(ninv[drind])
            else:
                ninv_ret = ninv
            ninv_cho = np.sqrt(ninv)
        else:
            ninv_cho = np.zeros_like(ninv)
            for drind in range(self.num_draw):
                ninv_cho[drind] = cholesky(ninv[drind], lower=True)
            ninv_ret = ninv

        return ninv_ret, ninv_cho




    def get_sys_cov_samp(self, sys_current, num_class, aff_mode):
        """
        Get a covariance or variance sample based on the current systematic
        realizations and which data are considered as contaminated.

        Parameters:
            sys_current (array_like):
                The current systematic realizations.

            num_class (int):
                Number of data vectors falling into whichver class is being sampled.

            aff_mode (bool):
                Whether sampling from "affected" class (True) or "unaffected"
                class (False).

        Returns:
            sys_cov_samp (array_like or float):
                A covariance or variance sample depending on which class is
                being sampled.
        """


        if aff_mode:
            df = self.df_prior_aff
            scale = self.scale_prior_aff
            scatmat = np.einsum('ij,ik->jk', sys_current, sys_current, optimize=True)
            sys_cov_samp = invwishart(df=df + num_class, scale=scale + scatmat).rvs()
        else:
            df = self.df_prior_unaff
            scale = self.scale_prior_unaff
            sum_squares = np.sum(sys_current**2)
            sys_cov_samp = invgamma(df + num_class*self.dat_dim / 2, scale=sum_squares/2 + scale).rvs()


        return sys_cov_samp

    def get_sys_samp(self, sys_cov_current, data_use, ninv_use, ninv_cho_use):
        """
        Get the systematic realizations based on the current class labels and
        systematic covariances.

        Parameters:
            sys_cov_current (array or int):
                Current sysematic covariance or variance sample.
            data_use (array):
                data being used for samples
            ninv_use (array):
                Associated inverse noise for data being
                used
            ninv_cho_use (array):
                Associated inverse noise cholesky decomposition for data being
                used

        Returns:
            sys_samp (array): Systematic realization samples.
        """
        num_data_use = data_use.shape[0]
        sys_cov_cho = cholesky(sys_cov_current, lower=True)
        flx = np.random.normal(size=(2, num_data_use, self.dat_dim))

        if self.diag_noise:
            cov_ninv = sys_cov_current * ninv_use[:, np.newaxis] # num_draw, dat_dim, dat_dim
            rhs_flx0 = (sys_cov_current * (ninv_cho_use * flx[0])[:, np.newaxis]).sum(axis=2)
        else:
            cov_ninv = sys_cov_current @ ninv_use # num_draw, dat_dim, dat_dim
            cho_term = (ninv_cho_use * flx[0][:, np.newaixs]).sum(axis=2) # num_draw, dat_dim
            rhs_flx0 = (sys_cov_current[np.newaxis] * cho_term[:, np.newaxis]).sum(axis=2)

        rhs_dat = (cov_ninv * data_use[:, np.newaxis]).sum(axis=2)
        rhs_flx1 = (sys_cov_cho * flx[1][:, np.newaxis]).sum(axis=2)

        rhs_vec = rhs_dat + rhs_flx0 + rhs_flx1
        lhs_op = cov_ninv + np.eye(self.dat_dim) # num_draw, dat_dim, dat_dim

        sys_samp = np.linalg.solve(lhs_op, rhs_vec)

        return sys_samp

    def get_aff_current(self, sys_current, Caff_current, var_current):
        """
        Draw a sample from the distribution of possible classification realizations
        based on the current systematic realizations and covariances/variance samples.

        Parameters:
            sys_current (array): Current systematic realizations
            Caff_current (array): Current sample for systematic covariance
            var_current (array): Current variance sample for unaffected data.

        Returns:
            aff_samp (array):
                Array of bools indicating which data are classified as
                contaminated (True) or uncontaminated (False) for the current
                sample.
        """

        loglikes_aff = multivariate_normal(mean=np.zeros(self.dat_dim), cov=Caff_current).logpdf(sys_current)
        loglikes_unaff = multivariate_normal(mean=np.zeros(self.dat_dim), cov=var_current * np.eye(self.dat_dim)).logpdf(sys_current)

        # FIXME: Needs prior prob if not flat prior!
        log_odds_aff = loglikes_aff - loglikes_unaff
        # this is odds_aff / (odds_aff + 1)
        p_affs = 1 / (np.exp(-log_odds_aff) + 1)

        aff_samp = (np.random.uniform(size=self.num_draw) < p_affs)

        return aff_samp


    def get_post_samps(self, s0, Caff0, var0, aff0, Niter=int(1e5), do_affsamps=False):
        """
        Get samples from the joint posterior of the systematic realizations,
        systematic covariance, systematic variance for unaffected data
        (should always be nearly 0), and classification realizations using Gibbs
        sampling.

        Parameters:
            s0 (array):
                Initial sample for systematic realizations in each data vector.
            Caff0 (array):
                Initial sample for systematic covariance
            Niter (int):
                Number of joint samples to draw
            do_affsamps (bool):
                Whether to classify the data into affected and unaffected classes.
                True means classify, else assume every data vector is affected
                and skip classification steps (much faster, but inaccurate covariance
                estimation when only some of the data is contaminated).

        Returns:
            sys_samps (array):
                Systematic realization samples
            Caff_samps (array):
                Systematic covariance realization samples
            var_samps (array):
                Samples for systematic variance of unaffected data. This is
                basically a dummy variable since in theory it should always be
                zero, however using a delta-function prior produces numerical
                instabilities.
            aff_samps (array):
                Samples of which data vectors are considered as contaminated in
                each iteration.
        """
        sys_samps = np.zeros((Niter, self.num_draw, self.dat_dim))
        Caff_samps = np.zeros((Niter, self.dat_dim, self.dat_dim))
        var_samps = np.zeros(Niter)
        aff_samps = np.zeros((Niter, self.num_draw))

        Caff_current = Caff0
        var_current = var0
        num_sys = self.num_draw
        sys_current = s0
        aff_current = aff0
        for iter_ind in range(Niter):
            if do_affsamps:

                not_aff_current = np.logical_not(aff_current)
                num_aff_current = aff_current.sum()
                # Update sys
                sys_current[aff_current] = self.get_sys_samp(Caff_current,
                                                             self.data[aff_current],
                                                             self.ninv[aff_current],
                                                             self.ninv_cho[aff_current])
                sys_current[not_aff_current] = self.get_sys_samp(var_current * np.eye(self.dat_dim),
                                                                 self.data[not_aff_current],
                                                                 self.ninv[not_aff_current],
                                                                 self.ninv_cho[not_aff_current])
                sys_samps[iter_ind] = sys_current

                # Update covariances
                Caff_current = self.get_sys_cov_samp(sys_current[aff_current], num_aff_current, aff_mode=True)
                Caff_samps[iter_ind] = Caff_current


                var_current = self.get_sys_cov_samp(sys_current[not_aff_current],
                                                    self.num_draw - num_aff_current,
                                                    aff_mode=False)
                var_samps[iter_ind] = var_current

                # Update affected
                aff_current = self.get_aff_current(sys_current, Caff_current, var_current)
                aff_samps[iter_ind] = aff_current

            else:
                sys_current = self.get_sys_samp(Caff_current, self.data)
                sys_samps[iter_ind] = sys_current

                Caff_current = self.get_sys_cov_samp(sys_current, self.num_draw, aff_mode=True)
                Caff_samps[iter_ind] = Caff_current

        return sys_samps, Caff_samps, var_samps, aff_samps
