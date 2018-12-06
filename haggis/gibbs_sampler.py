from __future__ import print_function, division, absolute_import

import numpy as np

from scipy.special import logsumexp


class gibbs_sampler(object):
    """ A class to perform generalised Gibbs sampling.

    Parameters
    ----------

    lnlike : function
        Function which returns the log-likelihood for the model.

    prior_transform : function
        Function which transforms from the unit cube to prior volume.

    n_dim : int
        Dimensionality (number of free parameters) for the fitted model.

    cdf_resolution : float, optional
        Tuning parameter. The required resolution for the cumulative
        distribution function.

    n_start_grid : int, optional
        Tuning parameter. The initial number of grid points for the cdf.

    n_split_grid : int, optional
        Tuning parameter. Number of points to split the cdf grid points
        into when insufficient sampling is found.
    """

    def __init__(self, lnlike, prior_transform, n_dim, cdf_resolution=0.025,
                 n_start_grid=10, n_split_grid=2):

        self.lnlike = lnlike
        self.prior_transform = prior_transform
        self.n_dim = n_dim

        self.cdf_resolution = cdf_resolution
        self.n_start_grid = n_start_grid
        self.n_split_grid = n_split_grid

    def run(self, n_steps, start_param=None, verbose=False):
        """ Run the sampler.

        Parameters
        ----------

        n_steps : int
            The number of Gibbs sampling steps to take in each param.

        start_param : array_like, optional
            Parameter values from which to start Gibbs sampler.

        verbose : bool, optional
            Whether to print sampler progress.
        """

        self.n_like = 0

        # Construct an array to contain the samples.
        samples = np.zeros((n_steps+1, self.n_dim))

        # Set the start parameters if values have been passed.
        if start_param is not None:
            samples[0, :] = start_param

        # Perform Gibbs sampling.
        for i in range(n_steps):

            # Choose a random order in which to sample the parameters.
            order = np.random.choice(np.arange(self.n_dim).astype(int),
                                     size=self.n_dim, replace=False)

            # For each parameter, sample the conditional pdf.
            for j in range(self.n_dim):
                p = order[j]
                sample = samples[i, :]
                sample[p] = self._conditional(sample, p)

            samples[i+1, :] = sample

            if verbose:
                print("Step:", i, "Efficiency:",
                      np.round(100.*(i+1)/self.n_like, 4), "%")

        self.samples = samples

    def _conditional(self, param, param_no):
        """ Sample the conditional distribution for parameter param_no.
        This works by taking gridding the probability density function
        to construct the cumulative distribution function, from which a
        sample is then drawn. """

        # Find the values at which to grid the pdf in cube space.
        self.edges = np.linspace(0., 1., self.n_start_grid+1)
        self.vals = None
        self.pdf = None
        self.cdf = None

        self._update_distributions(param, param_no)
        new_edges = self._get_new_edges()

        if np.max(np.isnan(self.cdf)):
            raise ValueError("No probability found, increase n_start_grid.")

        # Iteratively generate cdfs until desired sampling is reached.
        while not np.all(new_edges == self.edges):
            self.edges = new_edges
            self._update_distributions(param, param_no)
            new_edges = self._get_new_edges()

        # Sample the pdf in cube space and return the sample.
        cube = np.zeros(self.n_dim) + 0.01
        cube[param_no] = np.interp(np.random.rand(), self.cdf, self.edges)

        return self.prior_transform(cube)[param_no]

    def _update_distributions(self, param, param_no):
        """ Evaluate the cumulative distribution function for parameter
        param_no at the (cube) values specified in the edges array. """

        vals = (self.edges[1:] + self.edges[:-1])/2.
        widths = self.edges[1:] - self.edges[:-1]
        n_points = vals.shape[0]

        # Set up necessary arrays for the parameter values, pdf and cdf.
        param = np.copy(param)
        pdf = np.zeros_like(vals)
        cdf = np.zeros_like(self.edges)

        # Calculate values for the pdf on a grid.
        for i in range(n_points):

            # Check if likelihood at this point has already been found.
            if self.vals is not None and vals[i] in self.vals:
                index = self.vals.tolist().index(vals[i])
                pdf[i] = self.pdf[index]
                cube = np.zeros(self.n_dim) + 0.01
                cube[param_no] = vals[i]
                param[param_no] = self.prior_transform(cube)[param_no]

            # If not, compute the likelihood.
            else:
                cube = np.zeros(self.n_dim) + 0.01
                cube[param_no] = vals[i]
                param[param_no] = self.prior_transform(cube)[param_no]
                pdf[i] = self.lnlike(param)
                self.n_like += 1

        # Calculate the cdf by normalising and summing the pdf.
        cdf[1:] = np.cumsum(widths*np.exp(pdf - logsumexp(pdf, b=widths)))

        self.vals = vals
        self.pdf = pdf
        self.cdf = cdf

    def _get_new_edges(self):
        """ Find proposed new cdf bin edges given the previous cdf. """

        diffs = self.cdf[1:] - self.cdf[:-1]

        new_edges = [0.]

        # Propose new bin edges to try to achieve desired cdf sampling.
        for i in range(len(diffs)):
            if diffs[i] > self.cdf_resolution:
                new_edges = new_edges[:-1]
                new_edges += np.linspace(self.edges[i], self.edges[i+1],
                                         self.n_split_grid+1).tolist()

            else:
                new_edges.append(self.edges[i+1])

        return np.array(new_edges)
