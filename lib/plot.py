#
# Quick diagnostic plots.
#
# This file is part of PINTS.
#  Copyright (c) 2017-2018, University of Oxford.
#  For licensing information, see the LICENSE file distributed with the PINTS
#  software package.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import warnings
import numpy as np
import scipy.stats as stats
import pints

def function(f, x, lower=None, upper=None, evaluations=20):
    """
    Creates 1d plots of a :class:`LogPDF` or a :class:`ErrorMeasure` around a
    point `x` (i.e. a 1-dimensional plot in each direction).

    Arguments:

    ``f``
        A :class:`pints.LogPDF` or :class:`pints.ErrorMeasure` to plot.
    ``x``
        A point in the function's input space.
    ``lower``
        (Optional) Lower bounds for each parameter, used to specify the lower
        bounds of the plot.
    ``upper``
        (Optional) Upper bounds for each parameter, used to specify the upper
        bounds of the plot.
    ``evaluations``
        (Optional) The number of evaluations to use in each plot.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check function get dimension
    if not (isinstance(f, pints.LogPDF) or isinstance(f, pints.ErrorMeasure)):
        raise ValueError(
            'Given function must be pints.LogPDF or pints.ErrorMeasure.')
    dimension = f.n_parameters()

    # Check point
    x = pints.vector(x)
    if len(x) != dimension:
        raise ValueError(
            'Given point `x` must have same dimension as function.')

    # Check boundaries
    if lower is None:
        # Guess boundaries based on point x
        lower = x * 0.95
        lower[lower == 0] = -1
    else:
        lower = pints.vector(lower)
        if len(lower) != dimension:
            raise ValueError(
                'Lower bounds must have same dimension as function.')
    if upper is None:
        # Guess boundaries based on point x
        upper = x * 1.05
        upper[upper == 0] = 1
    else:
        upper = pints.vector(upper)
        if len(upper) != dimension:
            raise ValueError(
                'Upper bounds must have same dimension as function.')

    # Check number of evaluations
    evaluations = int(evaluations)
    if evaluations < 1:
        raise ValueError('Number of evaluations must be greater than zero.')

    # Create points to plot
    xs = np.tile(x, (dimension * evaluations, 1))
    for j in range(dimension):
        i1 = j * evaluations
        i2 = i1 + evaluations
        xs[i1:i2, j] = np.linspace(lower[j], upper[j], evaluations)

    # Evaluate points
    fs = pints.evaluate(f, xs, parallel=False)

    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(12, 7))
    for j, p in enumerate(x):
        i1 = j * evaluations
        i2 = i1 + evaluations
        a1 = j % 4
        a2 = j // 4
        axes[a1, a2].plot(xs[i1:i2, j], fs[i1:i2], c='green', label='Function')
        axes[a1, a2].axvline(p, c='blue', label='Value')
        axes[a1, a2].set_xlabel('Parameter ' + str(1 + j))
        if j == 0:
            axes[a1, a2].legend()

    plt.tight_layout()
    return fig, axes


def pairwise(samples,
             kde=False,
             heatmap=False,
             opacity=None,
             ref_parameters=None,
             n_percentiles=None,
             fig_axes=None):
    """
    Takes a markov chain or list of `samples` and creates a set of pairwise
    scatterplots for all parameters (p1 versus p2, p1 versus p3, p2 versus p3,
    etc.).

    The returned plot is in a 'matrix' form, with histograms of each individual
    parameter on the diagonal, and scatter plots of parameters ``i`` and ``j``
    on each entry ``(i, j)`` below the diagonal.

    Arguments:

    ``samples``
        A list of samples, with shape ``(n_samples, dimension)``, where
        ``n_samples`` is the number of samples in the list and ``dimension`` is
        the number of parameters.
    ``kde``
        (Optional) Set to ``True`` to use kernel-density estimation for the
        histograms and scatter plots.
    ``opacity``
        (Optional) When ``kde=False``, this value can be used to manually set
        the opacity of the points in the scatter plots.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # Check samples size
    try:
        n_sample, n_param = samples.shape
    except ValueError:
        raise ValueError('`samples` must be of shape (n_sample, n_param).')

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')

    # Create figure
    fig_size = (3 * n_param, 3 * n_param)
    if fig_axes is None:
        fig, axes = plt.subplots(n_param, n_param, figsize=fig_size)
    else:
        fig, axes = fig_axes

    bins = 25
    for i in range(n_param):
        for j in range(n_param):
            if i == j:

                # Diagonal: Plot a histogram
                if n_percentiles is None:
                    xmin, xmax = np.min(samples[:, i]), np.max(samples[:, i])
                else:
                    xmin = np.percentile(samples[:, i],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, i],
                                         50 + n_percentiles / 2.)
                xbins = np.linspace(xmin, xmax, bins)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].hist(samples[:, i], bins=xbins, normed=True)

                # Add kde plot
                if kde:
                    x = np.linspace(xmin, xmax, 100)
                    axes[i, j].plot(x, stats.gaussian_kde(samples[:, i])(x))

                # Add reference parameters if given
                if ref_parameters is not None:
                    ymin_tv, ymax_tv = axes[i, j].get_ylim()
                    axes[i, j].plot(
                        [ref_parameters[i], ref_parameters[i]],
                        [0.0, ymax_tv],
                        '--', c='k', lw=2)

            elif i < j:
                # Top-right: no plot
                axes[i, j].axis('off')

            else:
                # Lower-left: Plot the samples as density map
                if n_percentiles is None:
                    xmin, xmax = np.min(samples[:, j]), np.max(samples[:, j])
                    ymin, ymax = np.min(samples[:, i]), np.max(samples[:, i])
                else:
                    xmin = np.percentile(samples[:, j],
                                         50 - n_percentiles / 2.)
                    xmax = np.percentile(samples[:, j],
                                         50 + n_percentiles / 2.)
                    ymin = np.percentile(samples[:, i],
                                         50 - n_percentiles / 2.)
                    ymax = np.percentile(samples[:, i],
                                         50 + n_percentiles / 2.)
                axes[i, j].set_xlim(xmin, xmax)
                axes[i, j].set_ylim(ymin, ymax)

                if not kde and not heatmap:
                    # Create scatter plot

                    # Determine point opacity
                    num_points = len(samples[:, i])
                    if opacity is None:
                        if num_points < 10:
                            opacity = 1.0
                        else:
                            opacity = 1.0 / np.log10(num_points)

                    # Scatter points
                    axes[i, j].scatter(
                        samples[:, j], samples[:, i], alpha=opacity, s=0.1)

                elif kde:
                    # Create a KDE-based plot

                    # Plot values
                    values = np.vstack([samples[:, j], samples[:, i]])
                    axes[i, j].imshow(
                        np.rot90(values), cmap=plt.cm.Blues,
                        extent=[xmin, xmax, ymin, ymax])

                    # Create grid
                    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])

                    # Get kernel density estimate and plot contours
                    kernel = stats.gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    axes[i, j].contourf(xx, yy, f, cmap='Blues')
                    axes[i, j].contour(xx, yy, f, colors='k')

                    # Force equal aspect ratio
                    # See: https://stackoverflow.com/questions/7965743
                    im = axes[i, j].get_images()
                    ex = im[0].get_extent()
                    # Matplotlib raises a warning here (on 2.7 at least)
                    # We can't do anything about it, so no other option than
                    # to suppress it at this stage...
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', UnicodeWarning)
                        axes[i, j].set_aspect(
                            abs((ex[1] - ex[0]) / (ex[3] - ex[2])))

                elif heatmap:
                    # Create a heatmap-like plot
                    xbins = np.linspace(xmin, xmax, bins)
                    ybins = np.linspace(ymin, ymax, bins)
                    axes[i, j].hist2d(samples[:, j], samples[:, i],
                                    bins=[xbins, ybins], normed=True,
                                    cmap='Blues')

                # Add reference parameters if given
                if ref_parameters is not None:
                    axes[i, j].plot(
                        [ref_parameters[j], ref_parameters[j]],
                        [ymin, ymax],
                        '--', c='k', lw=2)
                    axes[i, j].plot(
                        [xmin, xmax],
                        [ref_parameters[i], ref_parameters[i]],
                        '--', c='k', lw=2)

            # Set tick labels
            if i < n_param - 1:
                # Only show x tick labels for the last row
                axes[i, j].set_xticklabels([])
            else:
                # Rotate the x tick labels to fit in the plot
                for tl in axes[i, j].get_xticklabels():
                    tl.set_rotation(45)

            if j > 0:
                # Only show y tick labels for the first column
                axes[i, j].set_yticklabels([])

        # Set axis labels
        axes[-1, i].set_xlabel('Parameter %d' % (i + 1))
        if i == 0:
            # The first one is not a parameter
            axes[i, 0].set_ylabel('Frequency')
        else:
            axes[i, 0].set_ylabel('Parameter %d' % (i + 1))

    return fig, axes


def hist(samples, ref_parameters=None, n_percentiles=None):
    """
    Takes one or more markov chains or lists of samples as input and creates
    and returns a plot showing histograms and traces for each chain or list of
    samples.

    Arguments:

    ``samples``
        A list of lists of samples, with shape
        ``(n_lists, n_samples, dimension)``, where ``n_lists`` is the number of
        lists of samples, ``n_samples`` is the number of samples in one list
        and ``dimension`` is the number of parameters.
    ``ref_parameters``
        (Optional) A set of parameters for reference in the plot. For example,
        if true values of parameters are known, they can be passed in for
        plotting.
    ``n_percentiles``
        (Optional) Shows only the middle n-th percentiles of the distribution.
        Default shows all samples in ``samples``.

    Returns a ``matplotlib`` figure object and axes handle.
    """
    import matplotlib.pyplot as plt

    # If we switch to Python3 exclusively, bins and alpha can be keyword-only
    # arguments
    bins = 40
    alpha = 0.5
    n_list = len(samples)
    _, n_param = samples[0].shape

    # Check number of parameters
    for samples_j in samples:
        if n_param != samples_j.shape[1]:
            raise ValueError(
                'All samples must have the same number of parameters.'
            )

    # Check reference parameters
    if ref_parameters is not None:
        if len(ref_parameters) != n_param:
            raise ValueError(
                'Length of `ref_parameters` must be same as number of'
                ' parameters.')

    # Set up figure
    fig, axes = plt.subplots(4, 3, figsize=(12, 7))

    # Plot first samples
    for i in range(n_param):
        a1 = i % 4
        a2 = i // 4
        for j_list, samples_j in enumerate(samples):
            # Add histogram subplot
            axes[a1, a2].set_xlabel('Parameter ' + str(i + 1))
            if n_percentiles is None:
                xmin = np.min(samples_j[:, i])
                xmax = np.max(samples_j[:, i])
            else:
                xmin = np.percentile(samples_j[:, i],
                                     50 - n_percentiles / 2.)
                xmax = np.percentile(samples_j[:, i],
                                     50 + n_percentiles / 2.)
            xbins = np.linspace(xmin, xmax, bins)
            axes[a1, a2].hist(samples_j[:, i], bins=xbins, alpha=alpha)
                              # label='Samples ' + str(1 + j_list))

        # Add reference parameters if given
        if ref_parameters is not None:
            # For histogram subplot
            ymin_tv, ymax_tv = axes[i, 0].get_ylim()
            axes[a1, a2].plot(
                [ref_parameters[i], ref_parameters[i]],
                [0.0, ymax_tv],
                '--', c='k')

    if n_list > 1:
        axes[0, 0].legend()
    axes[1, 0].set_ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    return fig, axes
