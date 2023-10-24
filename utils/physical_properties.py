'''
Author: Nina Prakash
Copied from ml-image-segmentation, SHA b38ff99806ca85d4a3c2314d100806674fdd1a44
'''


import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes, mesh_surface_area, perimeter
from skimage.segmentation import find_boundaries
import taufactor as tau


def compute_tau_and_D_eff(segmentation, verbose="per_iter", conv_crit=0.02):
    """
    Use TauFactor to compute tau and D_eff from a segmented volume.

    Args:
        segmentation (np.ndarray)       Two-phase microstructed image volume.
                                        1 represents the conductive phase and 0
                                        otherwise.

        verbose (str, bool)             Argument to TauFactor solver. Whether
                                        or not to print progress.

        conv_crit (float)               Argument to TauFactor solver. Min
                                        percent difference between max and min flux
                                        through a given layer

    Return:
        taufactor_dict (dict)           Dictionary with keys "tau", "D_eff", and
                                        "D_rel".
    """

    s = tau.Solver(segmentation)
    s.solve(verbose=verbose, conv_crit=conv_crit)

    taufactor_dict = {
        "tau": s.tau,
        "D_eff": s.D_eff,
        "D_rel": s.D_rel,
    }

    return taufactor_dict


def volume_fraction(segmentation, phase_class=0):
    """
    Compute the volume fraction of the phase in a 2D image
    or volume based on its segmentation.

    Args:
        segmentation (np.ndarray)   The segmented image
        phase_class (int)           The label for the phase of interest

    Return:
        Volume fraction in range [0, 1]
    """

    phase_volume = (segmentation == phase_class).sum()
    total_volume = segmentation.size

    return phase_volume / total_volume


def total_surface_area(segmentation, particle_class=None):
    """
    Given a 2D segmented image or a segmented volume,
    compute the total surface area normalized by the volume.

    If a particle class is passed in, then return the total
    surface area at the particle surfaces.
    Otherwise, return the surface area between any two phases.

    """

    if particle_class is not None:
        # Binarize (1 = particle, 0 = everything else)
        seg = np.zeros(segmentation.shape)
        seg[segmentation == particle_class] = 1
        seg = seg.squeeze()

        # Remove internal holes
        seg_filled = ndimage.morphology.binary_fill_holes(seg)

        # Using skimage find_boundaries
        boundaries = find_boundaries(seg_filled, mode="inner", background=0)

    else:
        boundaries = find_boundaries(segmentation.squeeze())

    sa = boundaries.sum()
    volume = boundaries.size

    return sa / volume


def calculate_sphere_cdf(r, ndim=2):
    d = np.linspace(0, r, num=1200)
    if ndim == 2:
        C_edm = (r - d) ** 2 / r * 2
    else:
        C_edm = (r - d) ** 3 / r**3
    return d, C_edm


def particle_size(segmentation, particle_label=1):
    """
    Given a 2D segmented image or a segmented volume,
    compute the distribution of particle diameters.
    """

    # Binarize (1 = particle, 0 = everything else)
    seg = np.zeros(segmentation.shape)
    seg[segmentation == particle_label] = 1
    segmentation = segmentation.squeeze()

    # Compute the Euclidean distance map on this binary segmentation.
    # i.e. for each pixel, get distance to nearest 0-valued pixel.
    distance_map = ndimage.distance_transform_edt(segmentation)
    distance_map = distance_map.flatten()
    nonzero_distances = distance_map[distance_map > 0]
    nonzero_distances_rounded = nonzero_distances.round()

    # Get the CDF of these distances as the cumulative sum of normalized frequencies
    x_empirical, frequencies = np.unique(nonzero_distances_rounded, return_counts=True)
    pdf = frequencies / len(nonzero_distances_rounded)
    cdf_empirical = 1 - np.cumsum(pdf)

    # Fit CDF of spheres/circles with known radius
    r_min = 0.05  # In MATBOX, this is 1/2 the voxel size. I'm doing this without the unit conversion for now
    r_max = 2 * x_empirical.max()
    num_r = 400
    rs = np.linspace(r_min, r_max, num_r)

    errors = []
    for r in rs:
        x_analytical, cdf_analytical = calculate_sphere_cdf(r)

        # Set the two CDFs to be the same range
        if r < x_empirical.max():
            # If the analytical radius is less than the empirical max distance,
            # extend the range of the analytical distances.
            x_empirical_ = x_empirical
            cdf_empirical_ = cdf_empirical

            x_analytical_ = np.append(x_analytical, x_empirical.max())
            cdf_analytical_ = np.append(cdf_analytical, 0)

        elif r > x_empirical.max():
            # If the analytical radius is greater than the empirical max distance,
            # extend the range of the empirical distances.
            x_empirical_ = np.append(x_empirical, r)
            cdf_empirical_ = np.append(cdf_empirical, 0)

            x_analytical_ = x_analytical
            cdf_analytical_ = cdf_analytical

        else:
            # If the analytical radius is the same as the empirical max distance,
            # then both have the same range. Nothing to do here.
            pass

        # Compute error between empirical and analytical CDFs
        common_xaxis = np.linspace(0, x_empirical_.max(), num=1200)
        empirical_ = np.interp(common_xaxis, x_empirical_, cdf_empirical_)
        analytical_ = np.interp(common_xaxis, x_analytical_, cdf_analytical_)

        difference = analytical_ - empirical_
        area_under_difference = np.trapz(difference)
        errors.append(abs(area_under_difference))

    # Get the radius with smallest error
    best_r = rs[np.argmin(errors)]
    best_D = 2 * best_r
    return best_D


def circularity_sphericity(segmentation):
    # TODO
    pass


# Other Properties to Potentially Implement
## fractal dimension
## grain orientation
## skeletoning on an image
## hydraulic diameter vs. min/max diameter
