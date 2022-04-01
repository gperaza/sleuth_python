import numpy as np
import constants as C
from functools import reduce
from scipy.ndimage import convolve, label, center_of_mass


def spread(sng, sdc, og, rt, pop, urban):

    return avg_slope, num_growth_pix, sng, sdc, og, rt, pop


def phase1n3(grid, coeffs, critical, year, prng):
    """ Spontaneous growth and new urban centers. """

    assert C.MIN_DIFFUSION <= coeffs['diffusion'] <= C.MAX_DIFFUSION
    assert C.MIN_BREED <= coeffs['breed'] <= C.MAX_BREED

    nrows, ncols = grid.slope.shape
    # Diffusion value is at most 5% of image diagonal
    # This defines the number of candidate pixels to urbanize
    diffusion_val = ((coeffs['diffusion'] * 0.005)
                     * np.sqrt(nrows**2 + ncols**2))

    # Get candidate pixel coordinates, ignore borders
    # In orginal SLEUTH borders are ignore likely to avoid
    # treating neighbor search as a special case
    coords = prng.integers((nrows-1, ncols-1),
                           size=diffusion_val)

    # Check which ones are available for urbanization,
    # several checks must pass
    coords, fails = urbanize(coords, grid,
                             coeffs, critical,
                             year, prng)
    # Log successes
    urb_successes = len(coords)

    # Spontaneous growth
    # Update available pixels in delta grid
    grid.delta.values[coords[:, 0], coords[:, 1]] = C.PHASE1G

    # Grow new urban centers wit probability given by breed
    # Filter candidate new centers
    coords = coords[prng.integers(101, size=len(coords))
                    < coeffs['breed']]
    for i, j in coords:
        # get urbanizable neighbors, 2 max
        ncoords, fails = urbanize_nghbrs(i, j, grid,
                                         coeffs, critical,
                                         year, prng)
        urb_successes += len(ncoords)
        # Update z grid with values for phase 3
        grid.delta.values[ncoords[:, 0], ncoords[:, 1]] = C.PHASE3G


def phase4(grid, coeff, prng):
    """ Edge growth """

    # SLEUTH searches for neighbors of URBANIZED cells, and
    # if two or more neighbors are urbanized, a random neighbor is
    # selected for potential urbanization
    # This incurss in two random tests, the breed coefficient test,
    # and the neighbor sampling. It is possible that the neighbor
    # chosen is already urbanized, and the compound probability
    # of a succesful urbanization of a non-uban pixel is found from
    # p_breed*(# urb_nnbr_clusters) * p_chosen(1/8)
    # On the other hand if we look direcly at the non-urban pixels
    # this is no longer "edge" growth, as its neighbors may not be
    # connected.

    # Loop over pixels or convolution? Original SLEUTH loops over
    # pixels, so convolution can't be worse. But potential improvement
    # if a set of urban pixels is mantained.
    kernel = np.array([[1,  1, 1],
                       [1, -8, 1],
                       [1,  1, 1]])

    # Non urban pixels have neighbor values 8-16, while urban
    # pixels have 0-8
    n_nbrs = convolve(grid.z.values, kernel,
                      mode='constant', output=int) + 8

    # Spread centers are urban pixels wirh 2-7 neighbors
    # Obtain their indices
    sprd_centers = np.logical_and(n_nbrs >= 2, n_nbrs <= 7)
    sprd_ci, sprd_cj = np.where(sprd_centers)

    # Apply breed test
    mask = prng.integers(101, size=len(sprd_ci)) < coeff['spread']
    sprd_ci, sprd_cj = sprd_ci[mask], sprd_cj[mask]

    # Choose a random neighbor and attempt urbanization


def urbanize(coords, grid, coeffs, critical, year, prng):
    """ Determine wether pixels are subject to urbanization.  """

    ic, jc = coords[:, 0], coords[:, 1]
    z = grid.Z.sel(year=year).values[ic, jc]
    delta = grid.delta.values[ic, jc]
    slope = grid.slope.values[ic, jc]
    excld = grid.excluded.values[ic, jc]

    # Check if not already urbanized
    z_mask = (z == 0)
    delta_mask = (delta == 0)

    # Apply slope restrictions
    # Sweights give the probability of rejecting urbanization
    sweights = slope_weight(slope,
                            coeffs['slope_resistance'],
                            critical['slope_resistance'])
    slp_mask = (prng.random(size=len(sweights)) >= sweights)

    # Apply excluded restrictions, excluded values >= 100 are
    # completely unavailable, 0 are always available
    # excld value is the 100*probability of rejecting urbanization
    excld_mask = (prng.integers(100, size=len(excld)) >= excld)

    z_fails = (~z_mask).sum()
    delta_fails = (~delta_mask).sum()
    slope_fails = (~slp_mask).sum()
    excld_fails = (~excld_mask).sum()

    mask = reduce(np.logical_and,
                  [z_mask, delta_mask, slp_mask, excld_mask])

    fails = {'z': z_fails, 'delta': delta_fails,
             'slope': slope_fails, 'excluded': excld_fails}

    return coords[mask], fails


def urbanize_nghbrs(i, j, grid, coeffs, critical, year, prng):
    """ Attempt to urbanize the neiggborhood of (i,j).

    Neighbors are chosen in random order until two successful
    urbanizations or all neighbors have been chosen. """

    # Original SLEUTH code samples up to 8 neighbors with replacement,
    # meaning a neighbor can be chosen more than onceand urbanization
    # may fail even if available neighbors exist.
    # Here we try all neighbors and choose 2 urbanizable ones.

    nlist = (i, j) + np.array(((-1, -1), (0, -1), (+1, -1), (+1, 0),
                               (+1, +1), (0, +1), (-1, +1), (-1, 0)))

    # Obtain urbanizable neighbors
    nlist, fails = urbanize(nlist, grid, coeffs, critical, year, prng)
    if len(nlist) > 2:
        nlist = prng.choice(nlist, 2, replace=False)

    return nlist, fails


def slope_weight(slope, slp_res, critical_slp):
    # if slope >= critical_slp:
    #     return 1.0
    # else:
    #     val = ((critical_slp - slope)/critical_slp)
    #     exp = (2*slp_res/C.MAX_SLOPE_RESISTANCE)
    #     return 1.0 - val**exp

    slope = np.asarray(slope)
    val = ((critical_slp - slope)/critical_slp)
    exp = (2*slp_res/C.MAX_SLOPE_RESISTANCE)
    slope_w = np.where(slope >= critical_slp, 0, val)
    slope_w = 1.0 - slope_w**exp

    return slope_w
