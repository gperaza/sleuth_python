import numpy as np
import constants as C
from functools import reduce
from scipy.ndimage import convolve
from timer import timers
from logconf import logger
from grid import igrid_init

"""
This module implmentes the 4 phases of urban growth:
- phase1n3: Spontaneous growth and new urban centers
- phase4: Edge growth
- phase5: Road influenced growth
"""


def temp_driver(data_dir):

    grid = igrid_init(data_dir)
    prng = np.random.default_rng()

    coef_diffusion = 10
    coef_breed = 50
    coef_spread = 50
    coef_slope = 10
    coef_road = 50
    crit_slope = 50
    np.copyto(grid.Z.values, grid.urban.sel(year=2019))
    for year in range(2020, 2040):
        spread(
            grid.Z.values, grid.delta.values, grid.slope.values,
            grid.excluded.values, grid.roads.values, grid.dist.values,
            grid.road_i.values, grid.road_j.values,
            coef_diffusion, coef_breed, coef_spread,
            coef_slope, coef_road, crit_slope, prng)
        grid.Z.rio.to_raster(data_dir / f'sleuth_slow_{year}.tif')

    coef_diffusion = 20
    coef_breed = 75
    coef_spread = 75
    coef_slope = 10
    coef_road = 75
    crit_slope = 50
    np.copyto(grid.Z.values, grid.urban.sel(year=2019))
    for year in range(2020, 2040):
        spread(
            grid.Z.values, grid.delta.values, grid.slope.values,
            grid.excluded.values, grid.roads.values, grid.dist.values,
            grid.road_i.values, grid.road_j.values,
            coef_diffusion, coef_breed, coef_spread,
            coef_slope, coef_road, crit_slope, prng)
        grid.Z.rio.to_raster(data_dir / f'sleuth_usual_{year}.tif')

    coef_diffusion = 30
    coef_breed = 100
    coef_spread = 100
    coef_slope = 10
    coef_road = 100
    crit_slope = 50
    np.copyto(grid.Z.values, grid.urban.sel(year=2019))
    for year in range(2020, 2040):
        spread(
            grid.Z.values, grid.delta.values, grid.slope.values,
            grid.excluded.values, grid.roads.values, grid.dist.values,
            grid.road_i.values, grid.road_j.values,
            coef_diffusion, coef_breed, coef_spread,
            coef_slope, coef_road, crit_slope, prng)
        grid.Z.rio.to_raster(data_dir / f'sleuth_fast_{year}.tif')


def grow(proc_type,
         grid,
         coeffs,
         crit_slope,
         start_year, stop_year,
         current_run, total_runs,
         current_mc, total_mc):
    """Loop over simulated years.

    Loop over simulated years.

    """
    timers.GROW.start()

    if proc_type == 'predicting':
        current_year = 2021  # Get prediction start date
    else:
        current_year = 1990  # Get urban year 0

    # Initialize Z grid
    # Why SLEUTH aleays initializes to urban index 0?
    # Does it takes care of these during initialization?
    np.copyto(grid.Z.values, grid.urban.sel(year=current_year))

    logger.info('******************************************')
    if proc_type == 'calibrating':
        logger.info(f'Run = {current_run} of {total_runs}'
                    f' ({100*current_run/total_runs:8.1f} percent complete)')
    logger.info(f'Monter Carlo = {current_mc} of {total_mc}')
    logger.info(f'Current year = {current_year}')
    logger.info(f'Stop year = {stop_year}')

    while current_year < stop_year:
        current_year += 1
        logger.info(f'  {current_year}/{stop_year}')

        # Apply CA rules for current year
        sng = sdg = sdc = og = rt = pop = 0
        timers.SPR_TOTAL.start()
#        sng, sdg, sdc, og, rt, pop, num_grow_pix, avg_slope = spread(
        _ = spread(
            grid.Z.values, grid.delta.values, grid.slope.values,
            grid.excluded.values, grid.roads.values, grid.dist.values,
            grid.road_i.values, grid.road_j.values,
            coeffs['diffusion'], coeffs['breed'], coeffs['spread'],
            coeffs['slope'], coeffs['road'], crit_slope)
        timers.SPR_TOTAL.stop()

        # Do satistics

        # Do self modification

    timers.GROW.stop()


def spread(grd_Z, grd_delta, grd_slope, grd_excluded,
           grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
           coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
           crit_slope,
           prng):

    # Clean up delta grid
    grd_delta.fill(0)

    # Slope coef and crit are constant during a single step
    # Precalculate all slope weights?

    timers.SPR_PHASE1N3.start()
    phase1n3(grd_Z, grd_delta, grd_slope, grd_excluded,
             coef_diffusion, coef_breed, coef_slope,
             crit_slope,
             prng)
    timers.SPR_PHASE1N3.stop()

    timers.SPR_PHASE4.start()
    phase4(grd_Z, grd_delta, grd_slope, grd_excluded,
           coef_spread, coef_slope,
           crit_slope,
           prng)
    timers.SPR_PHASE4.stop()

    timers.SPR_PHASE5.start()
    phase5(grd_Z, grd_delta, grd_slope, grd_excluded,
           grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
           coef_road, coef_diffusion, coef_breed, coef_slope,
           crit_slope,
           prng)
    timers.SPR_PHASE5.stop()

    # Urbanize in Z array for accumulated urbanization.
    mask = grd_delta > 0
    grd_Z[mask] = grd_delta[mask]
    avg_slope = grd_slope[mask].mean()
    num_growth_pix = mask.sum()
    pop = (grd_Z >= C.PHASE0G).sum()

    return avg_slope, num_growth_pix, pop


def phase1n3(grd_Z, grd_delta, grd_slope, grd_excluded,
             coef_diffusion, coef_breed, coef_slope,
             crit_slope,
             prng):
    """ Spontaneus growth and possible new urban centers.

    This function implements the first two stages of growth, the appearance
    of new urban cells at random anywhere on the grid, and the possible
    urbanization of two of their neighbors to create new urban centers.

    Parameters
    ----------
    grid : XArray
        Contains all required rasters.
    coeffs : 5
        Contains all coefficients that control growth.
    critical : 7
        8
    year : 9
        10
    prng : 11
        The random number generator.

    """

    assert C.MIN_DIFFUSION <= coef_diffusion <= C.MAX_DIFFUSION
    assert C.MIN_BREED <= coef_breed <= C.MAX_BREED

    nrows, ncols = grd_slope.shape
    # Diffusion value is at most 5% of image diagonal
    # This defines the number of candidate pixels to urbanize
    diffusion_val = int((coef_diffusion * 0.005)
                        * np.sqrt(nrows**2 + ncols**2))

    # Get candidate pixel coordinates, ignore borders
    # In orginal SLEUTH borders are ignore likely to avoid
    # treating neighbor search as a special case
    coords = prng.integers((1, 1), (nrows-1, ncols-1),
                           size=(diffusion_val, 2))

    # Check which ones are available for urbanization,
    # several checks must pass
    mask, fails = urbanizable(
        coords,
        grd_Z, grd_delta, grd_slope, grd_excluded,
        coef_slope,
        crit_slope,
        prng)
    coords = coords[mask]
    # Log successes
    urb_successes = len(coords)

    # Spontaneous growth
    # Update available pixels in delta grid
    grd_delta[coords[:, 0], coords[:, 1]] = C.PHASE1G

    # Grow new urban centers wit probability given by breed
    # Filter candidate new centers
    coords = coords[prng.integers(101, size=len(coords))
                    < coef_breed]
    for i, j in coords:
        # Original SLEUTH code samples up to 8 neighbors with
        # replacement, meaning a neighbor can be chosen more than
        # once and urbanization may fail even if available neighbors
        # exist. Here we try all neighbors and choose 2 urbanizable
        # ones.

        # get urbanizable neighbors
        ncoords, mask, fails = urbanizable_nghbrs(
            i, j,
            grd_Z, grd_delta, grd_slope, grd_excluded,
            coef_slope,
            crit_slope,
            prng)
        # choose two urbanizable neighbors
        ncoords = ncoords[mask][:2]
        urb_successes += len(ncoords)
        # Update delta grid with values for phase 3
        grd_delta[ncoords[:, 0], ncoords[:, 1]] = C.PHASE3G


def phase4(grd_Z, grd_delta, grd_slope, grd_excluded,
           coef_spread, coef_slope,
           crit_slope,
           prng):
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
    # Here we do it differently, we look only for urban centers, and
    # apply the spread test to urban centers, instead to all urban pixels.
    # That way the interpretation of the spread coef is more closely
    # related to the fraction of urban CENTERS that attempt urbanization,
    # not to the fraction of total urban pixels that attempt urbanization.

    # Loop over pixels or convolution? Original SLEUTH loops over
    # pixels, so convolution can't be worse. But potential improvement
    # if a set of urban pixels is mantained.
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])

    # Non urban pixels have neighbor values 8-16, while urban
    # pixels have 0-8
    # Taking advantage that slices are views in numpy
    # we can ignore edges by creating the return array and
    # using the output parameter.
    n_nbrs = np.zeros_like(grd_Z, dtype=np.int32)
    convolve(grd_Z[1:-1, 1:-1], kernel,
             mode='constant', output=n_nbrs[1:-1, 1:-1])
    n_nbrs[1:-1, 1:-1] += 8

    # Spread centers are urban pixels wirh 2-7 neighbors
    # Obtain their indices
    sprd_centers = np.logical_and(n_nbrs >= 2, n_nbrs <= 7)
    sprd_ci, sprd_cj = np.where(sprd_centers)

    # Apply breed test, for coef=100, all growth is accepted
    # Breed coef is the percentage of urban centers (pixels)
    #  that will attempt to urbanize a neighbor.
    mask = prng.integers(101, size=len(sprd_ci)) <= coef_spread
    sprd_ci, sprd_cj = sprd_ci[mask], sprd_cj[mask]

    # Choose a random neighbor and attempt urbanization
    # This tries to urbanize a random neighbor, and may fail
    # even if a different neighbor might succeed.
    # We can change this behavious and always urbanize if there
    # is an available neighbor.
    # If we want to keep the original behaviour we do not need
    # the whole neighbor list, and this step can be optimized.
    for i, j in zip(sprd_ci, sprd_cj):
        ncoords, mask, fails = urbanizable_nghbrs(
            i, j,
            grd_Z, grd_delta, grd_slope, grd_excluded,
            coef_slope,
            crit_slope,
            prng)
        if mask[0]:
            # Urbanize in delta grid
            grd_delta[ncoords[0, 0], ncoords[0, 1]] = C.PHASE4G


def phase5(grd_Z, grd_delta, grd_slope, grd_excluded,
           grd_roads, grd_road_dist, grd_road_i, grd_road_j,
           coef_road, coef_diffusion, coef_breed, coef_slope,
           crit_slope,
           prng):
    """ Road influenced growth. """

    assert coef_road > 0
    assert coef_diffusion > 0
    assert coef_breed > 0

    # calculate tha maximum distance to search for a road
    # this is the chebyshev distance and is precomputed in grid
    # maxed at ~ 1/32 image perimeter
    nrows, ncols = grd_delta.shape
    max_dist = int(coef_road/C.MAX_ROAD
                   * (nrows + ncols)/16.0)
    # number of neighbors up to distance max_dist
    # useless in our code, but original SLEUTH uses this to
    # spiral search the neighborhood of a pixel
    # nneighbors_at_d = 4 * road_gravity * (1 + road_gravity)

    # In this phase we search urbanization on delta grid,
    # not z grid, meaning only new urbanized cells from previous
    # phase are considered.
    # From new growth cells, a fraction is selected according to the
    # breed coefficient.
    # In original SLEUTH this is a sample with replacement, which
    # wasteful, and the difference is likely to be absorbed into the
    # calibrated value of the coefficient, so we may want to consider
    # sample without replacement.
    # Though a more thorough analysis of the probabilistic
    # model of SLEUTH is warranted.
    # The problem of sampling without replacement is that having less
    # candidates than breed_coef will result in an error, in such a case,
    # we may return the whole set of candidates.

    # Coordinates of new growth
    coords = np.array(np.where(grd_delta > 0)).T

    # Select n=breed growth candidates
    coords = prng.choice(
        coords, size=int(coef_breed) + 1, replace=True)

    # Search for nearest road, and select only if road is close enough
    dists = grd_road_dist[coords[:, 0], coords[:, 1]]
    coords = coords[dists <= max_dist]
    road_i = grd_road_i[coords[:, 0], coords[:, 1]]
    road_j = grd_road_j[coords[:, 0], coords[:, 1]]
    rcoords = np.column_stack([road_i, road_j])

    # For selected roads perform a random walk and attempt urbanization
    nlist = np.array(((-1, -1), (0, -1), (+1, -1), (+1, 0),
                      (+1, +1), (0, +1), (-1, +1), (-1, 0)))

    # Here we apply road search as defined in patch_01, which is
    # actually never implmented in official SLEUTH code, despite
    # the official site claiming otherwise.
    # See:
    # http://www.ncgia.ucsb.edu/projects/gig/Dnload/dn_describe3.0p_01.html
    new_sites = np.zeros_like(rcoords)
    for i, rc in enumerate(rcoords):
        for step in range(int(coef_diffusion)):
            prng.shuffle(nlist)
            nbrs = rc + nlist
            nbrs = nbrs[grd_roads[nbrs[:, 0], nbrs[:, 1]] > 0]
            rc = nbrs[0]
        new_sites[i] = rc

    # Apply urbanization test based on road weights
    mask = prng.integers(100) < grd_roads[new_sites[:, 0], new_sites[:, 1]]
    new_sites = new_sites[mask]

    # Attempt to urbanize
    mask, fails = urbanizable(
        new_sites,
        grd_Z, grd_delta, grd_slope, grd_excluded,
        coef_slope,
        crit_slope,
        prng)
    new_sites = new_sites[mask]
    # Log successes
    urb_successes = len(new_sites)
    # Update available pixels in delta grid
    grd_delta[new_sites[:, 0], new_sites[:, 1]] = C.PHASE5G

    # Attempt to create new urban centers, urbanize 2 neighbors
    for i, j in new_sites:
        # get urbanizable neighbors
        ncoords, mask, fails = urbanizable_nghbrs(
            i, j,
            grd_Z, grd_delta, grd_slope, grd_excluded,
            coef_slope,
            crit_slope,
            prng)
        # choose two urbanizable neighbors
        ncoords = ncoords[mask][:2]
        urb_successes += len(ncoords)
        # Update delta grid with values for phase 3
        grd_delta[ncoords[:, 0], ncoords[:, 1]] = C.PHASE5G


def urbanizable(coords,
                grd_Z, grd_delta, grd_slope, grd_excluded,
                coef_slope,
                crit_slope,
                prng):
    """ Determine wether pixels are subject to urbanization.

    Pixels subject to urbanization are not already urbanized pixels that
    pass the tests of slope and the exclution region.

    Parameters
    ----------
    coords : np.array
        Numpy array of pair of (i, j) coordinates of candidate pixels.
    grd_z: np.array
        2D binary array with original urbanization for the current step.
    grd_delta: np.array
        2D binary array where new urbanization is temporary created.
    grd_slope: np.array
        2D int array with slope values normalized to 1-100 range.
    grd_excluded: np.array
        2D int array with excluded zones. Values are 100*probability of
        rejecting urbanization. 0 is always available, over 100 means
        urbanization  is always rejected.
    coef_slope : float
        Slope coefficient controlling the probability of rejecting
        urbanization due to a steep slope.
    crit_slope : float
        The slope treshold above wich urbanization is always rejected.
    prng : TODO
        The random number generator class instance.

    Returns
    -------

    mask: np.array
        1D boolean mask for array of candidate coordinates, True if available
        for urbanization.
    fails: dict
        Dictionary with number of candidates not available for urbanization
        per grid.
    """

    # Extract vectors of grid values for candidate pixels.
    # TODO: handle missing grids
    ic, jc = coords[:, 0], coords[:, 1]
    z = grd_Z[ic, jc]
    delta = grd_delta[ic, jc]
    slope = grd_slope[ic, jc]
    excld = grd_excluded[ic, jc]

    # Check if not already urbanized in original and delta grid
    z_mask = (z == 0)
    delta_mask = (delta == 0)

    # Apply slope restrictions
    # sweights give the probability of rejecting urbanization
    sweights = slope_weight(slope, coef_slope, crit_slope)
    slp_mask = (prng.random(size=len(sweights)) >= sweights)

    # Apply excluded restrictions, excluded values >= 100 are
    # completely unavailable, 0 are always available
    # excld value is the 100*probability of rejecting urbanization
    excld_mask = (prng.integers(100, size=len(excld)) >= excld)

    # Number of pixels not available for urbanization
    z_fails = (~z_mask).sum()
    delta_fails = (~delta_mask).sum()
    slope_fails = (~slp_mask).sum()
    excld_fails = (~excld_mask).sum()

    mask = reduce(np.logical_and,
                  [z_mask, delta_mask, slp_mask, excld_mask])

    fails = {'z': z_fails, 'delta': delta_fails,
             'slope': slope_fails, 'excluded': excld_fails}

    return mask, fails


def urbanizable_nghbrs(i, j,
                       grd_Z, grd_delta, grd_slope, grd_excluded,
                       coef_slope,
                       crit_slope,
                       prng):
    """Attempt to urbanize the neiggborhood of (i, j).

    Neighbors are chosen in random order until two successful
    urbanizations or all neighbors have been chosen.

    Parameters
    ----------
    i : int
        Row coordinate of center pixel.
    j : int
        Column coordinate of center pixel.
    grd_z: np.array
        2D binary array with original urbanization for the current step.
    grd_delta: np.array
        2D binary array where new urbanization is temporary created.
    grd_slope: np.array
        2D int array with slope values normalized to 1-100 range.
    grd_excluded: np.array
        2D int array with excluded zones. Values are 100*probability of
        rejecting urbanization. 0 is always available, over 100 means
        urbanization  is always rejected.
    coef_slope : float
        Slope coefficient controlling the probability of rejecting
        urbanization due to a steep slope.
    crit_slope : float
        The slope treshold above wich urbanization is always rejected.
    prng : TODO
        The random number generator class instance.

    Returns
    -------
    nlist: np.array
        Array of neighbor coordinates in random order.
    mask: np.array
       Boolean array for urbanizable neighbors, True if neighbor is
       urbanizable. Same shape as nlist.
    fails: dic
       Dictionary with number non urbanizable neighbors per grid test.
    """

    nlist = (i, j) + np.array(((-1, -1), (0, -1), (+1, -1), (+1, 0),
                               (+1, +1), (0, +1), (-1, +1), (-1, 0)))
    prng.shuffle(nlist)
    # Obtain urbanizable neighbors
    mask, fails = urbanizable(
        nlist, grd_Z, grd_delta, grd_slope, grd_excluded,
        coef_slope,
        crit_slope,
        prng)

    return nlist, mask, fails


def slope_weight(slope, slp_res, critical_slp):
    """Calculate slope resistance weights.

    Calculates the slope resistance for slope values. The slope
    resistance is a monotonous increasing function of slope boundend in 0-1
    which specific shape is controlled by slp_res coefficient and the
    slope critical value.

    It is interpreted as the probability of resisting urbanization due to
    a steep slope.

    The function is:

    min(1, 1 - (crit_slp - slp)/crit_slp ** 2(slp_res/MAX_SLP_RES))

    Note: The max exponent is 2, which bounds the growth rate of
    rejection probability. A more generall sigmoid function may be
    useful here. Another advantage of a more general sigmoid is that
    the effect of the critical cutoff can be handled by a shape parameter,
    which is differentiable.

    Parameters
    ----------
    slope : float or np.array
        Slope values for which to calculate resistance.
    slp_res : float
        Coefficient governing the shape of the resistance function.
    critical_slp : float
        Critical slope value above which all weights are one.

    Returns
    -------
    slope_w: np.array
        1D array of slope weights.

    """

    slope = np.asarray(slope)
    val = ((critical_slp - slope)/critical_slp)
    exp = (2*slp_res/C.MAX_SLOPE_RESISTANCE)
    slope_w = np.where(slope >= critical_slp, 0, val)
    slope_w = 1.0 - slope_w**exp

    return slope_w
