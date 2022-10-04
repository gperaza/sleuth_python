import numpy as np
import constants as C
from functools import reduce
from scipy.ndimage import convolve
from timer import timers
from logconf import logger
from stats import compute_stats
from stats import UrbAttempt, Record

# Must be log outside grow
# if proc_type == 'calibrating':
#     logger.info(f'Run = {current_run} of {total_runs}'
#                 f' ({100*current_run/total_runs:8.1f} percent complete)')
# logger.info(f'Monter Carlo = {current_mc} of {total_mc}')
# logger.info(f'Current year = {current_year}')
# logger.info(f'Stop year = {stop_year}')


def driver(run, total_mc, start_year, end_year,
           urban_seed, true_urban, true_years,
           grd_slope, grd_excluded,
           grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
           coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
           crit_slope,
           boom, bust, sens_slope, sens_road, critical_high, critical_low,
           prng):
    """ Performs a single run consisting of several montecarlo interations. """

    # Create urbanization attempt data structure to log
    # failed urbanizations
    urb_attempt = UrbAttempt()

    # Simulated years include end_year but no seed year (start_year)
    year_list = np.arange(start_year + 1, end_year + 1)
    nyears = end_year - start_year

    # Create records to accumulate statistics during growth
    records = [Record(run=run, year=year, has_true=(year in true_years))
               for year in year_list]

    # Create monte carlo grid to acculate probability of urbanization
    # one grid per simulated year
    nrows, ncols = urban_seed.shape
    grid_MC = np.zeros((nyears, nrows, ncols))

    # Driver MonteCarlo(grd_Z_cumulative)
    for current_mc in range(1, total_mc + 1):
        # Reset the coefficients to current run starting values

        # Reset urbanization attempts data structure
        urb_attempt.reset()

        # Perform simlation from start to end year
        grow(urban_seed, grid_MC,
             true_urban, true_years,
             grd_slope, grd_excluded,
             grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
             coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
             crit_slope,
             boom, bust, sens_slope, sens_road, critical_high, critical_low,
             prng,
             records, urb_attempt)

        # Ouput urb attempts

    # Normalize probability grid
    grid_MC /= total_mc

    # Output urban MC grids, how to ouput? numpy binary? only if predicting?

    # stats analysis function

    # Output records


def grow(urban_seed, grid_MC,
         true_urban, true_idxs,
         grd_slope, grd_excluded,
         grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
         coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
         crit_slope,
         boom, bust, sens_slope, sens_road, critical_high, critical_low,
         prng,
         records, urb_attempt):
    """ Loop over simulated years for a single MC iteration inside a
    single run.

    This functions performs a complete growth cycle from start year to end
    year, starting from the urban configuration of the given seed and
    using the given growth coefficients.

    Urbanization is accumulated in grd_Z during the simulation.
    Each year urbanization is accumulated in its respective
    montecarlo accumulation grid, one gird per year.

    If performing calibration, true_urban contains the ground truth
    of urbanization for years indicated in true_idxs, so calibration statistics
    can be calculated when ground truth is available.

    Statistics related to growth and calibration are stored in a record
    data structure. There is a record for each year inside the records list.

    The coefficients may change during simulation due self modification rules
    applied at the end of each year.
    The function returns set of coefficients at the end of the simulation.


    """

    # timers.GROW.start()

    # The number of years to simulate is dictated by the shape of the MC grid
    nyears = grid_MC.shape[0]
    assert nyears == len(records)

    # Initialize Z grid to the seed
    grd_Z = urban_seed.copy()

    for year in range(nyears):
        record = records[year]

        logger.info(f'  {year}|{record.year}/{nyears}')
        # record.year = year
        record.monte_carlo += 1

        # Apply CA rules for current year

        # timers.SPR_TOTAL.start()
        (sng, sdc, og, rt, num_growth_pix) = spread(
            grd_Z, grd_slope, grd_excluded,
            grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
            coef_diffusion, coef_breed, coef_spread,
            coef_slope, coef_road, crit_slope, prng,
            urb_attempt)
        # timers.SPR_TOTAL.stop()

        # Send stats to current year (ints)
        record.this_year.sng = sng
        record.this_year.sdc = sdc
        record.this_year.og = og
        record.this_year.rt = rt
        record.this_year.num_growth_pix = num_growth_pix

        # Store coefficients
        record.this_year.diffusion = coef_diffusion
        record.this_year.spread = coef_spread
        record.this_year.breed = coef_breed
        record.this_year.slope_resistance = coef_slope
        record.this_year.road_gravity = coef_road

        # Compute stats
        (record.this_year.area,
         record.this_year.edges,
         record.this_year.clusters,
         record.this_year.pop,
         record.this_year.xmean,
         record.this_year.ymean,
         record.this_year.slope,
         record.this_year.rad,
         record.this_year.mean_cluster_size,
         record.this_year.percent_urban) = compute_stats(grd_Z, grd_slope)

        # Growth
        record.this_year.growth_rate = (100.0
                                        * num_growth_pix/record.this_year.pop)

        # If there exists a corresponding urban grid,
        # calculate intersection over union for calibration
        if year in true_idxs:
            turban = true_urban[true_idxs.index(year)]
            record.this_year.leesalee = (
                np.loagical_and(grd_Z, turban).sum(dtype=float)
                / np.logical_or(grd_Z, turban).sum(dtype=float)
            )
        else:
            record.this_year.leesalee = 0.0

        # Update mean and sum of squares
        record.update_mean_std()

        # Accumulate MC samples
        grid_MC[grd_Z > 0] += 1

        # Do self modification
        (coef_diffusion, coef_breed,
         coef_spread, coef_slope, coef_road) = coef_self_modification(
             coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
             boom, bust, sens_slope, sens_road,
             record.this_year.growth_rate, record.this_year.percent_urban,
             critical_high, critical_low)

        # Store modified coefficients
        record.this_year.diffusion_mod = coef_diffusion
        record.this_year.spread_mod = coef_spread
        record.this_year.breed_mod = coef_breed
        record.this_year.slope_resistance_mod = coef_slope
        record.this_year.road_gravity_mod = coef_road

    # timers.GROW.stop()

    return coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road


def spread(grd_Z, grd_slope, grd_excluded,
           grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
           coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
           crit_slope,
           prng,
           urb_attempt):
    """ Simulate a year's urban growth.

    This function executes urban growth phases for a single step (year).
    Takes urbanization in previous years from the Z grid and temporary
    stores new urbanization in delta grid. Finally updates Z grid at the
    end the the growth phase.

    Parameters
    ----------
    grd_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grd_slope : np.array
        Array with slope values.
    grd_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    grd_roads : np.array
        Array with roads labeled by their importance normalized between ??-??.
    grd_road_dist : np.array
        Array with each pixels distance to the closest road.
    grd_road_i : np.array
        Array with the i coordinate of the closest road.
    grd_road_j : np.array
        Array with the j coordinate of the closest road.
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_spread : float
        Spread coefficient.
    coef_slope : float
        Slope coefficient.
    coef_road : float
        Road coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    sng: int
        Number of newly urbanized pixels during spontaneous growth.
    sdc: int
        Number of newly urbanized pixels corresponding to new urban centers.
    og: int
        Number of newly urbanized pixels during edge growth
    rt: int
        Number of newly urbanized pixels during road influenced growth.
    num_growth_pix: int
        Total number of new urban pixels.

    """

    # Initialize delta grid to store temporal urbanization.
    grd_delta = np.zeros_like(grd_Z)

    # Slope coef and crit are constant during a single step
    # TODO:Precalculate all slope weights?

    # timers.SPR_PHASE1N3.start()
    sng, sdc = phase1n3(grd_Z, grd_delta, grd_slope, grd_excluded,
                        coef_diffusion, coef_breed, coef_slope,
                        crit_slope,
                        prng, urb_attempt)
    # timers.SPR_PHASE1N3.stop()

    # timers.SPR_PHASE4.start()
    og = phase4(grd_Z, grd_delta, grd_slope, grd_excluded,
                coef_spread, coef_slope,
                crit_slope,
                prng, urb_attempt)
    # timers.SPR_PHASE4.stop()

    # timers.SPR_PHASE5.start()
    rt = phase5(grd_Z, grd_delta, grd_slope, grd_excluded,
                grd_roads, grd_roads_dist, grd_road_i, grd_road_j,
                coef_road, coef_diffusion, coef_breed, coef_slope,
                crit_slope,
                prng, urb_attempt)
    # timers.SPR_PHASE5.stop()

    # Urbanize in Z array for accumulated urbanization.
    mask = grd_delta > 0
    grd_Z[mask] = grd_delta[mask]
    # avg_slope = grd_slope[mask].mean()
    num_growth_pix = mask.sum()
    # pop = (grd_Z >= C.PHASE0G).sum()

    return sng, sdc, og, rt, num_growth_pix


def phase1n3(grd_Z, grd_delta, grd_slope, grd_excluded,
             coef_diffusion, coef_breed, coef_slope,
             crit_slope,
             prng,
             urb_attempt):
    """ Spontaneus growth and possible new urban centers.

    This function implements the first two stages of growth, the appearance
    of new urban cells at random anywhere on the grid, and the possible
    urbanization of two of their neighbors to create new urban centers.
    Takes the existing urbanizatiojn in the Z grid and writes new urbanization
    into the delta grid.

    Parameters
    ----------
    grd_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grd_delta : np.array
        Urban delta grid, stores new urbanization for the simulated year.
    grd_slope : np.array
        Array with slope values.
    grd_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_slope : float
        Slope coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    sng: int
        Number of newly urbanized pixels during spontaneous growth.
    sdc: int
        Number of newly urbanized pixels corresponding to new urban centers.

    """

    assert C.MIN_DIFFUSION <= coef_diffusion <= C.MAX_DIFFUSION
    assert C.MIN_BREED <= coef_breed <= C.MAX_BREED

    nrows, ncols = grd_slope.shape

    # Diffusion value is at most 5% of image diagonal
    # Diffusion values is the number of candidate pixels to urbanize
    diffusion_val = int((coef_diffusion * 0.005)
                        * np.sqrt(nrows**2 + ncols**2))

    # Get candidate pixel coordinates, ignore borders
    # In orginal SLEUTH borders are ignore likely to avoid
    # treating neighbor search as a special case
    coords = prng.integers((1, 1), (nrows-1, ncols-1),
                           size=(diffusion_val, 2))

    # Check which ones are available for urbanization,
    # several checks must pass
    mask = urbanizable(
        coords,
        grd_Z, grd_delta, grd_slope, grd_excluded,
        coef_slope,
        crit_slope,
        prng, urb_attempt)
    coords = coords[mask]
    # Log number of urbanized pixels in phase 2 (sng)
    sng = len(coords)

    # Spontaneous growth
    # Update available pixels in delta grid
    grd_delta[coords[:, 0], coords[:, 1]] = C.PHASE1G

    # Grow new urban centers wit probability given by breed
    # Filter candidate new centers
    coords = coords[prng.integers(101, size=len(coords))
                    < coef_breed]
    sdc = 0
    for i, j in coords:
        # Original SLEUTH code samples up to 8 neighbors with
        # replacement, meaning a neighbor can be chosen more than
        # once and urbanization may fail even if available neighbors
        # exist. Here we try all neighbors and choose 2 urbanizable
        # ones.

        # get urbanizable neighbors
        ncoords, mask = urbanizable_nghbrs(
            i, j,
            grd_Z, grd_delta, grd_slope, grd_excluded,
            coef_slope,
            crit_slope,
            prng, urb_attempt)
        # choose two urbanizable neighbors
        ncoords = ncoords[mask][:2]
        # Log number of urbanizable pixels in phase 3 (sdc)
        sdc += len(ncoords)
        # Update delta grid with values for phase 3
        grd_delta[ncoords[:, 0], ncoords[:, 1]] = C.PHASE3G

    return sng, sdc


def phase4(grd_Z, grd_delta, grd_slope, grd_excluded,
           coef_spread, coef_slope,
           crit_slope,
           prng,
           urb_attempt):
    """ Edge growth of existing urban clusters composed of 3 or more pixels.

    This function executes edge growth. Each empty pixel with 3 or more
    urbanized neighbors has a probability of urbanize.
    Takes the existing urbanizatiojn in the Z grid and writes new urbanization
    into the delta grid.

    Parameters
    ----------
    grd_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grd_delta : np.array
        Urban delta grid, stores new urbanization for the simulated year.
    grd_slope : np.array
        Array with slope values.
    grd_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    coef_spread : float
        Spread coefficient.
    coef_slope : float
        Slope coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    og: int
        Number of newly urbanized pixels.

    """

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
    kernel = np.array([[1,  1, 1],
                       [1, -8, 1],
                       [1,  1, 1]])

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
    # I find the new behaviour more intiutive.
    # In the old way, if an edge pixel has many neighbors, then
    # urbanizing the available empty pixels has low probability.
    # When in fact we might expect that a larger urban footprint
    # leads to more urbanization pressure.
    # The new way at least keeps the probability constant.
    og = 0
    for i, j in zip(sprd_ci, sprd_cj):
        ncoords, mask = urbanizable_nghbrs(
            i, j,
            grd_Z, grd_delta, grd_slope, grd_excluded,
            coef_slope,
            crit_slope,
            prng, urb_attempt)
        if mask[0]:
            # Urbanize in delta grid
            grd_delta[ncoords[0, 0], ncoords[0, 1]] = C.PHASE4G
            # Log number of urbanizable pixels in phase 4 (og)
            og += 1

    return og


def phase5(grd_Z, grd_delta, grd_slope, grd_excluded,
           grd_roads, grd_road_dist, grd_road_i, grd_road_j,
           coef_road, coef_diffusion, coef_breed, coef_slope,
           crit_slope,
           prng,
           urb_attempt):
    """ Road influenced growth.

    This function executes growth influenced by the presence of the
    road network. It looks for urban pixels near a road and attempts to
    urbanize pixels at the road near new urbanizations.
    For each succesful road urbanization, it then attempts to grow a new urban
    center.
    Takes the existing urbanization in the DELTA grid and writes new urbanization
    into the delta grid. The Z grid is still needed as to not overwrite previous
    urbanization unnecessarily.

    Parameters
    ----------
    grd_Z : np.array
        Urban Z grid, stores urbanization from previous year.
    grd_delta : np.array
        Urban delta grid, stores new urbanization for the simulated year.
    grd_slope : np.array
        Array with slope values.
    grd_excluded : np.array
        Array wich labels excluded regions where urbanization is not allowed.
    grd_roads : np.array
        Array with roads labeled by their importance normalized between ??-??.
    grd_road_dist : np.array
        Array with each pixels distance to the closest road.
    grd_road_i : np.array
        Array with the i coordinate of the closest road.
    grd_road_j : np.array
        Array with the j coordinate of the closest road.
    coef_road : float
        Road coefficient.
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_slope : float
        Slope coefficient.
    crit_slope : float
        Critical slope value above which urbanization is rejected.
    prng : np.random.Generator
        The random number generator.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------
    rt: int
        Number of newly urbanized pixels.

    """

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
    # phase are considered. Why?
    # Woudn't it be more apprpriate to keep the infuence of roads constant
    # and influencing non urbanized pixels?
    # The method considers that only new urbanization is influenced by roads.
    # Most of this new urbanization comes from the spontaneous phase,
    # which means we are implicitly chosing pixels at random.

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
    # It is perhaps faster justo to choose a road pixel at random and
    # attempt urbanization close to it?
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
    mask = urbanizable(
        new_sites,
        grd_Z, grd_delta, grd_slope, grd_excluded,
        coef_slope,
        crit_slope,
        prng, urb_attempt)
    new_sites = new_sites[mask]
    # Log successes
    rt = len(new_sites)
    # Update available pixels in delta grid
    grd_delta[new_sites[:, 0], new_sites[:, 1]] = C.PHASE5G

    # Attempt to create new urban centers, urbanize 2 neighbors
    for i, j in new_sites:
        # get urbanizable neighbors
        ncoords, mask = urbanizable_nghbrs(
            i, j,
            grd_Z, grd_delta, grd_slope, grd_excluded,
            coef_slope,
            crit_slope,
            prng, urb_attempt)
        # choose two urbanizable neighbors
        ncoords = ncoords[mask][:2]
        rt += len(ncoords)
        # Update delta grid with values for phase 5
        grd_delta[ncoords[:, 0], ncoords[:, 1]] = C.PHASE5G

    return rt


def urbanizable(coords,
                grd_Z, grd_delta, grd_slope, grd_excluded,
                coef_slope,
                crit_slope,
                prng,
                urb_attempt):
    """ Determine wether pixels are subject to urbanization.

    Pixels subject to urbanization are not already urbanized pixels that
    pass the tests of slope and the exclution region.

    Parameters
    ----------
    coords : np.array
        Numpy array of pair of (i, j) coordinates of candidate pixels.
    grd_Z: np.array
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
    prng : np.random.Generator
        The random number generator class instance.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

    Returns
    -------

    mask: np.array
        1D boolean mask for array of candidate coordinates, True if available
        for urbanization.
    """

    # Extract vectors of grid values for candidate pixels.
    ic, jc = coords[:, 0], coords[:, 1]
    z = grd_Z[ic, jc]
    delta = grd_delta[ic, jc]
    slope = grd_slope[ic, jc]
    excld = grd_excluded[ic, jc]

    # Check if not already urbanized in original and delta grid
    z_mask = (z == 0)
    delta_mask = (delta == 0)
    urb_attempt.z_failure += (~z_mask).sum()
    urb_attempt.delta_failure += (~delta_mask).sum()

    # Apply slope restrictions
    # sweights give the probability of rejecting urbanization
    sweights = slope_weight(slope, coef_slope, crit_slope)
    slp_mask = (prng.random(size=len(sweights)) >= sweights)
    urb_attempt.slope_failure += (~slp_mask).sum()

    # Apply excluded restrictions, excluded values >= 100 are
    # completely unavailable, 0 are always available
    # excld value is the 100*probability of rejecting urbanization
    excld_mask = (prng.integers(100, size=len(excld)) >= excld)
    urb_attempt.excluded_failure += (~excld_mask).sum()

    mask = reduce(np.logical_and,
                  [z_mask, delta_mask, slp_mask, excld_mask])
    urb_attempt.successes += mask.sum()

    return mask


def urbanizable_nghbrs(i, j,
                       grd_Z, grd_delta, grd_slope, grd_excluded,
                       coef_slope,
                       crit_slope,
                       prng,
                       urb_attempt):
    """Attempt to urbanize the neiggborhood of (i, j).

    Neighbors are chosen in random order until two successful
    urbanizations or all neighbors have been chosen.

    Parameters
    ----------
    i : int
        Row coordinate of center pixel.
    j : int
        Column coordinate of center pixel.
    grd_Z: np.array
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
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.


    Returns
    -------
    nlist: np.array
        Array of neighbor coordinates in random order.
    mask: np.array
       Boolean array for urbanizable neighbors, True if neighbor is
       urbanizable. Same shape as nlist.
    """

    nlist = (i, j) + np.array(((-1, -1), (0, -1), (+1, -1), (+1, 0),
                               (+1, +1), (0, +1), (-1, +1), (-1, 0)))
    prng.shuffle(nlist)
    # Obtain urbanizable neighbors
    mask = urbanizable(
        nlist, grd_Z, grd_delta, grd_slope, grd_excluded,
        coef_slope,
        crit_slope,
        prng, urb_attempt)

    return nlist, mask


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


def coef_self_modification(
        coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
        boom, bust, sens_slope, sens_road,
        growth_rate, percent_urban,
        critical_high, critical_low):
    """ Applies self modification rules to growth coefficients.

    When growth is extreme during a given year, coefficients are modified to
    further mantain the growth trend.
    High growth results in a boom year, and coefficients are adjusted to
    further encourage growth.
    Low growth results in a bust year, and coefficients are adjusted to further
    restrict growth.

    Parameters
    ----------
    coef_diffusion : float
        Diffusion coefficient.
    coef_breed : float
        Breed coefficient.
    coef_spread : float
        Spread coefficient.
    coef_slope : float
        Slope coefficient.
    coef_road : float
        Road coefficient.
    boom : float
        Constant greater than one that multiplies coefficients after
        a boom year.
    bust : float
        Constant less than one that multiplies coefficients after a bust year.
    sens_slope : float
        Slope sensitivity.
    sens_spread : float
        Spread sensitivity.
    growth_rate: float
        Percent of newly urbanized pixels with respect to total urbanization.
    percent_urban: float
        Percentage of urban pixels with respect to total pixels.
    critical_high:
        Growth rate treshold beyon which a boom is triggered.
    critical_low:
        Growth rate treshold below which a bust is triggered.

    Returns
    -------
    coef_diffusion : float
        Modified diffusion coefficient.
    coef_breed : float
        Modified breed coefficient.
    coef_spread : float
        Modified spread coefficient.
    coef_slope : float
        Modified slope coefficient.
    coef_road : float
        Modified road coefficient.

    """

    # The self modification rules described in the official
    # documentation are not the ones implemented on the
    # official SLEUTH code.
    # Here we follow the code implementation.

    # BOOM year
    if growth_rate > critical_high:
        coef_slope -= percent_urban * sens_slope
        coef_slope = max(coef_slope, 1.0)

        coef_road += percent_urban * sens_road
        coef_road = min(coef_road, 100.0)

        if coef_diffusion < 100.0:
            coef_diffusion *= boom
            coef_diffusion = min(coef_diffusion, 100.0)

            coef_breed *= boom
            coef_breed = min(coef_breed, 100.0)

            coef_spread *= boom
            coef_spread = min(coef_spread, 100.0)

    # BOOST year
    if growth_rate < critical_low:
        coef_slope += percent_urban * sens_slope
        coef_slope = min(coef_slope, 100.0)

        coef_road -= percent_urban * sens_road
        coef_road = max(coef_road, 1.0)

        if coef_diffusion > 1:
            coef_diffusion *= bust
            coef_diffusion = max(coef_diffusion, 1.0)

            coef_breed *= bust
            coef_breed = max(coef_breed, 1.0)

            coef_spread *= bust
            coef_spread = max(coef_spread, 1.0)

    return coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road
