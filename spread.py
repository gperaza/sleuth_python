import numpy as np
# import constants as C
from functools import reduce
# from scipy.ndimage import convolve
# from timer import timers
# from logconf import logger
from stats import compute_stats
from stats import Record
from scipy.stats import pearsonr


def driver(
        total_mc,
        start_year,
        end_year,
        calibrating,
        grds_urban,
        urban_years,
        calibration_stats,
        grd_slope,
        grd_excluded,
        grd_roads,
        grd_roads_dist,
        grd_road_i,
        grd_road_j,
        coef_diffusion,
        coef_breed,
        coef_spread,
        coef_slope,
        coef_road,
        crit_slope,
        boom,
        bust,
        sens_slope,
        sens_road,
        critical_high,
        critical_low,
        prng,
        out_path,
        write_mc=False,
        write_records=False,
        write_iter=False
):
    """ Performs a single run consisting of several montecarlo interations.

    Returns
    -------
    optimal: float
        Optimal SLEUTH parameter.
    """

    # Create urbanization attempt data structure to log
    # failed urbanizations
    # urb_attempt = UrbAttempt()

    # Simulated years include end_year but no seed year (start_year)
    year_list = np.arange(start_year + 1, end_year + 1)
    nyears = end_year - start_year

    # Create records to accumulate statistics during growth
    records = [
        Record(
            year=year,
            diffusion=coef_diffusion,
            breed=coef_breed,
            spread=coef_spread,
            slope=coef_slope,
            road=coef_road,
            write_records=write_records,
            write_iter=write_iter,
            out_path=out_path
        )
        for year in year_list
    ]

    # Create monte carlo grid to accumulate probability of urbanization
    # one grid per simulated year
    nrows, ncols = grds_urban[0].shape
    grid_MC = np.zeros((nyears, nrows, ncols))

    # Driver MonteCarlo(grd_Z_cumulative)
    for current_mc in range(1, total_mc + 1):

        # Reset the coefficients to current run starting values
        # This is not need, as input values do no change

        # Perform simlation from start to end year
        (
            coef_diffusion_mod,
            coef_breed_mod,
            coef_spread_mod,
            coef_slope_mod,
            coef_road_mod
        ) = grow(
            grid_MC,
            grds_urban,
            urban_years,
            grd_slope,
            grd_excluded,
            grd_roads,
            grd_roads_dist,
            grd_road_i,
            grd_road_j,
            coef_diffusion,
            coef_breed,
            coef_spread,
            coef_slope,
            coef_road,
            crit_slope,
            boom,
            bust,
            sens_slope,
            sens_road,
            critical_high,
            critical_low,
            prng,
            records,
            # urb_attempt,
            # out_path,
            # write_iter,
        )

    # Normalize probability grid
    grid_MC /= total_mc

    # Output urban MC grids in native numpy format for postprocessing
    if write_mc:
        np.save(out_path / 'MC.npy', grid_MC)

    # Compute std of records and dump aggregated stats to disk
    for record in records:
        record.compute_std()
        record.write_fields()

    # Calculate calibration statistics
    # The modified coefficients are extracted
    # from the last oberseved year average record.
    osm = None
    if calibrating:
        # Create control file
        control_file = (
            out_path
            /
            'control_'
            f'_diffusion_{coef_diffusion}'
            f'_breed_{coef_breed}'
            f'_spread_{coef_spread}'
            f'_slope_{coef_slope}'
            f'_road_{coef_road}'
            '.csv'
        )
        with open(control_file, 'w') as f:
            # Write header
            f.write('Diffusion,Breed,Spread,Slope,Road,'
                    'Diffusion_mod,Breed_mod,Spread_mod,Slope_mod,Road_mod,'
                    'Diffusion_std,Breed_std,Spread_std,Slope_std,Road_std,'
                    'Compare,Leesalee,pop,edges,clusters,mean_clust_size,'
                    'avg_slope,percent_urban,xmean,ymean,rad,'
                    'product,osm\n')

            f.write(
                f'{coef_diffusion},{coef_breed},{coef_spread},'
                f'{coef_slope},{coef_road},'
            )
        osm = optimal_sleuth(
            records, calibration_stats, urban_years, control_file)

    return osm


def optimal_sleuth(records, calibration_stats, urban_years, control_file):

    # When calculating the fit statistics, we want to write these
    # statistics to disk toguether with the initial coefficients that
    # generated the simulations.
    # Since they are being written to disk, there is no need to store them

    # The optimal SLEUTH metric is the product of:
    # compare, pop, edges, clusters, slope, x_mean, and y_mean
    # Extract mean records for urban years
    sim_stats = [record.average for record in records
                 if record.year in urban_years]
    sim_stds = [record.std for record in records
                if record.year in urban_years]

    # Write modified coefficients from record corresponding to the
    # last urban year.
    diffusion_mod = sim_stats[-1].diffusion_mod
    spread_mod = sim_stats[-1].spread_mod
    breed_mod = sim_stats[-1].breed_mod
    slope_resistance_mod = sim_stats[-1].slope_resistance_mod
    road_gravity_mod = sim_stats[-1].road_gravity_mod
    diffusion_std = sim_stds[-1].diffusion_mod
    spread_std = sim_stds[-1].spread_mod
    breed_std = sim_stds[-1].breed_mod
    slope_resistance_std = sim_stds[-1].slope_resistance_mod
    road_gravity_std = sim_stds[-1].road_gravity_mod
    with open(control_file, 'a') as f:
        f.write(f'{diffusion_mod},{spread_mod},{breed_mod},'
                f'{slope_resistance_mod},{road_gravity_mod},'
                f'{diffusion_std},{spread_std},{breed_std},'
                f'{slope_resistance_std},{road_gravity_std},')

    # Compare: ratio of final urbanizations at last control years
    final_pop_sim = sim_stats[-1].pop
    final_pop_urb = calibration_stats[-1].pop
    compare = (min(final_pop_sim, final_pop_urb)
               / max(final_pop_sim, final_pop_urb))

    # leesalee: mean leesalee metric
    leesalee = np.mean([s.leesalee for s in sim_stats])

    # Find regression coefficients, ignore seed year
    metrics_names = ['pop', 'edges', 'clusters', 'mean_cluster_size',
                     'slope', 'percent_urban', 'xmean', 'ymean', 'rad']
    osm_metrics_names = ['pop', 'edges', 'clusters', 'slope', 'xmean', 'ymean']
    metrics = []
    osm_metrics = []
    for metric in metrics_names:
        # simulation skips seed year
        sim_vals = [getattr(s, metric) for s in sim_stats]
        urb_vals = [getattr(s, metric) for s in calibration_stats[1:]]
        r, _ = pearsonr(sim_vals, urb_vals)
        r = r**2
        metrics.append(r)
        if metric in osm_metrics_names:
            osm_metrics.append(r)

    # Product of all statistics
    product = np.prod(metrics) * compare * leesalee

    # Optimal metric
    osm = np.prod(osm_metrics) * compare

    # Append to control file
    with open(control_file, 'a') as f:
        f.write(f'{compare},{leesalee},')
        for m in metrics:
            f.write(f'{m},')
        f.write(f'{product},{osm}\n')

    # Reurnt the optimal sleuth metric for calibration purposes
    return osm


def grow(
        grid_MC,
        grds_urban,
        urban_years,
        grd_slope,
        grd_excluded,
        grd_roads,
        grd_roads_dist,
        grd_road_i,
        grd_road_j,
        coef_diffusion,
        coef_breed,
        coef_spread,
        coef_slope,
        coef_road,
        crit_slope,
        boom,
        bust,
        sens_slope,
        sens_road,
        critical_high,
        critical_low,
        prng,
        records,
        # urb_attempt,
        # out_path,
        # write_iter,
):
    """ Loop over simulated years for a single MC iteration inside a
    single run.

    This functions performs a complete growth cycle from start year to end
    year, starting from the urban configuration of the given seed and
    using the given growth coefficients.

    Urbanization is accumulated in grd_Z during the simulation.
    Each year urbanization is accumulated in its respective
    montecarlo accumulation grid, one gird per year.

    If performing calibration, grds_urban contains the ground truth
    of urbanization for years indicated in urban_years, so calibration
    statistics can be calculated when ground truth is available.

    Statistics related to growth and calibration are stored in a record
    data structure. There is a record for each year inside the records list.

    The coefficients may change during simulation due self modification rules
    applied at the end of each year.
    The function returns set of coefficients at the end of the simulation.

    Parameters
    ----------
    grd_MC : np.array
        Monte Carlo accumulation grids, stores urbanization counts for
        each year for each monte carlo simulation.
    grds_urban: np.array
        Contains true urbanization for observed years in a
        (years, nrows, ncols) array.
        The first element [0] is the taken as the seed.
    urban_years: List
        List of years with available true urbanization state
        stored in grds_urban, with the same order as that of the first
        dimension on grds_urban.
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
    boom : float
        Constant greater than one that multiplies coefficients after
        a boom year.
    bust : float
        Constant less than one that multiplies coefficients after a bust year.
    sens_slope : float
        Slope sensitivity.
    sens_road : float
        Road sensitivity.
    critical_high:
        Growth rate treshold beyon which a boom is triggered.
    critical_low:
        Growth rate treshold below which a bust is triggered.
    prng : np.random.Generator
        The random number generator.
    records: List
        List of record data structures to store statistics, one per
        simulated year.
    urb_attempt : UrbAttempt
        Data class instance to log urbanization attempt metrics.

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

    # timers.GROW.start()

    # The number of years to simulate is dictated by the shape of the MC grid
    nyears = grid_MC.shape[0]
    # Make sure we have a record per simulated year
    assert nyears == len(records)

    # Initialize Z grid to the seed (first urban grid)
    # TODO: Zero grid instead of creating new one.
    grd_Z = grds_urban[0].copy()

    # Precalculate/reset slope weighs
    # This can change due to self-modification during growth.
    sweights = 1 - slope_weight(grd_slope, coef_slope, crit_slope)

    # Normalize urbanization probabilities
    nrows, ncols = grd_Z.shape
    diag = np.sqrt(nrows**2 + ncols**2)
    p_diff = 0.005 * coef_diffusion * diag / (nrows * ncols)
    p_breed = coef_breed / 100.0

    for year in range(nyears):
        record = records[year]
        # Reset urbanization attempts data structure
        # urb_attempt.reset()

        # logger.info(f'  {record.year}|{year}/{nyears}')
        record.monte_carlo += 1

        # Apply CA rules for current year

        # timers.SPR_TOTAL.start()
        (sng, sdc, og, rt, num_growth_pix) = spread(
            grd_Z,
            grd_slope,
            grd_excluded,
            grd_roads,
            grd_roads_dist,
            grd_road_i,
            grd_road_j,
            coef_diffusion,
            coef_breed,
            coef_spread,
            coef_slope,
            coef_road,
            crit_slope,
            prng,
            # urb_attempt,
            sweights,
        )
        # timers.SPR_TOTAL.stop()

        # Send stats to current year (ints)
        record.this_year.sng = sng
        record.this_year.sdc = sdc
        record.this_year.og = og
        record.this_year.rt = rt
        record.this_year.num_growth_pix = num_growth_pix
        # record.this_year.successes = urb_attempt.successes
        # record.this_year.z_failure = urb_attempt.z_failure
        # record.this_year.delta_failure = urb_attempt.delta_failure
        # record.this_year.slope_failure = urb_attempt.slope_failure
        # record.this_year.excluded_failure = urb_attempt.excluded_failure

        # Store coefficients
        record.this_year.diffusion = coef_diffusion
        record.this_year.spread = coef_spread
        record.this_year.breed = coef_breed
        record.this_year.slope_resistance = coef_slope
        record.this_year.road_gravity = coef_road

        # Compute stats
        (
            record.this_year.edges,
            record.this_year.clusters,
            record.this_year.pop,
            record.this_year.xmean,
            record.this_year.ymean,
            record.this_year.slope,
            record.this_year.rad,
            record.this_year.mean_cluster_size,
            record.this_year.percent_urban
        ) = compute_stats(grd_Z, grd_slope)

        # Growth
        record.this_year.growth_rate = (
            100.0 * num_growth_pix / record.this_year.pop
        )

        # If there exists a corresponding urban grid,
        # calculate intersection over union for calibration
        if record.year in urban_years:
            grd_urban = grds_urban[urban_years.index(record.year)]
            record.this_year.leesalee = (
                np.logical_and(grd_Z, grd_urban).sum(dtype=float)
                / np.logical_or(grd_Z, grd_urban).sum(dtype=float)
            )
        else:
            record.this_year.leesalee = 0.0

        # Accumulate MC samples
        # TODO: avoid indexing making sure Z grid is at most 1.
        grid_MC[year][grd_Z > 0] += 1

        # Do self modification
        coef_slope_prev = coef_slope
        coef_diffusion_prev = coef_diffusion
        coef_breed_prev = coef_breed
        # (coef_diffusion, coef_breed,
        #  coef_spread, coef_slope, coef_road) = coef_self_modification(
        #      coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road,
        #      boom, bust, sens_slope, sens_road,
        #      record.this_year.growth_rate, record.this_year.percent_urban,
        #      critical_high, critical_low)
        if coef_slope_prev != coef_slope:
            # Recalculate slope weights
            sweights = 1 - slope_weight(grd_slope, coef_slope, crit_slope)
        if coef_diffusion_prev != coef_diffusion:
            p_diff = 0.005 * coef_diffusion * diag / (nrows * ncols)
        if coef_breed_prev != coef_breed:
            p_breed = coef_breed / 100.0

        # Store modified coefficients
        record.this_year.diffusion_mod = coef_diffusion
        record.this_year.spread_mod = coef_spread
        record.this_year.breed_mod = coef_breed
        record.this_year.slope_resistance_mod = coef_slope
        record.this_year.road_gravity_mod = coef_road

        # Update mean and sum of squares
        record.update_mean_std()

        # Write iteration to disk
        record.write_iteration()

    # timers.GROW.stop()

    return coef_diffusion, coef_breed, coef_spread, coef_slope, coef_road


def spread(
        grd_Z,
        grd_slope,
        grd_excluded,
        grd_roads,
        grd_roads_dist,
        grd_road_i,
        grd_road_j,
        coef_diffusion,
        coef_breed,
        coef_spread,
        coef_slope,
        coef_road,
        crit_slope,
        prng,
        # urb_attempt,
        sweights,
):
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
    # TODO: zero grid instead of creating
    grd_delta = np.zeros_like(grd_Z)

    # Slope coef and crit are constant during a single step
    # TODO:Precalculate all slope weights?

    # timers.SPR_PHASE1N3.start()
    sng, sdc = phase1n3_new(
        grd_Z,
        grd_delta,
        # grd_slope,
        # grd_excluded,
        coef_diffusion,
        coef_breed,
        # coef_slope,
        # crit_slope,
        prng,
        # urb_attempt,
        sweights,
    )
    # timers.SPR_PHASE1N3.stop()

    # timers.SPR_PHASE4.start()
    og = phase4_new(
        grd_Z,
        grd_delta,
        # grd_slope,
        # grd_excluded,
        coef_spread,
        # coef_slope,
        # crit_slope,
        prng,
        # urb_attempt,
        sweights,
    )
    # timers.SPR_PHASE4.stop()

    # timers.SPR_PHASE5.start()
    rt = phase5(
        grd_Z,
        grd_delta,
        grd_slope,
        grd_excluded,
        grd_roads,
        grd_roads_dist,
        grd_road_i,
        grd_road_j,
        coef_road,
        coef_diffusion,
        coef_breed,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
    )
    # timers.SPR_PHASE5.stop()

    # Performe excluded test, in this new implmentation
    # this test is only performed once per step
    # in the delta grid
    # excld value is the 100*probability of rejecting urbanization
    # TODO: implement as boolean operation without indexing
    excld_mask = prng.random(size=grd_delta.shape)*100 < grd_excluded
    grd_delta[excld_mask] = 0

    # Urbanize in Z array for accumulated urbanization.
    # TODO: Try to avoid indexing, implement as boolean operation.
    mask = grd_delta > 0
    grd_Z[mask] = grd_delta[mask]
    # avg_slope = grd_slope[mask].mean()
    num_growth_pix = mask.sum()
    # pop = (grd_Z >= C.PHASE0G).sum()

    return sng, sdc, og, rt, num_growth_pix


def phase_spontaneous(
        grd_delta,
        p_diff,
        prng,
        sweights
):
    # Adjust probability with slope rejection
    p_diff = sweights*p_diff

    # Apply urbanization test to all pixels
    mask = prng.random(size=grd_delta.shape) < p_diff
    sng = mask.sum()

    # Update delta grid
    np.logical_or(grd_delta, mask, out=grd_delta)

    return sng


def phase_breed(
        grd_delta,
        p_breed,
        prng,
        sweights,
):

    n_nbrs = count_neighbors(grd_delta)

    # Adjust probability with slope rejection
    p_breed = sweights*p_breed
    # The probability of urbanization is then (see notes)
    p_urb = (-1)**(n_nbrs + 1)*(p_breed / 4.0 - 1)**n_nbrs + 1

    # Apply random test
    mask = prng.random(size=grd_delta.shape) < p_urb
    sdc = mask.sum()

    # Update delta grid
    np.logical_or(grd_delta, mask, out=grd_delta)

    return sdc


def phase1n3_new(
        grd_Z,
        grd_delta,
        # grd_slope,
        # grd_excluded,
        coef_diffusion,
        coef_breed,
        # coef_slope,
        # crit_slope,
        prng,
        # urb_attempt,
        sweights,
):
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

    nrows, ncols = grd_Z.shape

    # Get urbanization probability per pixel
    diag = np.sqrt(nrows**2 + ncols**2)
    p_diff = 0.005 * coef_diffusion * diag / (nrows * ncols)
    # Adjust probability with slope rejection
    p_diff = sweights*p_diff

    # Apply urbanization test to all pixels
    # TODO: ignore borders
    mask = prng.random(size=grd_delta.shape) < p_diff
    sng = mask.sum()

    # Update delta grid
    np.logical_or(grd_delta, mask, out=grd_delta)

    # For PHASE 2
    # kernel = np.array([[1,  1, 1],
    #                    [1,  0, 1],
    #                    [1,  1, 1]])

    # Find number of neighbors in delta grid
    # n_nbrs = convolve(
    #     grd_delta,
    #     kernel,
    #     mode='constant'
    # )
    n_nbrs = count_neighbors(grd_delta)

    # Apply urbanization test
    # Pixels that past the test attempt urbanization
    p_breed = (coef_breed / 100.0)
    # Adjust probability with slope rejection
    p_breed = sweights*p_breed
    # The probability of urbanization is then (see notes)
    p_urb = (-1)**(n_nbrs + 1)*(p_breed / 4.0 - 1)**n_nbrs + 1

    # Apply random test
    mask = prng.random(size=grd_delta.shape) < p_urb
    sdc = mask.sum()

    # Update delta grid
    np.logical_or(grd_delta, mask, out=grd_delta)

    return sng, sdc


def phase4_new(
        grd_Z,
        grd_delta,
        # grd_slope,
        # grd_excluded,
        coef_spread,
        # coef_slope,
        # crit_slope,
        prng,
        # urb_attempt,
        sweights,
):
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
    # kernel = np.array([[1,  1, 1],
    #                    [1,  0, 1],
    #                    [1,  1, 1]])

    # Get array with number of neighbors for each pixel.
    # n_nbrs = convolve(
    #     grd_Z,
    #     kernel,
    #     mode='constant'
    # )
    n_nbrs = count_neighbors(grd_Z)

    # Spread centers are urban pixels with 2 or more neighbors
    cluster_labels = np.logical_and(n_nbrs >= 2, grd_Z)

    # Count cluster neighbors per pixel
    # n_nbrs = convolve(
    #     cluster_labels,
    #     kernel,
    #     mode='constant'
    # )
    n_nbrs = count_neighbors(cluster_labels)

    # Apply urbanization test
    # Pixels that past the test attempt urbanization
    p_sprd = coef_spread / 100.0
    # Adjust probability with slope rejection
    p_sprd = sweights*p_sprd
    # The probability of urbanization is then (see notes)
    p_urb = (-1)**(n_nbrs + 1)*(p_sprd / 8.0 - 1)**n_nbrs + 1

    # Apply random test
    mask = prng.random(size=grd_delta.shape) < p_urb

    # Update delta grid
    np.logical_or(grd_delta, mask, out=grd_delta)

    # Get urbanize pixels in this phase
    # Note: this double counts already urbanized pixels.
    og = mask.sum()

    return og


def phase5(
        grd_Z,
        grd_delta,
        grd_slope,
        grd_excluded,
        grd_roads,
        grd_road_dist,
        grd_road_i,
        grd_road_j,
        coef_road,
        coef_diffusion,
        coef_breed,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
):
    """ Road influenced growth.

    This function executes growth influenced by the presence of the
    road network. It looks for urban pixels near a road and attempts to
    urbanize pixels at the road near new urbanizations.
    For each succesful road urbanization, it then attempts to grow a new urban
    center.
    Takes the existing urbanization in the DELTA grid and writes new
    urbanization into the delta grid. The Z grid is still needed as to not
    overwrite previous urbanization unnecessarily.

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

    assert coef_road >= 0
    assert coef_diffusion >= 0
    assert coef_breed >= 0

    # calculate tha maximum distance to search for a road
    # this is the chebyshev distance and is precomputed in grid
    # maxed at ~ 1/32 image perimeter
    nrows, ncols = grd_delta.shape
    max_dist = int(coef_road / 100
                   * (nrows + ncols) / 16.0)
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
        coords,
        size=int(coef_breed) + 1,
        replace=True
    )

    # Search for nearest road, and select only if road is close enough
    dists = grd_road_dist[coords[:, 0], coords[:, 1]]
    coords = coords[dists < max_dist]
    road_i = grd_road_i[coords[:, 0], coords[:, 1]]
    road_j = grd_road_j[coords[:, 0], coords[:, 1]]
    rcoords = np.column_stack([road_i, road_j])

    # For selected roads perform a random walk and attempt urbanization
    # It is perhaps faster justo to choose a road pixel at random and
    # attempt urbanization close to it?
    nlist = np.array(
        (
            (-1, -1), (0, -1), (+1, -1), (+1, 0),
            (+1, +1), (0, +1), (-1, +1), (-1, 0)
        )
    )

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
        grd_Z,
        grd_delta,
        grd_slope,
        grd_excluded,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
    )
    new_sites = new_sites[mask]
    # Log successes
    rt = len(new_sites)
    # Update available pixels in delta grid
    grd_delta[new_sites[:, 0], new_sites[:, 1]] = 1

    # Attempt to create new urban centers, urbanize 2 neighbors
    for i, j in new_sites:
        # get urbanizable neighbors
        ncoords, mask = urbanizable_nghbrs(
            i,
            j,
            grd_Z,
            grd_delta,
            grd_slope,
            grd_excluded,
            coef_slope,
            crit_slope,
            prng,
            # urb_attempt
        )
        # choose two urbanizable neighbors
        ncoords = ncoords[mask][:2]
        rt += len(ncoords)
        # Update delta grid with values for phase 5
        grd_delta[ncoords[:, 0], ncoords[:, 1]] = 1

    return rt


def urbanizable(
        coords,
        grd_Z,
        grd_delta,
        grd_slope,
        grd_excluded,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
):
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
    # excld = grd_excluded[ic, jc]

    # Check if not already urbanized in original and delta grid
    z_mask = (z == 0)
    delta_mask = (delta == 0)
    # urb_attempt.z_failure += (~z_mask).sum()
    # urb_attempt.delta_failure += (~delta_mask).sum()

    # Apply slope restrictions
    # sweights give the probability of rejecting urbanization
    sweights = slope_weight(slope, coef_slope, crit_slope)
    slp_mask = (prng.random(size=len(sweights)) >= sweights)
    # urb_attempt.slope_failure += (~slp_mask).sum()

    # Apply excluded restrictions, excluded values >= 100 are
    # completely unavailable, 0 are always available
    # excld value is the 100*probability of rejecting urbanization
    # excld_mask = (prng.integers(100, size=len(excld)) >= excld)
    # urb_attempt.excluded_failure += (~excld_mask).sum()

    mask = reduce(
        np.logical_and,
        [
            z_mask,
            delta_mask,
            slp_mask,
            # excld_mask
        ]
    )
    # urb_attempt.successes += mask.sum()

    return mask


def urbanizable_nghbrs(
        i,
        j,
        grd_Z,
        grd_delta,
        grd_slope,
        grd_excluded,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
):
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

    nlist = (i, j) + np.array(
        ((-1, -1), (0, -1), (+1, -1), (+1, 0),
         (+1, +1), (0, +1), (-1, +1), (-1, 0))
    )
    # TODO: instead of shyffling in place, generate randon indices.
    prng.shuffle(nlist)
    # Obtain urbanizable neighbors
    mask = urbanizable(
        nlist,
        grd_Z,
        grd_delta,
        grd_slope,
        grd_excluded,
        coef_slope,
        crit_slope,
        prng,
        # urb_attempt
    )

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
    val = (critical_slp - slope) / critical_slp
    exp = 2 * slp_res / 100
    slope_w = np.where(slope >= critical_slp, 0, val)
    slope_w = 1.0 - slope_w**exp

    return slope_w


def coef_self_modification(
        coef_diffusion,
        coef_breed,
        coef_spread,
        coef_slope,
        coef_road,
        boom,
        bust,
        sens_slope,
        sens_road,
        growth_rate,
        percent_urban,
        critical_high,
        critical_low
):
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
    sens_road : float
        Road sensitivity.
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


def count_neighbors(Z):
    N = np.zeros(Z.shape, dtype=np.int8)
    N[1:-1, 1:-1] += (Z[:-2, :-2] + Z[:-2, 1:-1] + Z[:-2, 2:] +
                      Z[1:-1, :-2] + Z[1:-1, 2:] +
                      Z[2:, :-2] + Z[2:, 1:-1] + Z[2:, 2:])
    return N
