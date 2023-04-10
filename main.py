import argparse
import configparser
# from timer import timers
from pathlib import Path
import numpy as np
# from grid import igrid_init
import xarray as xr
from logconf import logger
from functools import partial
import subprocess
import stats
import spread
# from ax.service.managed_loop import optimize
from itertools import product
from ast import literal_eval


def parse_cl():
    parser = argparse.ArgumentParser(
        description='Simulate urban growth with the SLEUTH model.')
    # parser.add_argument('mode',
    #                     choices=['calibrate', 'predict'],
    #                     help="""Mode to run the simulator on.\n
    #   CALIBRATE will perform monte carlo runs through the
    #         historical data using every combination of the coefficient
    #         values indicated for all possible permutations of given
    #         ranges and increments.\n
    #   PREDICTION will perform a single run, in monte carlo
    #         fashion, using the BEST_FIT values for initialization.""")
    parser.add_argument('scenario_file',
                        help='Path to file with configuration.')

    args = parser.parse_args()
    ini_path = Path(args.scenario_file)
    # mode = args.mode

    return ini_path  # , mode


def main():
    # timers.TOTAL_TIME.start()

    githash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    logger.info(f'Running pySLEUTH with git hash: {githash}')

    # Initialize parameters
    # ini_path, mode = parse_cl()
    ini_path = parse_cl()

    logger.info(f'Loading configuration file: {ini_path}')
    config = configparser.ConfigParser()
    config.read(ini_path)

    mode = config['DEFAULT']['MODE']
    logger.info(f'Excecuting in mode: {mode}')

    input_dir = Path(config['DEFAULT']['INPUT_DIR'])
    logger.info(f'Input directory: {input_dir}')

    input_f_name = config['DEFAULT']['INPUT_FILE']
    logger.info(f'Input file: {input_f_name}')

    output_dir = Path(config['DEFAULT']['OUTPUT_DIR'])
    logger.info(f'Output directory: {output_dir}')

    random_seed = config['DEFAULT'].getint('RANDOM_SEED')
    logger.info(f'Random seed: {random_seed}')

    mc_iters = config['DEFAULT'].getint('MONTE_CARLO_ITERS')
    logger.info(f'Number of monte carlo iterations: {mc_iters}')

    coef_diffusion = config['COEFFICIENTS'].getint('DIFFUSION')
    logger.info(f'Diffusions coefficient: {coef_diffusion}')

    coef_breed = config['COEFFICIENTS'].getint('BREED')
    logger.info(f'Breed coefficient: {coef_breed}')

    coef_spread = config['COEFFICIENTS'].getint('SPREAD')
    logger.info(f'Spread coefficient: {coef_spread}')

    coef_slope = config['COEFFICIENTS'].getint('SLOPE')
    logger.info(f'Slope coefficient: {coef_slope}')

    critical_slope = config['COEFFICIENTS'].getfloat('CRITICAL_SLOPE')
    logger.info(f'Critical slope value: {critical_slope}')

    coef_road = config['COEFFICIENTS'].getint('ROAD')
    logger.info(f'Road coefficient: {coef_road}')

    start_year = config['DEFAULT'].getint('START_YEAR')
    logger.info(f'Being of simulation year: {start_year}')

    stop_year = config['DEFAULT'].getint('STOP_YEAR')
    logger.info(f'End of simulation year: {stop_year}')

    do_self_mod = config['SELF MODIFICATION'].getboolean('SELF_MOD')
    logger.info(f'Enable self modification: {do_self_mod}')

    critical_low = config['SELF MODIFICATION'].getfloat('CRITICAL_LOW')
    logger.info(f'Critical low self modification treshold: {critical_low}')

    critical_high = config['SELF MODIFICATION'].getfloat('CRITICAL_HIGH')
    logger.info(f'Critical high self modification treshold: {critical_high}')

    boom = config['SELF MODIFICATION'].getfloat('BOOM')
    logger.info(f'Boom self modification multiplier: {boom}')

    bust = config['SELF MODIFICATION'].getfloat('BUST')
    logger.info(f'Bust self modification multiplier: {bust}')

    road_sensitivity = config['SELF MODIFICATION'].getfloat('ROAD_SENS')
    logger.info(f'Road sensitivity (self modification): {road_sensitivity}')

    slope_sensitivity = config['SELF MODIFICATION'].getfloat('SLOPE_SENS')
    logger.info(f'Slope sensitivity (self modification): {slope_sensitivity}')

    optimizer = config['DEFAULT']['OPTIMIZER']
    logger.info(f'Optimizer: {optimizer}')

    diffusion_grid = literal_eval(config['GRID SEARCH']['DIFFUSION'])
    breed_grid = literal_eval(config['GRID SEARCH']['BREED'])
    spread_grid = literal_eval(config['GRID SEARCH']['SPREAD'])
    slope_grid = literal_eval(config['GRID SEARCH']['SLOPE'])
    road_grid = literal_eval(config['GRID SEARCH']['ROAD'])

    # Initialize grid
    # igrid = igrid_init(input_dir)
    igrid = xr.open_dataset(
        input_dir / input_f_name,
        cache=False,
        mask_and_scale=False)
    igrid.load()
    igrid.close()
    logger.info(print(str(igrid)))
    logger.info(f'Grid is using arround {int(igrid.nbytes/(1024**2))} MB')

    # Initiate PRNG
    # prng = np.random.default_rng(seed=random_seed)
    prng = np.random.Generator(np.random.SFC64(random_seed))

    # Compute base statistics against which calibration will take place
    base_stats, urban_years = stats.compute_base_stats(
        igrid, output_dir, start_year)
    logger.info(f'Taking historic growth from years: {urban_years}.')

    if mode == 'calibrate':
        # start_year = urban_years[0]
        stop_year = urban_years[-1]
        write_mc = False
        calibrating = True
        write_records = False
        write_iter = False
    elif mode == 'derive_coefs':
        # Average forecasting coefficients at the end of
        # historic growth.
        # Since coefficients may change due to self modification,
        # the best set of forecasting coefficients is the average
        # of 100 iterations at the end of the observed period.
        stop_year = urban_years[-1]
        write_mc = False
        calibrating = True
        write_records = False
        write_iter = False
    elif mode == 'predict':
        start_year = urban_years[-1]
        write_mc = True
        calibrating = False
        write_records = True
        write_iter = False
    else:
        raise ValueError("Mode not supported.")

    # Filter grid to years to simulate
    urban_years = [y for y in urban_years if y >= start_year]
    igrid = igrid.sel(year=urban_years)

    # Create a partial function that only accepts coefficient values
    driver = partial(
        spread.driver,
        total_mc=mc_iters,
        start_year=start_year,
        end_year=stop_year,
        calibrating=calibrating,
        grds_urban=igrid['urban'].values,
        urban_years=urban_years,
        calibration_stats=base_stats,
        grd_slope=igrid['slope'].values,
        grd_excluded=igrid['excluded'].values,
        grd_roads=igrid['roads'].values,
        grd_roads_dist=igrid['dist'].values,
        grd_road_i=igrid['road_i'].values,
        grd_road_j=igrid['road_j'].values,
        crit_slope=critical_slope,
        boom=boom,
        bust=bust,
        sens_slope=slope_sensitivity,
        sens_road=road_sensitivity,
        critical_high=critical_high,
        critical_low=critical_low,
        prng=prng,
        out_path=output_dir,
        write_mc=write_mc,
        write_records=write_records,
        write_iter=write_iter,
    )

    if calibrating:
        # if optimizer == 'ax':
        #     best_trial = optimize_ax(driver)
        if optimizer == 'grid_search':
            best_trial = grid_search(
                driver,
                diffusion_grid,
                breed_grid,
                spread_grid,
                slope_grid,
                road_grid,
            )
        elif optimizer == 'auto_grid':
            best_trial = optimize_gridded(
                driver
            )
            best_diff = best_trial[0]
            best_breed = best_trial[1]
            best_sprd = best_trial[2]
            best_slp = best_trial[3]
            best_rd = best_trial[4]

            # TODO: Run high mc historic simulation to get
            # averaged prediction coefs. Only useful is self_mod is activated.

            # TODO: Write predict scenario file
            config['DEFAULT']['MODE'] = 'predict'
            config['COEFFICIENTS']['DIFFUSION'] = str(best_diff)
            config['COEFFICIENTS']['BREED'] = str(best_breed)
            config['COEFFICIENTS']['SPREAD'] = str(best_sprd)
            config['COEFFICIENTS']['SLOPE'] = str(best_slp)
            config['COEFFICIENTS']['ROAD'] = str(best_rd)
            with open(
                    ini_path.with_name('scenario_predict.ini'),
                    'w'
            ) as configfile:
                config.write(configfile)

        else:
            raise NotImplementedError
    else:
        pass

    # timers.TOTAL_TIME.stop()


def grid_search(
        eval_f,
        diffusion_grid,
        breed_grid,
        spread_grid,
        slope_grid,
        road_grid,
):

    def tup_2_grid(grid):
        if isinstance(grid, tuple):
            start = grid[0]
            stop = grid[1]
            step = grid[2]
            grid = list(range(start, stop, step))
        assert isinstance(grid, list)
        return grid

    diffusion_grid = tup_2_grid(diffusion_grid)
    breed_grid = tup_2_grid(breed_grid)
    spread_grid = tup_2_grid(spread_grid)
    slope_grid = tup_2_grid(slope_grid)
    road_grid = tup_2_grid(road_grid)

    best_osm = 0
    best_diffusion = 0
    best_breed = 0
    best_spread = 0
    best_slope = 0
    best_road = 0

    total_iters = (
        len(diffusion_grid)
        * len(breed_grid)
        * len(spread_grid)
        * len(slope_grid)
        * len(road_grid)
    )

    combs = product(
        diffusion_grid,
        breed_grid,
        spread_grid,
        slope_grid,
        road_grid
    )

    for i, (diff, breed, sprd, slp, rd) in enumerate(combs):
        print(f'Executing iteration {i+1}/{total_iters}')
        osm = eval_f(
            coef_diffusion=diff,
            coef_breed=breed,
            coef_spread=sprd,
            coef_slope=slp,
            coef_road=rd
        )
        if osm > best_osm:
            best_osm = osm
            best_diffusion = diff
            best_breed = breed
            best_spread = sprd
            best_slope = slp
            best_road = rd
            print(f'New best: osm={best_osm}')
    print(
        f'{best_diffusion =}',
        f'{best_breed =}',
        f'{best_spread =}',
        f'{best_slope =}',
        f'{best_road =}',
    )


def optimize_gridded(eval_f):
    # Hierarchichal gridded calibration
    # This is the original caribration method of SLEUTH
    # Need to record the best 3 values of the OSM with
    # possible ties during the calibration process

    # Define function to genrate trials
    def generate_grid(p_min, p_max, n_p=5):
        assert p_min <= p_max
        delta = p_max - p_min
        if delta == 0:
            return [p_min]
        if n_p > delta + 1:
            n_p = delta + 1
        step = int(delta/(n_p - 1))
        last = step*(n_p - 1)
        deltas = np.arange(0, last + 1, step, dtype=int)
        remainder = delta - last
        adjust = [0]*(n_p - remainder) + list(range(1, remainder + 1))
        deltas += adjust
        return deltas + p_min

    # Keep two dicts, one to store all evaluated trials
    # indexed by the coef 5-tuple to avoid reevaluating
    # another small dict indexed by top 3 osm to keep
    # track of best fit

    trials = {}
    best_osm = {
        0.0: [], 0.001: [], 0.002: []
    }

    # Initialize search grids
    grids = [
        generate_grid(1, 100),  # diffusion
        generate_grid(1, 100),  # breed
        generate_grid(1, 100),  # spread
        generate_grid(1, 100),  # slope
        generate_grid(1, 100),  # road
    ]

    # Hierarchical loop, 3 levels
    for cstep in range(len(['coarse', 'fine', 'final'])):
        total_iters = np.prod(
            [len(grid) for grid in grids]
        )

        combs = product(*[grid for grid in grids])

        for i, trial in enumerate(combs):
            print(
                f'Executing iteration {i+1}/{total_iters}'
                f' in calibration lvl {cstep} '
                f'with trial {trial}.'
            )

            if trial in trials:
                # If already evaluated retrieve osm
                osm = trials[trial]
            else:
                osm = eval_f(*trial)
                # Add trial to dict
                trials[trial]: osm

            if osm in best_osm.keys():
                # Check if osm ties with any top 3
                # If so, append trial
                best_osm[osm].append(trial)
                continue

            # Retrive top osm metrics
            min_osm = min(best_osm)
            if osm > min_osm:
                # Its al least better than first value
                best_osm.pop(min_osm)
                best_osm[osm] = [trial]
                print(f'New good osm={osm} for trial {trial}.')

        # Adjust grids
        # Array of coefficients, row per coef
        best_trials = np.array(
            [i for j in best_osm.values() for i in j]
        ).T
        assert len(grids) == len(best_trials)
        for i, coef in enumerate(best_trials):
            c_min = coef.min()
            c_max = coef.max()
            # If single value return a finer grid
            if c_min == c_max:
                # Explore grid and select surrouding values
                grid = grids[i]
                idx = np.argwhere(grid == c_min).item()
                c_min = grid[max(0, idx - 1)]
                c_max = grid[min(idx + 1, len(grid) - 1)]
            grids[i] = generate_grid(c_min, c_max)

    max_osm = max(best_osm)
    best_trials = best_osm[max_osm]
    # Select a single trial selecting prioritizing smallest coef values
    # as ordered in the list (dif is more importante than breed)
    best_trial = min(best_trials)

    return best_trial


# def optimize_ax(eval_f):
#     # SE is unknown so just partial driver, it returns osm as float.
#     def driver_eval_f(parameterization):
#         diffusion = parameterization.get('diffusion')
#         breed = parameterization.get('breed')
#         spread = parameterization.get('spread')
#         slope = parameterization.get('slope')
#         road = parameterization.get('road')
#         osm = eval_f(coef_diffusion=diffusion,
#                      coef_breed=breed,
#                      coef_spread=spread,
#                      coef_slope=slope,
#                      coef_road=road)
#         return osm

#     best_parameters, values, experiment, model = optimize(
#         parameters=[
#             {'name': 'diffusion',
#              'type': 'range',
#              'bounds': [1.0, 100.0],
#              'value_type': 'float',
#              },
#             {'name': 'breed',
#              'type': 'range',
#              'bounds': [1.0, 100.0],
#              'value_type': 'float',
#              },
#             {'name': 'spread',
#              'type': 'range',
#              'bounds': [1.0, 100.0],
#              'value_type': 'float',
#              },
#             {'name': 'slope',
#              'type': 'range',
#              'bounds': [1.0, 100.0],
#              'value_type': 'float',
#              },
#             {'name': 'road',
#              'type': 'range',
#              'bounds': [1.0, 100.0],
#              'value_type': 'float',
#              }
#         ],
#         experiment_name="sleuth",
#         objective_name="osm",
#         evaluation_function=driver_eval_f,
#         minimize=False,
#         total_trials=100
#     )
#     print(best_parameters)
#     print(values)


if __name__ == '__main__':
    main()
