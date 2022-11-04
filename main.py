import sys
import argparse
import configparser
# from timer import timers
from pathlib import Path
import numpy as np
from grid import igrid_init
from logconf import logger
from functools import partial
import subprocess
import stats
import spread


def parse_cl():
    parser = argparse.ArgumentParser(
        description='Simulate urban growth with the SLEUTH model.')
    parser.add_argument('mode',
                        choices=['calibrate', 'predict'],
                        help="""Mode to run the simulator on.\n
      CALIBRATE will perform monte carlo runs through the
            historical data using every combination of the coefficient
            values indicated for all possible permutations of given
            ranges and increments.\n
      PREDICTION will perform a single run, in monte carlo
            fashion, using the BEST_FIT values for initialization.""")
    parser.add_argument('scenario_file',
                        help='Path to file with configuration.')

    args = parser.parse_args()
    ini_path = Path(args.scenario_file)
    mode = args.mode

    return ini_path, mode


def main():
    # timers.TOTAL_TIME.start()

    githash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    logger.info(f'Running pySLEUTH with git hash: {githash}')

    # Initialize parameters
    ini_path, mode = parse_cl()
    logger.info(f'Excecuting in mode: {mode}')

    logger.info(f'Loading configuration file: {ini_path}')
    config = configparser.ConfigParser()
    config.read(ini_path)
    input_dir = Path(config['DEFAULT']['INPUT_DIR'])
    logger.info(f'Input directory: {input_dir}')
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

    # Initialize grid
    igrid = igrid_init(input_dir, lc_file=None)
    logger.info(f'Grid is using arround {igrid.nbytes/(1024**2)} MB')
    sys.exit(1)

    # Initiate PRNG
    prng = np.random.default_rng(seed=random_seed)

    # Compute base statistics against which calibration will take place
    base_stats, urban_years = stats.compute_base_stats(igrid, output_dir)

    if mode == 'calibrate':
        start_year = urban_years[0]
        stop_year = urban_years[-1]
        write_mc = False
        calibrating = True
        write_records = False
    elif mode == 'predicting':
        start_year = urban_years[-1]
        write_mc = True
        calibrating = False
        write_records = True
    else:
        raise ValueError("Mode not supported.")

    # Create a partial function that only accepts coefficient values
    driver = partial(
        spread.driver,
        total_mc=mc_iters,
        start_year=start_year,
        end_year=stop_year,
        calibrating=calibrating,
        grds_urban=igrid,
        urban_years=urban_years,
        calibration_stats=base_stats,
        grd_slope=igrid,
        grd_excluded=igrid,
        grd_roads=igrid,
        grd_roads_dist=igrid,
        grd_road_i=igrid,
        grd_road_j=igrid,
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
        write_records=write_records
    )

    # Simple test, debugging
    osm = driver(
        coef_diffusion=coef_diffusion,
        coef_breed=coef_breed,
        coef_spread=coef_spread,
        coef_slope=coef_slope,
        coef_road=coef_road
    )
    print(osm)

    # timers.TOTAL_TIME.stop()


def optimize_dummy():
    # Dummy function to test,
    # Runs a single set of random coefficients
    pass


def optimize_gridded():
    # Hierarchichal gridded calibration
    # This is the original caribration method of SLEUTH
    pass


def optimize_brute_full():
    # Test all integer coefficient values
    # HUGE seach space, VERY expensive
    pass


def optimize_ax():
    # Bayesian optimization with Facebook's Ax
    # Create evaluation function

    # Create tunable parameters

    # Peform optimization

    pass


def optimize_cd():
    # Simple coordinate descent
    pass


def optimize_local_search():
    # Local hill climbing
    pass


if __name__ == '__main__':
    main()
