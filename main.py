import argparse
from timer import timers
from scenario import Scenario
from pathlib import Path
import numpy as np
from pprint import pformat
from grid import igrid_init
from logconf import logger
import subprocess
from dataclasses import asdict
import pandas as pd
import stats


def parse_cl():
    parser = argparse.ArgumentParser(
        description='Simulate urban growth with the SLEUTH model.')
    parser.add_argument('-m', '--mode', dest='mode',
                        choices=['calibrate', 'restart',
                                 'test', 'predict'],
                        help="""Mode to run the simulator on.\n
      TEST mode will perform a single run through the historical
            data using the start values to initialize
            growth, complete the mc interations, and then conclude
            execution.\n
      CALIBRATE will perform monte carlo runs through the
            historical data using every combination of the coefficient
            values indicated for all possible permutations of given
            ranges and increments.\n
      PREDICTION will perform a single run, in monte carlo
            fashion, using the BEST_FIT values for initialization.\n
      RESTART will resume calibration from saved state.
            (Not implemented.)""")
    parser.add_argument('scenario_file',
                        help='Path to file with configuration.')

    args = parser.parse_args()
    ini_path = Path(args.scenario_file)
    mode = args.mode

    return ini_path, mode


def main():
    timers.TOTAL_TIME.start()

    githash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    logger.info(f'Running pySLEUTH with git hash: {githash}')

    # Initialize parameters
    ini_path, mode = parse_cl()
    scenario = Scenario(ini_path, mode)
    logger.info(pformat(scenario))

    # Restore parameterts if restarting
    if scenario.restart:
        # Look at main.c line 132-153, 333-354
        raise NotImplementedError

    # Initialize grid
    igrid = igrid_init(scenario.input_dir, lc_file=None)
    logger.info('Grid is using arround {grid.nbytes/(1024**2)} MB')

    # Initiate PRNG
    prng = np.random.default_rng(seed=scenario.random_seed)

    if scenario.mode == 'calibrate':
        logger.info(f'Total runs: {scenario.total_runs}')
    # TODO: move this to scenario?
    last_run_flag = False
    last_mc_flag = False
    last_run = total_runs - 1
    last_mc = scenario.mc_iters - 1

    # Compute base statistics against which calibration will
    # take place
    df_actual = stats.stats_init(scenario, igrid)

    if scenario.current_run == 0:
        # not restarting
        if scenario.mode != 'predict':
            # stored in control_stats.log
            df_control = pd.DataFrame(
                columns=['Run', 'Product', 'Compare', 'Pop',
                         'Edges', 'Clusters', 'Size', 'Leesalee',
                         'slope', 'percent_urban', 'Xmean', 'Ymean',
                         'Rad', 'Fmatch', 'Diff', 'Brd', 'Sprd',
                         'Slp', 'RG'])
        col_names = ['run', 'year', 'index', 'sng', 'sdg',
                     'sdc', 'og', 'rt', 'pop', 'area', 'edges',
                     'clusters', 'xmean', 'ymean', 'rad', 'slope',
                     'cl_size', 'diffus', 'spread', 'breed',
                     'slp_res', 'rd_grav', '%urban', '%road',
                     'grw_rate', 'leesalee', 'grw_pix']
        df_std = pd.DataFrame(columns=col_names)
        df_avg = pd.DataFrame(columns=col_names)
    else:
        # load data frames from previous run
        df_std = pd.read_csv('std_dev.csv')
        df_avg = pd.DataFrame('avg.csv')
    df_coeff = pd.DataFrame(
        columns=['Run', 'MC', 'Year', 'Diffusion', 'Breed',
                 'Spread', 'SlopeResist', 'RoadGrav'])

    if scenario.mode == 'predict':
        # Prediction, set date and coeffs to best values
        scenario.stop_year = scenario.prediction_stop_date

        scenario.coeffs.current = scenario.coeffs.best_fit
        scenario.coeffs.saved = scenario.coeffs.best_fit

        driver.driver()
    else:
        # Calibration or test, set date to last available in input
        # Loop over full parameter space
        pass

    timers.TOTAL_TIME.stop()


if __name__ == '__main__':
    main()
