import argparse
from timer import timers
from scenario import Scenario
from pathlib import Path
import numpy as np
from pprint import pformat
from grid import igrid_init
from logconf import logger
import subprocess


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
        # Look at main.c line 333-354
        raise NotImplementedError

    # Initialize grid
    igrid = igrid_init(scenario.input_dir, lc_file=None)

    # Initiate PRNG
    prng = np.random.default_rng(seed=scenario.random_seed)

    # Some log files for stats, should we use dataframes?
    # open xypoints.log (fpverd2)
    xypoints_cols = ['%run', 'mc', 'diff', 'breed',
                     'spread', 'slope', 'road_grav',
                     'year area']
    # open slope.log (fpverd3)
    # open ratio.log (fpverd4)

    

    timers.TOTAL_TIME.stop()


if __name__ == '__main__':
    main()
