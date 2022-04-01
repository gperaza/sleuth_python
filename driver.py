from timer import timers
import numpy as np
from logconf import logger
from dataclasses import dataclass
import constants as C


@dataclass
class UrbanizationAttempts:
    successes: int = 0
    z_failure: int = 0
    delta_failure: int = 0
    slope_failure: int = 0
    excluded_failure: int = 0


def driver(igrid):
    # Main function to drive the simulation
    timers.DRIVER.start()

    # Monte Carlo Simulation
    monte_carlo()

    timers.DRIVER.stop()


def monte_carlo(scenario, niter):
    coeff_hist = []
    for imc in range(niter):
        scenario.current_monte_carlo = imc

        # Reset the parameters TODO: why?
        scenario.coeffs.current = scenario.coeffs.saved

        # Log coefficients
        coeff_hist.append({'mc': imc, **scenario.coeffs.current})

        # Run full simulation over all years, one mc run
        # Get new urbanization attempts counter
        urbanization_attempts = UrbanizationAttempts()
        grow(igrid, scenario, imc)

        # Update cumulative grid

    # Normalize cumulative urban image
    pass


def fmatch():
    pass


def grow(igrid, scenario, imc):
    timers.GROW.start()

    crun = scenario.current_run
    truns = scenario.total_runs
    mc_iters = scenario.mc_iters
    stop_year = scenario.stop_year
    if scenario.mode == 'predict':
        cyear = scenario.prediction_start_date
    else:
        cyear = igrid['urban'].year[0].item()

    # Initialize z_grid to initial urbanization with
    # value indicating initial growth phase
    seed_urban = igrid.urban[0].values
    igrid['Z'].values = np.where(seed_urban > 0, C.PHASE0G, 0)

    logger.info('******************************************')
    if scenario.mode == 'calibrate':
        logger.info(f'Run = {crun} of {truns} '
                    f'({100*crun/truns:8.1f} percent complete)')
    logger.info('Monte Carlo = {imc+1} of {mc_iters}')
    logger.info('Initial year: {cyear}, Stop year: {stop_year}')

    while cyear < stop_year:
        cyear += 1
        logger.info('Simulating year: {cyear}, stop year: {stop_year}')

        sng = sdg = sdc = og = rt = pop = 0

        timers.SPR_TOTAL.start()
        spread()
        timer.SPR_TOTAL.stop()

    timers.GROW.stop()

