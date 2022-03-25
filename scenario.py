import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Coeff:
    start: int
    stop: int
    step: int
    saved: int
    current: int
    best_fit: int
    sensitivity: float
    critical: float


class Scenario:

    def __init__(self, filepath, mode=None):
        # Read in the config file
        config = configparser.ConfigParser()
        config.read(filepath)

        self.filename = filepath
        if mode is not None:
            self.mode = mode
        else:
            self.mode = config['mode']['MODE']
        assert self.mode in ['test', 'calibrate',
                             'prediction', 'restart']
        self.restart = False
        if self.mode == 'restart':
            self.restart = True
            self.mode = 'calibrate'

        paths = config['paths']
        self.input_dir = Path(paths['INPUT_DIR'])
        self.output_dir = Path(paths['OUTPUT_DIR'])

        # Self modification params
        smod = config['self_modification']
        self.critical_low = smod.getfloat('CRITICAL_LOW')
        self.critical_high = smod.getfloat('CRITICAL_HIGH')
        self.boom = smod.getfloat('BOOM')
        self.bust = smod.getfloat('BUST')

        # Simulation Parameters
        mc = config['MC']
        self.random_seed = mc.getint('RANDOM_SEED')
        self.monte_carlo_iters = mc.getint('MONTE_CARLO_ITERS')
        # This defines the search grid for paramerers
        # and initial values
        self.diffusion = self.init_coeff(config['DIFFUSION'])
        self.breed = self.init_coeff(config['BREED'])
        self.spread = self.init_coeff(config['SPREAD'])
        self.slope_resistance = self.init_coeff(config['SLOPE'])
        self.road_gravity = self.init_coeff(config['ROAD'])
        pred_date = config['prediction_date']
        self.prediction_start_date = pred_date['START']
        self.prediction_stop_date = pred_date['STOP']

        # These are new to track progress
        self.current_run = 0
        self.current_monte_carlo = 0
        self.current_year = 0
        # These are a mistery
        self.aux_diffusion_coeff = -1
        self.aux_breed_coeff = -1
        self.aux_diffusion_mult = -1

    def __str__(self):
        name = type(self).__name__
        vars_list = [f'{key}={str(value)}'
                     for key, value in vars(self).items()]
        vars_str = '\n'.join(vars_list)
        return f'{name}\n{vars_str}'

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self))

    def init_coeff(self, param):
        coef = Coeff(start=param.getint('START'),
                     stop=param.getint('STOP'),
                     step=param.getint('STEP'),
                     current=0,
                     saved=0,
                     best_fit=param.getint('BEST_FIT'),
                     sensitivity=param.getfloat('SENSITIVITY',
                                                fallback=0.0),
                     critical=param.getfloat('CRITICAL', fallback=0.0))
        return coef
