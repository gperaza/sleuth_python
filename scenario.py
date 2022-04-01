import configparser
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict


# @dataclass
# class Coeff:
#     start: int
#     stop: int
#     step: int
#     saved: int
#     current: int
#     best_fit: int
#     sensitivity: float
#     critical: float

#     # def __str__(self):
#     #     name = type(self).__name__
#     #     vars_list = [f'{key}={str(value)}'
#     #                  for key, value in vars(self).items()]
#     #     vars_str = '\n'.join(vars_list)
#     #     return f'{name}\n{vars_str}'

#     def __repr__(self):
#         from pprint import pformat
#         return pformat(vars(self), indent=2)


# @dataclass
# class Coeffs:
#     diffusion: Coeff
#     breed: Coeff
#     spread: Coeff
#     slope_resistance: Coeff
#     road_gravity: Coeff

#     def __repr__(self):
#         from pprint import pformat
#         return pformat(vars(self))


@dataclass
class Coeffs:
    start: dict
    stop: dict
    step: dict
    saved: dict
    current: dict
    best_fit: dict
    sensitivity: dict
    critical: dict

    # def __str__(self):
    #     name = type(self).__name__
    #     vars_list = [f'{key}={str(value)}'
    #                  for key, value in vars(self).items()]
    #     vars_str = '\n'.join(vars_list)
    #     return f'{name}\n{vars_str}'

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self), indent=2)


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
                             'predict', 'restart']
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
        self.mc_iters = mc.getint('MONTE_CARLO_ITERS')
        # This defines the search grid for paramerers
        # and initial values

        # Simulation coefficients
        self.coeffs = Coeffs(
            start=self.init_coeffs(config, 'START'),
            stop=self.init_coeffs(config, 'STOP'),
            step=self.init_coeffs(config, 'STEP'),
            saved=self.init_coeffs(config, 'SAVED'),
            current=self.init_coeffs(config, 'CURRENT'),
            best_fit=self.init_coeffs(config, 'BEST_FIT'),
            sensitivity=self.init_coeffs(config, 'SENSITIVITY'),
            critical=self.init_coeffs(config, 'CRITICAL'))
        # self.coeffs = Coeffs(
        #     diffusion=self.init_coeff(config['DIFFUSION']),
        #     breed=self.init_coeff(config['BREED']),
        #     spread=self.init_coeff(config['SPREAD']),
        #     slope_resistance=self.init_coeff(config['SLOPE']),
        #     road_gravity=self.init_coeff(config['ROAD'])
        # )

        pred_date = config['prediction_date']
        self.prediction_start_date = pred_date['START']
        self.prediction_stop_date = pred_date['STOP']

        # These are new to track progress
        self.current_run = 0
        self.total_runs = self.calc_total_runs()
        self.current_monte_carlo = 0
        self.current_year = 0
        self.stop_year = 0
        # These are a mistery
        self.aux_diffusion_coeff = -1
        self.aux_breed_coeff = -1
        self.aux_diffusion_mult = -1

    def calc_total_runs(self):
        truns = 1
        for start, stop, step in zip(
                self.coeffs.start.values(),
                self.coeffs.stop.values(),
                self.coeffs.step.values()):
            d_runs = len(list(range(start, stop, step)))
            truns *= max(d_runs, 1)
        return truns

    def __str__(self):
        name = type(self).__name__
        vars_list = [f'{key}={str(value)}'
                     for key, value in vars(self).items()]
        vars_str = '\n'.join(vars_list)
        return f'{name}\n{vars_str}'

    def __repr__(self):
        from pprint import pformat
        return pformat(vars(self))

    def init_coeffs(self, config, field):
        paramlist = ['DIFFUSION', 'SPREAD', 'BREED',
                     'SLOPE', 'ROAD']
        isfloat = ['SENSITIVITY', 'CRITICAL']
        coeffs = OrderedDict()
        for param in paramlist:
            # if not config.has_option(param, field):
            #     continue
            if field in isfloat:
                coeffs[param.lower()] =\
                    config[param].getfloat(field, fallback=0.0)
            else:
                coeffs[param.lower()] =\
                    config[param].getint(field, fallback=0)
        return coeffs

        # def init_coeff(self, param):
    #     coef = Coeff(start=param.getint('START'),
    #                  stop=param.getint('STOP'),
    #                  step=param.getint('STEP'),
    #                  current=0,
    #                  saved=0,
    #                  best_fit=param.getint('BEST_FIT'),
    #                  sensitivity=param.getfloat('SENSITIVITY',
    #                                             fallback=0.0),
    #                  critical=param.getfloat('CRITICAL', fallback=0.0))
    #     return coef
