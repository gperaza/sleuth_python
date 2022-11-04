import configparser
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict


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
        assert self.mode in ['calibrate', 'predict']

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

        pred_date = config['prediction_date']
        self.prediction_start_date = pred_date['START']
        self.prediction_stop_date = pred_date['STOP']

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
