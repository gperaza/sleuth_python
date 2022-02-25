import configparser
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class LandClass:
    num: int
    iD: str
    name: str
    idx: int
    color: int
    EXC: bool
    trans: bool


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
        if mode:
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

        # Look for input files in input dir
        # Urban
        self.urban_data_file = list(
            self.input_dir.glob('*urban*'))
        self.urban_data_file_count = len(self.urban_data_file)
        # Road
        self.road_data_file = list(
            self.input_dir.glob('*roads*'))
        self.road_data_file_count = len(self.road_data_file)
        # Land Use
        self.land_use_data_file = list(
            self.input_dir.glob('*landuse*'))
        self.land_use_data_file_count = len(self.land_use_data_file)
        # Single files
        self.excluded_data_file = list(
            self.input_dir.glob('*excluded*'))[0]
        self.slope_data_file = list(
            self.input_dir.glob('*slope*'))[0]
        self.background_data_file = list(
            self.input_dir.glob('*hillshade*'))[0]

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

        # Deltatron parameters
        # Read in class dictionary
        landclass_f = self.input_dir / 'landclasses.yaml'
        if landclass_f.exists():
            # Load classes if def file exists
            with open(landclass_f, 'r') as f:
                landclass_dict = yaml.safe_load(f)

            self.num_landuse_classes = len(landclass_dict)
            # List of land classes
            self.landuse_class = [
                self.init_landuse_class(key, val)
                for key, val in landclass_dict.items()]
        else:
            # If not def file, deltatron does not run
            self.num_landuse_classes = 0
            self.landuse_class = []

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

    def init_landuse_class(self, name, lu_def):
        if 'type' in lu_def.keys():
            _type = lu_def['type']
        else:
            _type = None

        lu_class = LandClass(
            num=lu_def['value'],
            iD=_type,
            name=name,
            idx=None,
            color=lu_def['color'],
            EXC=(True if _type == 'EXC' else False),
            trans=(False if _type == 'EXC' else True))

        return lu_class
