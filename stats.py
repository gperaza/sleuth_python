import numpy as np
from scipy.ndimage import convolve, label, center_of_mass
from logconf import logger
from dataclasses import dataclass, fields, field
from pathlib import Path


@dataclass
class StatsVal:
    sng: float = 0.0
    sdc: float = 0.0
    og: float = 0.0
    rt: float = 0.0
    num_growth_pix: float = 0.0
    diffusion: float = 0.0
    spread: float = 0.0
    breed: float = 0.0
    slope_resistance: float = 0.0
    road_gravity: float = 0.0
    edges: float = 0.0
    clusters: float = 0.0
    pop: float = 0.0
    xmean: float = 0.0
    ymean: float = 0.0
    slope: float = 0.0
    rad: float = 0.0
    mean_cluster_size: float = 0.0
    percent_urban: float = 0.0
    growth_rate: float = 0.0
    leesalee: float = 0.0
    diffusion_mod: float = 0.0
    spread_mod: float = 0.0
    breed_mod: float = 0.0
    slope_resistance_mod: float = 0.0
    road_gravity_mod: float = 0.0
    successes: float = 0.0
    z_failure: float = 0.0
    delta_failure: float = 0.0
    slope_failure: float = 0.0
    excluded_failure: float = 0.0

    def reset(self):
        for f in fields(self):
            setattr(self, f.name, 0.0)


@dataclass
class Record:
    year: int
    diffusion: int
    breed: int
    spread: int
    slope: int
    road: int
    write_records: bool
    write_iter: bool
    out_path: Path
    monte_carlo: int = 0
    this_year: StatsVal = field(default_factory=StatsVal)
    average: StatsVal = field(default_factory=StatsVal)
    std: StatsVal = field(default_factory=StatsVal)
    record_file: Path = field(init=False)
    iters_file: Path = field(init=False)

    def __post_init__(self):
        fslug = (
            f'_diffusion_{self.diffusion}'
            f'_breed_{self.breed}'
            f'_spread_{self.spread}'
            f'_slope_{self.slope}'
            f'_road_{self.road}'
        )
        self.record_file = Path(self.out_path / f'records{fslug}.csv')
        self.iters_file = Path(self.out_path / f'iterations{fslug}.csv')

        self.init_iter_file()
        self.init_output_file()

    def update_mean_std(self):
        # Update the mean and sum of squares using
        # Welford's algorithm

        for f in fields(self.this_year):
            value = getattr(self.this_year, f.name)
            prev_mean = getattr(self.average, f.name)
            prev_sum = getattr(self.std, f.name)

            new_mean = prev_mean + (value - prev_mean) / self.monte_carlo
            new_sum = prev_sum + (value - prev_mean) * (value - new_mean)

            setattr(self.average, f.name, new_mean)
            setattr(self.std, f.name, new_sum)

    def compute_std(self):
        for f in fields(self.this_year):
            sum_sq = getattr(self.std, f.name)
            setattr(self.std, f.name, np.sqrt(sum_sq / self.monte_carlo))

    def init_output_file(self):
        with open(self.record_file, 'w') as f:
            f.write('year,mc')
            for fld in fields(self.average):
                f.write(f',{fld.name}_mean')
            for fld in fields(self.std):
                f.write(f',{fld.name}_std')
            f.write('\n')

    def init_iter_file(self):
        with open(self.iters_file, 'w') as f:
            f.write('year,mc')
            for fld in fields(self.this_year):
                f.write(f',{fld.name}')
            f.write('\n')

    def write_fields(self):
        if not self.write_records:
            return
        with open(self.record_file, 'a') as f:
            f.write(f'{self.year},{self.monte_carlo}')
            for fld in fields(self.average):
                f.write(f',{getattr(self.average, fld.name)}')
            for fld in fields(self.std):
                f.write(f',{getattr(self.std, fld.name)}')
            f.write('\n')

    def write_iteration(self):
        if not self.write_iter:
            return
        with open(self.iters_file, 'a') as f:
            f.write(f'{self.year},{self.monte_carlo}')
            for fld in fields(self.this_year):
                f.write(f',{getattr(self.this_year, fld.name)}')
            f.write('\n')


@dataclass
class UrbAttempt:
    successes: int = 0
    z_failure: int = 0
    delta_failure: int = 0
    slope_failure: int = 0
    excluded_failure: int = 0

    def reset(self):
        for f in fields(self):
            setattr(self, f.name, 0)


def compute_base_stats(grid, output_dir, start_year):

    urban_years = list(grid['urban'].year.values)
    assert start_year in urban_years
    start_idx = urban_years.index(start_year)
    urban_years = urban_years[start_idx:]
    assert len(urban_years) >= 4

    stats_vals = [StatsVal() for year in urban_years]

    for i, year in enumerate(urban_years):
        stats_val = stats_vals[i]
        (
            stats_val.edges,
            stats_val.clusters,
            stats_val.pop,
            stats_val.xmean,
            stats_val.ymean,
            stats_val.slope,
            stats_val.rad,
            stats_val.mean_cluster_size,
            stats_val.percent_urban
        ) = compute_stats(grid.urban.sel(year=year).values, grid.slope.values)

    with open(output_dir / 'base_stats.csv', 'w') as f:
        f.write(
            'year,edges,clusters,pop,xmean,ymean,slope,'
            'rad,mean_cluster_size,percent_urban\n'
        )
        for year, sv in zip(urban_years, stats_vals):
            f.write(
                f'{year},{sv.edges},{sv.clusters},{sv.pop},'
                f'{sv.xmean},{sv.ymean},{sv.slope},{sv.rad},'
                f'{sv.mean_cluster_size},{sv.percent_urban}\n'
            )
    logger.info('Base stats saved.')

    return stats_vals, urban_years


def compute_stats(urban, slope):
    # Assuming binarized urban raster
    area = (urban > 0).sum()
    # orginal sleuth code discounts roads and excluded pixels
    # and include roads pixels as urban, which seems weird
    # anyhow, since excluded and roads are fixed, this just rescales
    percent_urban = area / np.prod(urban.size) * 100

    # number of pixels on urban edge
    edges = count_edges(urban)

    # Get a labeled array, by default considers von neumann neighbors
    clusters, nclusters = label(urban)
    assert nclusters > 0

    mean_cluster_size = area / nclusters

    avg_slope = slope[urban > 0].mean()

    # Centroid of urban pixels
    ymean, xmean = center_of_mass(urban)

    # radius of circle of area equal to urban area
    rad = np.sqrt(area/np.pi)

    # Returns a dict of statistics
    # Seems pop and area are the same in orginal SLEUTH code
    return (
        edges,
        nclusters,
        area,
        xmean,
        ymean,
        avg_slope,
        rad,
        mean_cluster_size,
        percent_urban
    )


def count_edges(urban):
    # Peform a convolution to search for edges
    # Orignal SLEUTH code searches in the Von Neuman
    # neighborhood for empty cells. This is akin to perform
    # a convolution with the Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Scipy's ndimage.convolve is faster than signal.convolve
    # for 2D images
    # signal.convolve is more general and handles ndim arrays
    # TODO: splicitly pass output array to save memory
    conv = convolve(urban, kernel, mode='constant', output=int)

    edges = (conv < 0).sum()

    # Alterantive: loop only over urbanized pixels. Urbanize pixel
    # coordinates may be stored in a set, which allows for fast
    # lookup, insertion and deletion. But the convolution operator may
    # be adapted for GPU computation.

    return edges
