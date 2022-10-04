import numpy as np
import pandas as pd
from scipy.ndimage import convolve, label, center_of_mass
from scipy.stats import pearsonr
from logconf import logger
from dataclasses import dataclass, asdict, fields, field


"""
This module implements functions to calculate and update the
following statistics:
- sng
- sdg
- sdc
- og
- rt
- pop
- area
- edges
- clusters
- xmean
- ymean
- rad
- slope
- mean_cluster_size
- diffusion
- spread
- breed
- slope_resistance
- road_gravity
- percent_urban
- percent_road
- growth_rate
- leesalee
- num_growth_pix

For each statistics, for each year, we calculate the average,
standard deviation, and running total.

We also use this statistics to perform a linear regression during calibration.

We also accumulate statistics on success and failure rate of urbanization
attempts.
"""

# Original SLEUTH keeps a set of structures that are continuously updated.
# In a similar way to writing statistics to a file, we can write to
# predefined dictionaries by creatring functions that update such dictionaries.
# Keeping global variables or updating dicts available in a global scope are no
# good practices, we could instead pass arround a dictionary or data class that
# is updated in the local scope during the simulation.
# To keep the code clean, we can create data classes similar
# to structs.


@dataclass
class Gstats:
    sng: int = 0
    # sdg: int = 0
    sdc: int = 0
    og: int = 0
    rt: int = 0
    pop: int = 0

    def reset(self):
        for f in fields(self):
            setattr(self, f.name, 0)


@dataclass
class StatsVal:
    sng: float = 0.0
    # sdg: float = 0.0
    sdc: float = 0.0
    og: float = 0.0
    rt: float = 0.0
    num_growth_pix: float = 0.0
    diffusion: float = 0.0
    spread: float = 0.0
    breed: float = 0.0
    slope_resistance: float = 0.0
    road_gravity: float = 0.0
    area: float = 0.0
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

    def reset(self):
        for f in fields(self):
            setattr(self, f.name, 0.0)


@dataclass
class Record:
    this_year: StatsVal = field(default_factory=StatsVal)
    average: StatsVal = field(default_factory=StatsVal)
    std: StatsVal = field(default_factory=StatsVal)
    gstats: Gstats = field(default_factory=Gstats)
    monte_carlo: int = 0
    year: int
    run: int
    has_true: bool

    def update_mean_std(self):
        # Update the mean and sum of squares using
        # Welford's algorithm

        for f in fields(self.this_year):
            value = getattr(self.this_year, f.name)
            prev_mean = getattr(self.average, f.name)
            prev_sum = getattr(self.std, f.name)

            new_mean = prev_mean + (value - prev_mean)/self.monte_carlo
            new_sum = prev_sum + (value - prev_mean)*(value - new_mean)

            setattr(self.average, f.name, new_mean)
            setattr(self.sum_sqs, f.name, new_sum)

    def compute_std(self):
        for f in fields(self.this_year):
            sum_sq = getattr(self.std, f.name)
            setattr(self.std, f.name, np.sqrt(sum_sq/self.monte_carlo))


@dataclass
class StatsInfo:
    area: float = 0.0
    edges: float = 0.0
    clusters: float = 0.0
    pop: float = 0.0
    xmean: float = 0.0
    ymean: float = 0.0
    rad: float = 0.0
    average_slope: float = 0.0
    mean_cluster_size: float = 0.0
    percent_urban: float = 0.0


@dataclass
class Aggregate:
    # fmatch: float = 0.0 # For landuse
    actual: float = 0.0
    simualted: float = 0.0
    compare: float = 0.0
    leesalee: float = 0.0
    product: float = 0.0


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



# def stats_update(grd_Z, grd_slope, grd_urban,
#                  coef_diffusion, coef_spread, coef_breed,
#                  coef_slope, coef_road,
#                  num_growth_pix,
#                  record, average, sum_sqs):

#     stats_val = record.this_year
#     # TODO: update run, montecarlo and year in record (in main)

#     # Compute this year stats
#     # stats_val is record.this_year in SLEUTH
#     (stats_val.area,
#      stats_val.edges,
#      stats_val.clusters,
#      stats_val.pop,
#      stats_val.xmean,
#      stats_val.ymean,
#      stats_val.slope,
#      stats_val.rad,
#      stats_val.mean_cluster_size,
#      stats_val.percent_urban) = compute_stats(grd_Z, grd_slope)

#     # Store coefficients
#     stats_val.diffusion = coef_diffusion
#     stats_val.spread = coef_spread
#     stats_val.breed = coef_breed
#     stats_val.slope_resistance = coef_slope
#     stats_val.road_gravity = coef_road

#     # Growth related quantites
#     stats_val.num_growth_pix = num_growth_pix
#     stats_val.growth_rate = 100.0 * stats_val.num_grow_pix/stats_val.pop

#     # If there exists a corresponding urban grid,
#     # calculate intersection over union for calibration
#     if grd_urban is not None:
#         stats_val.leesalee = (
#             np.loagical_and(grd_Z, grd_urban).sum(dtype=float)
#             / np.logical_or(grd_Z, grd_urban).sum(dtype=float)
#         )
#     else:
#         stats_val.leesalee = 0.0

#     # Remember to update mean and sum_sq


def stats_init(igrid, year_list):

    # This function is called in main and is
    # responsible for creating the data classes to
    # store statistics.
    # Instead of clearing data structures for each
    # year, we create new ones and let garbage collector get rid
    # of old ones

    # We need aggregation data structures for mean and std,
    # accumulating data for a single year before being
    # dumped to disk for further processing.

    # Base stats should only be computed once, so
    # we better use this function directly in main
    # stats_actual = compute_base_stats(igrid, output_dir)

    regression = StatsInfo()
    average = {year: StatsVal() for year in year_list}
    sum_sq = {year: StatsVal() for year in year_list}
    # running_total = {year: StatsVal() for year in year_list}

    record = Record(StatsVal())
    aggregate = Aggregate()
    urbanization_attempt = UrbAttempt()

    return (regression, average, sum_sq,
            record, aggregate, urbanization_attempt)


def compute_base_stats(grid, output_dir):
    stats_actual = {}
    stat_dicts = []
    for year in grid['urban'].year:
        stats_val = StatsInfo()
        (stats_val.area,
         stats_val.edges,
         stats_val.clusters,
         stats_val.pop,
         stats_val.xmean,
         stats_val.ymean,
         stats_val.slope,
         stats_val.rad,
         stats_val.mean_cluster_size,
         stats_val.percent_urban) = compute_stats(
             grid.urban.sel(year=year).values, grid.slope.values)
        stats_actual[year.item()] = stats_val

        sdict = asdict(stats_val)
        sdict['year'] = year.item()
        stat_dicts.append(sdict)

    base_stats = pd.DataFrame(stat_dicts).set_index('year')
    stats_file = output_dir / 'base_stats.csv'
    base_stats.to_csv(stats_file, index=True)
    logger.info(f'Base stats saved to {stats_file}')

    return stats_actual


def compute_stats(urban, slope):
    # Assuming binarized urban raster
    area = (urban > 0).sum()
    # orginal sleuth code discounts roads and excluded pixels
    # and include roads pixels as urban, which seems weird
    # anyhow, since excluded and roads are fixed, this just rescales
    percent_urban = area/np.prod(urban.size)*100

    # number of pixels on urban edge
    edges = count_edges(urban)

    # Get a labeled array, by default considers von neumann neighbors
    clusters, nclusters = label(urban)
    assert nclusters > 0

    mean_cluster_size = area/nclusters

    avg_slope = slope[urban > 0].mean()

    # Centroid of urban pixels
    ymean, xmean = center_of_mass(urban)

    # radius of circle of area equal to urban area
    rad = np.sqrt(area/np.pi)

    # Returns a dict of statistics
    # Seems pop and area are the same in orginal SLEUTH code
    return (area, edges, nclusters, area,
            xmean, ymean, avg_slope, rad,
            mean_cluster_size, percent_urban)


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
    conv = convolve(urban, kernel,
                    mode='constant', output=int)

    edges = (conv < 0).sum()

    # Alterantive: loop only over urbanized pixels. Urbanize pixel
    # coordinates may be stored in a set, which allows for fast
    # lookup, insertion and deletion. But the convolution operator may
    # be adapted for GPU computation.

    return edges


def analysis(run, years,
             stats_actual, average,
             regression, aggregate,
             output_dir):
    """ Analysis performed for each run to calculate scoring metrics.
    During calibratio compares to known urbanization.
    During prediction ..."""

    # Do regression for years with true urbanization
    # Years with available ground truth, ignore first
    # as is never simulated, seed value
    true_years = list(stats_actual.keys())
    last_year = true_years.pop(-1)
    # Extract list of metrics
    metrics_true = {'area': [], 'edges': [], 'clusters': [], 'pop': [],
                    'xmean': [], 'ymean': [], 'rad': [], 'average_slope': [],
                    'mean_cluster_size': [], 'percent_urban': []}
    metrics_sim = {k: v[:] for k, v in metrics_true.items()}
    for year in true_years:
        for metric in metrics_true.keys():
            metrics_true[metric].append(getattr(stats_actual, metric))
            metrics_sim[metric].append(getattr(average, metric))
    for metric in metrics_true.keys():
        setattr(regression, metric,
                pearsonr(metrics_true[metric], metrics_sim[metric])**2)

    # The aggregated score is a product of
    # several metrics, including regression coefficients
    aggregate.actual = stats_actual[last_year].pop
    assert aggregate.actual > 0
    aggregate.simulated = average[last_year].pop
    assert aggregate.simulated > 0
    aggregate.leesalee = (sum([average[year].leesalee for year in years])
                          / len(years))
    aggregate.compare = (min(aggregate.actual, aggregate.simulared)
                         / max(aggregate.actual, aggregate.simulared))

    aggregate.product = (aggregate.compare
                         * aggregate.leesalee
                         * regression.edges
                         * regression.clusters
                         * regression.pop
                         * regression.xmean
                         * regression.ymean
                         * regression.rad
                         * regression.average_slope
                         * regression.mean_cluster_size
                         * regression.percent_urban)
