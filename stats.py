import numpy as np
import pandas as pd
from scipy.ndimage import convolve, label, center_of_mass
from logconf import logger


var_columns = ["sng", "sdg", "sdc", "og", "rt", "pop", "area",
               "edges", "clusters", "xmean", "ymean", "rad",
               "slope", "mean_cluster_size", "diffusion",
               "spread", "breed", "slope_resistance",
               "road_gravity", "percent_urban", "percent_road",
               "growth_rate", "leesalee", "num_growth_pix"]


def stats_init(scenario, igrid):

    # TODO: clear previous stats
    # regression = pd.DataFrame() # info, one row
    # average = pd.DataFrame() # stats_val_t, year per row
    # std_dev = pd.DataFrame() # stats_val_t, year per row
    # running_total = pd.DataFrame() # stats_val_t, year per row

    # Create dataframes to store and accumulate stats
    # TODO: run only once
    stats_actual = compute_base_stats(igrid)
    stats_file = scenario.output_dir / 'base_stats.csv'
    stats_actual.to_csv(stats_file, index=False)
    logger.info(f'Base stats saved to {stats_file}')

    return stats_actual


def compute_base_stats(grid):
    stat_dicts = []
    for year in grid['urban'].year:
        sdict = compute_stats(grid.urban.sel(year=year), grid.slope)
        sdict['year'] = year.item()
        stat_dicts.append(sdict)

    return pd.DataFrame(stat_dicts)


def compute_stats(urban, slope):
    # Assuming binarized urban raster
    area = urban.sum().item()
    # orginal sleuth code discounts roads and excluded pixels
    # and include roads pixels as urban, which seems weird
    # anyhow, since excluded and roads are fixed, this just rescales
    percent_urban = area/np.prod(urban.size)*100

    # number of pixels on urban edge
    edges = count_edges(urban)

    # Get a labeled array, by default considers von neumann neighbors
    clusters, nclusters = label(urban.values)
    assert nclusters > 0

    mean_cluster_size = area/nclusters

    avg_slope = slope.where(urban > 0).mean().item()

    # Centroid of urban pixels
    ymean, xmean = center_of_mass(urban.values)

    # radius of circle of area equal to urban area
    rad = np.sqrt(area/np.pi)

    # Returns a dict of statistics
    # Seems pop and area are the same in orginal SLEUTH code
    return {'area': area, 'edges': edges,
            'clusters': nclusters, 'pop': area,
            'xmean': xmean, 'ymea': ymean,
            'rad': rad, 'average_slope': avg_slope,
            'mean_cluster_size': mean_cluster_size,
            'percent_urban': percent_urban}


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
    conv = convolve(urban.values, kernel,
                    mode='constant', output=int)

    edges = (conv < 0).sum()

    # Alterantive: loop only over urbanized pixels. Urbanize pixel
    # coordinates may be stored in a set, which allows for fast
    # lookup, insertion and deletion. But the convolution operator may
    # be adapted for GPU computation.

    return edges
