import sys
import rioxarray as rxr
import xarray as xr
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any
from logconf import logger


# We can heavily simplify grd objects using
# xarray metadata. Start by definig a function that
# loads an array.
# Start by loafing all rasters into ram, then think about
# loading and dropping keeping metadata or lazy loading with dask.

# We need to store:
# max
# min
# shape
# histogram
# year
# raster file path

# What is a grid?
# Seems to be a flattened 1D array with values 0-255
# GRID_P is an array of pixels
# IGRID is a struct holding all grids with arrays of pointer to
# individual grids.
# igrid_obj defines functions on girds(rasters)


@dataclass
class IGrid:
    urban: Dict      # Stores urban xarrays as year: array dict
    road: Dict       # Stores road xarrays as year: array dict
    landuse: Dict    # Stores landuse xarrays as year: array
    excluded: Any    # Stores excluded xarray
    slope: Any       # Stores slope xarray
    background: Any  # Stores hillshade xarray (with water)
    count: int       # Total stored arrays
    location: str    # ??
    nrows: int
    ncols: int
    npixels: int


def load_raster(path):
    # Must store as metadata:
    # - filename
    pass


def grid_dump(f):
    """ Dump some values pertaining to a given grid to f. """
    # Most of these are now useless
    # total_pixels
    # grid_ptr
    # filename
    # packed
    # color_bits
    # bits_per_pixel
    # size_words
    # nrows
    # ncols
    # max
    # min
    # year
    # histogram
    pass


def load_road_raster(path):
    # TODO: gnerate road raster from OSM
    # Or adapt algorithm to run on distance to polylines
    # Must store as metadata
    #   - pixel count
    #   - % of road pixels
    #   - year
    pass


def load_excluded_raster(path):
    # Must store as metadata
    #  - pixel count
    #  -
    pass


def load_urban_raster(path):
    # Must store as metadata:
    # - # urban pixels
    # - # non-urban pixels
    # - ratio of urbanization (0-1)
    pass


def igrid_Dump(xarray, f):
    """ Outputs xarray to raster. """
    pass


def normalizeRoads():
    """ normalize road grids to interval 0-100
    where 100 is max across ALL road rasters """
    # Uselss?
    pass


def validateGrids(igrid, scenario):
    """ validate all input grid values """
    AOK = (validateUrbanGrids(igrid)
           and validateRoadGrids(igrid)
           and validateLanduseGrids(igrid, scenario)
           and validateSlopeGrid(igrid)
           and validateExcludedGrid(igrid)
           and validateBackGroundGrid(igrid))
    if not AOK:
        logger.error("Input data images contain errors. ")
        sys.exit(1)
    else:
        logger.info("Input images validation OK.")

    # TODO: Verify all input images are corregistered.

    return AOK


def validateUrbanGrids(igrid):
    rv = True
    grids = igrid.urban
    ngrids = len(grids)
    if ngrids < 4:
        logger.error("At least 4 urban rasters are needed "
                     "for calibration. Missing input files.")
    for year, grid in grids.items():
        if grid.attrs.num_nonurban_pix == 0:
            logger.error(f"Input grid for year {year}"
                         " is completely urbanized.")
            rv = False
    return rv


def validateRoadGrids(igrid):
    # Assuming a rasterized road grid.
    rv = True
    grids = igrid.road
    ngrids = len(grids)
    if ngrids < 1:
        logger.error("At least 1 road rasters is needed "
                     "for calibration. Missing input files.")
    for year, grid in grids.items():
        if grid.attrs.num_nonroad_pix == 0:
            logger.error(f"Input grid for year {year}"
                         " is 100% roads.")
            rv = False
        elif grid.attrs.num_road_pix == 0:
            logger.error(f"Input grid for year {year}"
                         " is 0% roads.")
            rv = False
    return rv


def validateLanduseGrids(igrid, scenario):
    rv = True
    grids = igrid.landuse
    valid_values = [c['value'] for c in scenario.landuse_class]
    for year, grid in grids.items():
        grid_values = grid.unique()
        for v in grid_values:
            if v not in valid_values:
                logger.error(f"Landuse with value {v}"
                             " appears in landuse file for"
                             f" year {year}.")
                rv = False
    return rv


def validateSlopeGrid(igrid):
    rv = True
    slope_arr = igrid.slope
    if not isinstance(slope_arr, xr.DataArray):
        logger.error("Slope raster is not an xarray.")
        rv = False
    return rv


def validateExcludedGrid(igrid):
    rv = True
    exc_arr = igrid.excluded
    if not isinstance(exc_arr, xr.DataArray):
        logger.error("Excluded raster is not an xarray.")
        rv = False
    return rv


def validateBackGroundGrid(igrid):
    rv = True
    bck_arr = igrid.background
    if not isinstance(bck_arr, xr.DataArray):
        logger.error("Hillshade raster is not an xarray.")
        rv = False
    return rv
