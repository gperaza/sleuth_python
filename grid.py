import sys
import rioxarray as rxr
import xarray as xr
import numpy as np
from logconf import logger
import yaml
from collections import Counter
from pprint import pformat
from functools import reduce
import pandas as pd


def load_raster(fpath, raster_to_match, name=None, squeeze=True):
    """Generic function to load a raster as a Data Array

    Looks for a raster in fpath and optionally matches transform and
    bounds of raster_to_match.

    Parameters
    ----------
    fpath : Path
        Path to tif file to load
    raster_to_match : DataArray
        DataArray with raster data to which to match crs and transform
    name : str
        Name assign to DataArray
    squeeze : bool
        Wether to squeeze the loaded array. Defaults to True.

    Returns
    -------
    DataArray
        The raster as a DataArray.
    """

    raster = rxr.open_rasterio(fpath, default_name=name)

    if 'float' in str(raster.dtype):
        # Set nodata of float rasters to nan
        raster = raster.rio.reproject_match(raster_to_match,
                                            nodata=np.nan)
    else:
        raster = raster.rio.reproject_match(raster_to_match)

    if squeeze:
        return raster.squeeze()
    else:
        return raster


def store_metadata(xarr, fname, hist=True):
    """Stores raster metadata as array attributes.

    Stores useful raster metadata as array attributes.
    Useful for debugging.

    Parameters
    ----------
    xarr : DataArray
        input data array with raster data
    fname : Path or str
        path to raster file from which the DataArray was loaded

    """
    xarr.attrs['filename'] = fname
    xarr.attrs['shape'] = xarr.rio.shape
    xarr.attrs['resolution'] = xarr.rio.resolution()
    xarr.attrs['bounds'] = xarr.rio.bounds()
    xarr.attrs['sum'] = xarr.sum().item()
    xarr.attrs['crs'] = xarr.rio.crs.to_epsg()
    xarr.attrs['type'] = xarr.dtype
    if hist:
        xarr.attrs['histogram'] = Counter(xarr.values.ravel())
    xarr.attrs['min'] = xarr.min().item()
    xarr.attrs['max'] = xarr.max().item()


def load_slope_raster(input_dir, raster_to_match,
                      raster_name='slope.tif'):
    """Loads raster with slope data into xarray.

    The slope is commonly derived from a digital elevation model
    (DEM), but other elevation source data may be used. Cell values
    must be in percent slope (0-100).

    Parameters
    ----------
    input_dir : Path
        path to raster file
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform and
        bounds
    raster_name : str
        name of raster file


    Returns
    -------
    DataArray
        DataArray with slope data.

    """

    fname = input_dir / raster_name
    slope = load_raster(fname, raster_to_match, name='slope')
    # Slope is must be reescaled to 0-100,
    # Type must be int, zero values have issues, so use ceil
    # Zero slope values have been reported to unrealistically
    # attract urbanization. (TODO: reference needed)
    slope = np.ceil((slope+0.01)*100/90).astype(np.int32)

    logger.info(f"Loaded slope tif: {fname}")
    store_metadata(slope, fname)
    logger.info(pformat(slope.attrs))

    # Validate
    if slope.min() < 1 or slope.max() > 99:
        logger.error("Slope raster values out of range.")
        sys.exit(1)

    return slope


def load_road_raster(input_dir, raster_to_match):
    """Loads preprocessed road rasters.

    The road influenced growth dynamic included in SLEUTH simulates the
    tendency of urban development to be attracted to locations of
    increased accessibility. A transportation network can have major
    influence upon how a region develops.

    Road information is a combination of 4 rasters: road pixes, i and
    j coordinates of the closes road and the distance to the closest
    road.

    In any region some transportation lines may have more affect upon
    urbanization than others. Through road weighting this type of
    influence may be incorporated into the model. The highest road
    weight will allow a maximum distance, or number of steps, for the
    new urban center to travel along the road. Weights are in the
    range (1-100).

    Parameters
    ----------
    input_dir : Path
        path to raster file
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform and bounds

    Returns
    -------
    List
        List with 4 DataArrays: road pixels, closes road i coordinate,
        closest road j coordinate, distance to closest road

    """

    fname = input_dir / 'roads.tif'
    roads = load_raster(fname, raster_to_match, name='roads')
    logger.info(f"Loaded road tif: {fname}")
    store_metadata(roads, fname)
    # Road pixel count
    roads.attrs['num_road_pix'] = (roads > 0).sum().item()
    roads.attrs['num_nonroad_pix'] = (roads == 0).sum().item()
    roads.attrs['road_percent'] = roads.attrs['num_road_pix']/roads.size
    logger.info(pformat(roads.attrs))

    fname = input_dir / 'road_i.tif'
    road_i = load_raster(fname, raster_to_match, name='road_i')
    logger.info(f"Loaded road tif: {fname}")
    store_metadata(road_i, fname, hist=False)
    logger.info(pformat(road_i.attrs))

    fname = input_dir / 'road_j.tif'
    road_j = load_raster(fname, raster_to_match, name='road_j')
    logger.info(f"Loaded road tif: {fname}")
    store_metadata(road_j, fname, hist=False)
    logger.info(pformat(road_j.attrs))

    fname = input_dir / 'road_dist.tif'
    road_dist = load_raster(fname, raster_to_match, name='road_dist')
    logger.info(f"Loaded road tif: {fname}")
    store_metadata(road_dist, fname, hist=False)
    logger.info(pformat(road_dist.attrs))

    # Validate
    if roads.min() < 0:
        logger.error("Road raster values out of range.")
        sys.exit(1)
    if roads.attrs['num_nonroad_pix'] == 0:
        logger.error('Roads raster is 100% roads.')
        sys.exit(1)
    elif roads.attrs['num_road_pix'] == 0:
        logger.error('Roads raster is 0% roads.')
        sys.exit(1)

    # Normalize road weights to  0-100
    # set values to avoid loosing attrs
    roads.values =\
        (roads.values*100/roads.max().item()).astype(roads.dtype)

    return roads, road_i, road_j, road_dist


def load_excluded_raster(input_dir, raster_to_match,
                         excluded_list=['protected.tif', 'water.tif']):
    """Loads rasters denoting excluded areas.

    The excluded image defines all locations that are resistant to
    urbanization. Areas where urban development is considered
    impossible, open water bodies or national parks for example, are
    given a value of 100 or greater. Locations that are available for
    urban development have a value of zero (0).

    Pixels may contain any value between (0-100) if the representation
    of partial exclusion of an area is desired - unprotected wetlands
    could be an example: Development is not likely, but there is no
    zoning to prevent it.

    Parameters
    ----------
    input_dir : Path
        path to raster file
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform
        and bounds
    exclided_list: List
        list of raster filenames with exclided area information to
        include, it is assume values > 0 are excluded

    Returns
    -------
    DataArray
        DataArray with excluded pixel values

    """

    rasters = [load_raster(input_dir / f, raster_to_match)
               for f in excluded_list]
    rasters_bool = [r > 0 for r in rasters]

    excluded = reduce(np.logical_or, rasters_bool).astype(np.int32)
    # excluded = excluded.where(excluded == 0, 100)
    excluded = xr.where(excluded == 0, excluded, 100, keep_attrs=True)
    excluded.name = 'excluded'
    logger.info(f"Loaded excluded tifs: {excluded_list}")
    store_metadata(excluded, excluded_list)
    excluded.attrs['num_exc_pix'] = (excluded > 99).sum().item()
    excluded.attrs['num_nonexc_pix'] = (excluded == 0).sum().item()
    excluded.attrs['exc_percent'] = excluded.attrs['num_exc_pix']/excluded.size
    logger.info(pformat(excluded.attrs))

    # Validate
    if excluded.min() < 0:
        logger.error("Excluded raster values out of range.")
        sys.exit(1)
    if excluded.attrs['num_nonexc_pix'] == 0:
        logger.error('Excluded raster is 100% excluded.')
        sys.exit(1)

    return excluded


def load_urban_raster(input_dir, raster_to_match, slug='gisa'):
    """Loads rasters denoting urbanized areas.

    For calibration, the earliest urban year is used as the seed, and
    subsequent urban layers, or control years, are used to measure
    several statistical best fit values. For this reason, at least
    four urban layers are needed for calibration: one for
    initialization and three additional for a least-squares
    calculation.

    The expected values are 0: non-urbanized, 1: urbanized

    Parameters
    ----------
    input_dir : Path
        path to raster files
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform
        and bounds
    slug: str
        pattern used to identified urban files using their filename

    Returns
    -------
    DataArray
        DataArray with urban pixel values

    """
    urban_files = sorted(list(input_dir.glob(f'{slug}*.tif')))
    urban_years = [get_year_fpath(f) for f in urban_files]
    urban_arrays = [load_raster(f, raster_to_match, name='urban')
                    for f in urban_files]

    year_attr_dict = {}
    for year, raster, fname in zip(urban_years, urban_arrays, urban_files):
        logger.info(f"Loaded urban tif: {fname}")
        year_attr_dict[year] = {}
        year_attr_dict[year]['num_urb_pix'] = (raster == 1).sum().item()
        year_attr_dict[year]['num_nonurb_pix'] = (raster == 0).sum().item()
        year_attr_dict[year]['urb_ratio'] = \
            year_attr_dict[year]['num_urb_pix']/raster.size

    urban = xr.concat(urban_arrays,
                      dim=pd.Index(urban_years, name='year'))
    urban.attrs['year_attrs'] = year_attr_dict
    store_metadata(urban, urban_files)
    logger.info(pformat(urban.attrs))

    # Validate
    if urban.min() < 0 or urban.max() > 1:
        logger.error("Urban raster values out of range.")
        sys.exit(1)
    if len(urban_years) < 4:
        logger.error("At least 4 urban rasters are needed "
                     "for calibration. Missing input files.")
        sys.exit(1)
    for year, grid in year_attr_dict.items():
        if grid['num_nonurb_pix'] == 0:
            logger.error(f"Input grid for year {year}"
                         " is completely urbanized.")
            sys.exit(1)

    return urban


def load_landcover_raster(input_dir, raster_to_match, lc_dict,
                          slug='copernicus', remap=None):
    """Loads rasters specifying landcover per pixel.


    Parameters
    ----------
    input_dir : Path
        path to raster files
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform
        and bounds
    lc_dict: dict
        dictionary mapping pixel values to classes and additional
        metadata
    slug: str
        pattern used to identified files using their filename
    remap: dict
        if not none, remap classes using specified dictionary

    Returns
    -------
    DataArray
        DataArray with landcover pixel values

    """
    lc_files = sorted(list(input_dir.glob(f'{slug}*.tif')))
    lc_years = [get_year_fpath(f) for f in lc_files]
    lc_arrays = [load_raster(f, raster_to_match, name='landcover')
                 for f in lc_files]

    landcover = xr.concat(lc_arrays,
                          dim=pd.Index(lc_years, name='year'))
    if remap is not None:
        landcover = xr.apply_ufunc(lambda x: remap[x],
                                   landcover, vectorize=True)
    store_metadata(landcover, lc_files)
    landcover.attrs['classes'] = lc_dict

    year_attr_dict = {}
    for year, raster, fname in zip(lc_years, lc_arrays, lc_files):
        logger.info(f"Loaded landcover tif: {fname}")

        year_attr_dict[year] = {}
        year_attr_dict[year]['histogram'] = Counter(
            raster.values.ravel())
        year_attr_dict[year]['ratios'] = {
            k: v/raster.size
            for k, v in year_attr_dict[year]['histogram']
        }

    landcover.attrs['year_attrs'] = year_attr_dict
    logger.info(pformat(landcover.attrs))

    # Validate
    if ((landcover.min() < 0)
            or (landcover.max() > max(lc_dict.values()))):
        logger.error("Landcover raster values out of range.")
        sys.exit(1)
    if len(lc_years) < 2:
        logger.error("At least 2 landcover rasters are needed "
                     "for calibration. Missing input files.")
        sys.exit(1)

    return landcover


def load_clean_landsat(path):
    """ Auxiliary function to load a landsat raster with
     correct metadata. """
    landsat = rxr.open_rasterio(path)
    lbands = landsat.attrs['long_name']
    landsat = landsat.assign_coords(band=list(lbands))
    landsat.attrs.update(long_name='Landsat')
    return landsat


def get_year_fpath(fpath):
    """ Extract year from an input raster file name. """
    return int(str(fpath).split('-')[-1].split('.')[0])


def igrid_init(input_dir,
               lc_file='landclasses.simple.yaml',
               remap_file=None):

    # We choose the landsat raster as the projection to match
    # but landsat raster is not actually needed
    # we can potentially change this in the future
    # Search for landsat raster
    landsat_path = list(input_dir.glob('landsat*.tif'))[0]
    raster_to_match = load_clean_landsat(landsat_path)

    # Load rasters into DataArrays
    slope = load_slope_raster(input_dir, raster_to_match)
    roads, road_i, road_j, road_dist = load_road_raster(
        input_dir, raster_to_match)
    excluded = load_excluded_raster(input_dir, raster_to_match)
    urban = load_urban_raster(input_dir, raster_to_match)

    grids = [urban, slope, roads, road_i, road_j, road_dist, excluded]

    # Search for land class definition file
    remap_dict = None
    if lc_file is not None:
        lc_file = input_dir / lc_file
        with open(lc_file, 'r') as f:
            lc_dict = yaml.safe_load(f)
        if remap_file is not None:
            remap_file = input_dir / remap_file
            with open(remap_file, 'r') as f:
                remap_dict = yaml.safe_load(f)
        landcover = load_landcover_raster(input_dir, raster_to_match,
                                          lc_dict, remap=remap_dict)
        grids.append(landcover)

    # Create empty Z grid where urbanization takes place
    z = xr.DataArray(
        data=np.zeros_like(urban[0].values),
        coords=slope.coords,
        name='Z'
    )
    grids.append(z)

    igrid = xr.merge(grids)
    igrid.attrs['nrows'] = slope.shape[0]
    igrid.attrs['ncols'] = slope.shape[1]
    igrid.attrs['npixels'] = slope.size

    # Verify crs, resolution and shape integrity
    # TODO
    # logger.info()
    # logger.error()

    # TODO Verify landuse, urban year matches
    # logger.error("Last landuse year does not match last urban year.")
    # logger.error(f"   Last landuse year = {}, last urban year = {}.")

    # Generate sets of urbanized and excluded pixels for fast lookup
    excPix = set()
    urbPix = set()

    logger.info("Data Input Files: OK")

    return igrid


def build_pix_set(array):
    pix_set = set()
    # Builds a lookup set of active pixels
    # TODO use xr.where and add coords to set

    return pix_set
