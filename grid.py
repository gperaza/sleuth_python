import rioxarray as rxr
import xarray as xr
import numpy as np
from logconf import logger
from collections import Counter
from pprint import pformat
from functools import reduce
import pandas as pd
import geopandas as gpd
from geocube.api.core import make_geocube
from scipy.spatial import KDTree
from rasterio.enums import Resampling
import sys


def load_raster(fpath, raster_to_match, name=None, squeeze=True,
                nodata=np.nan, resampling=Resampling.nearest):
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
    nodata: int or float
        Value to fill regions not covered during reprojection.

    Returns
    -------
    DataArray
        The raster as a DataArray.
    """

    raster = rxr.open_rasterio(fpath, default_name=name)

    # if raster_to_match is None:
    #     # Reprojecto to UTM
    #     raster = raster.rio.reproject(
    #         dst_crs=raster.rio.estimate_utm_crs(),
    #         resolution=100,
    #         nodata=nodata,
    #         resampling=resampling)
    if raster_to_match is not None:
        dif_crs = raster_to_match.rio.crs.to_string() != raster.rio.crs.to_string()
        dif_bounds = raster_to_match.rio.bounds() != raster.rio.bounds()
        dif_res = raster_to_match.rio.resolution() != raster.rio.resolution()
        # print(dif_crs, dif_bounds, dif_res)
    if (
            (raster_to_match is not None)
            and (dif_crs or dif_bounds or dif_res)
    ):
        raster = raster.rio.reproject_match(
            raster_to_match, nodata=nodata, resampling=resampling)

    # if 'float' in str(raster.dtype):
    #     # Set nodata of float rasters to nan
    #     raster = raster.rio.reproject_match(raster_to_match,
    #                                         nodata=np.nan)
    # else:
    #     raster = raster.rio.reproject_match(raster_to_match)

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
    xarr.attrs['crs'] = xarr.rio.crs.to_string()
    xarr.attrs['type'] = xarr.dtype
    if hist:
        xarr.attrs['histogram'] = Counter(xarr.values.ravel())
    xarr.attrs['min'] = xarr.min().item()
    xarr.attrs['max'] = xarr.max().item()


def load_slope_raster(input_dir, raster_to_match, nodata,
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
    slope = load_raster(fname, raster_to_match,
                        nodata=nodata, name='slope',
                        resampling=Resampling.average)
    # Slope is must be reescaled to 0-100,
    # Type must be int, zero values have issues, so use ceil
    # Zero slope values have been reported to unrealistically
    # attract urbanization. (TODO: reference needed)
    slope = np.ceil((slope+0.01)*100/90).astype(np.int32)

    logger.info(f"Loaded slope tif: {fname}")
    store_metadata(slope, fname)
    logger.info(pformat(slope.attrs))

    # Validate
    if slope.min().item() < 1 or slope.max().item() > 99:
        logger.error("Slope raster values out of range.")
        sys.exit(1)

    return slope


def create_road_raster(input_dir, raster_to_match, d_metric=np.inf):
    # Assuming a vector dataset of roads exists,
    # which have been obtained from OSM,
    # with a column named 'weight' indicating road
    # accesibility, the following codes correspond
    # to OSM road types

    # https://wiki.openstreetmap.org/wiki/Key:highway
    road_types_dict = {
        'motorway': 7,
        'trunk': 6,
        'primary': 5,
        'secondary': 4,
        'tertiary': 3,
        'unclassified': 2,
        'residential': 1,
        'living_street': 1,
        'unknown': 1
    }

    # Load the vector file and keep up to tertiary roads,
    # as to not confuse SLEUTH with residential roads.
    edges = gpd.read_file(input_dir / 'roads.gpkg')
    edges = edges[edges['weight'] > road_types_dict['unclassified']]

    # Burn in roads into raster
    # Raster to match needs to be in the final intended
    # projection and bounds, less indexing erros will occur
    # upon reprojection and clipping.
    roads = make_geocube(vector_data=edges, measurements=['weight'],
                         like=raster_to_match, fill=0)['weight']

    # Remove roads from border to avoid neighbor lookup out of bounds
    roads.values[0, :] = 0
    roads.values[:, 0] = 0
    roads.values[-1, :] = 0
    roads.values[:, -1] = 0

    # Create bands with nearest roads indices
    roads.values = roads.values.astype(np.int32)
    road_idx = np.column_stack(np.where(roads.values > 0))
    # KDtree for fast neighbor lookup
    tree = KDTree(road_idx)
    # Explicitly create raster grid
    I, J = roads.values.shape
    grid_i, grid_j = np.meshgrid(range(I), range(J), indexing='ij')
    # Get coordinate pairs (i,j) to loop over
    coords = np.column_stack([grid_i.ravel(), grid_j.ravel()])
    # Find nearest road for every lattice point
    # p=inf is chebyshev distance (moore neighborhood)
    dist, idxs = tree.query(coords, p=d_metric)
    # Create bands
    dist = dist.reshape(roads.shape).astype(np.int32)
    road_i = road_idx[:, 0][idxs].reshape(roads.shape).astype(np.int32)
    road_j = road_idx[:, 1][idxs].reshape(roads.shape).astype(np.int32)

    roads.name = 'roads'
    road_i = roads.copy(data=road_i)
    road_i.name = 'road_i'
    road_j = roads.copy(data=road_j)
    road_j.name = 'road_j'
    road_dist = roads.copy(data=dist)
    road_dist.name = 'dist'

    # Save individual rasters and store metadata
    # roads_ar.rio.to_raster(data_path / 'roads.tif', dtype=np.int32)
    fname = input_dir / 'roads.tif'
    store_metadata(roads, fname)
    # Road pixel count
    roads.attrs['num_road_pix'] = (roads > 0).sum().item()
    roads.attrs['num_nonroad_pix'] = (roads == 0).sum().item()
    roads.attrs['road_percent'] = roads.attrs['num_road_pix']/roads.size
    roads.rio.to_raster(fname)
    logger.info(f"Created/loaded road tif: {fname}")
    logger.info(pformat(roads.attrs))

    fname = input_dir / 'road_i.tif'
    store_metadata(road_i, fname, hist=False)
    road_i.rio.to_raster(fname)
    logger.info(f"Created/loaded road tif: {fname}")
    logger.info(pformat(road_i.attrs))

    fname = input_dir / 'road_j.tif'
    store_metadata(road_j, fname, hist=False)
    road_j.rio.to_raster(fname)
    logger.info(f"Created/loaded road tif: {fname}")
    logger.info(pformat(road_j.attrs))

    fname = input_dir / 'road_dist.tif'
    store_metadata(road_dist, fname, hist=False)
    road_dist.rio.to_raster(fname)
    logger.info(f"Created/loaded road tif: {fname}")
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
    match_idxs = (
        roads.values[road_i.values.ravel(), road_j.values.ravel()] > 0).all()
    if not match_idxs:
        logger.error('Road idxs do not match roads.')
        sys.exit(1)

    # Normalize road weights to  0-100
    # set values to avoid loosing attrs
    roads.values =\
        (roads.values*100/roads.max().item()).astype(roads.dtype)

    # Check no roads on border
    if not (
            (roads.values[0, :] == 0).all()
            and (roads.values[:, 0] == 0).all()
            and (roads.values[-1, :] == 0).all()
            and (roads.values[:, -1] == 0).all()
    ):
        logger.error("Roads on border.")
        sys.exit(1)

    return roads, road_i, road_j, road_dist


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
    weight will increase the probability of accepting urbanization.
    Weights are in the range (1-100).

    Parameters
    ----------
    input_dir : Path
        path to raster file
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform and bounds

    Returns
    -------
    roads: DataArray

    road_i: DataArray

    road_j: DataArray

    road_dist: DataArray

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


def load_excluded_raster(input_dir, raster_to_match, nodata):
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

    Water raster refers to the fraction of the pixel occupied by water.
    Threshold water pixels to those with more than half content of water.

    Parameters
    ----------
    input_dir : Path
        path to raster file
    raster_to_match : DataArray
        DataArray with raster to which to match geotransform
        and bounds

    Returns
    -------
    DataArray
        DataArray with excluded pixel values

    """

    protected = load_raster(input_dir / 'protected.tif',
                            raster_to_match, nodata=nodata,
                            resampling=Resampling.mode)
    water = load_raster(input_dir / 'water.tif',
                        raster_to_match, nodata=nodata,
                        resampling=Resampling.average)
    rasters_bool = [
        protected > 0,
        water > 0.5
    ]

    excluded = reduce(np.logical_or, rasters_bool).astype(np.int32)
    orig_attrs = excluded['spatial_ref'].attrs
    # excluded = excluded.where(excluded == 0, 100)
    excluded = xr.where(excluded == 0, excluded, 100, keep_attrs=True)
    excluded['spatial_ref'].attrs = orig_attrs
    excluded.name = 'excluded'
    logger.info("Loaded excluded tifs: protected.tif, water.tif")
    store_metadata(excluded, ['protected.tif', 'water.tif'])
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


def load_urban_raster(input_dir, raster_to_match, nodata, slug='urban'):
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
    nodata : int
        Value for encoding missing data.
    slug: str
        pattern used to identified urban files using their filename

    Returns
    -------
    DataArray
        DataArray with urban pixel values

    """

    urban_files = sorted(list(input_dir.glob(f'{slug}*.tif')))
    urban_years = [get_year_fpath(f) for f in urban_files]
    urban_arrays = [load_raster(
        f, raster_to_match, nodata=nodata, name='urban',
        resampling=Resampling.mode)
                    for f in urban_files]

    # Test shape and bounds
    bounds = urban_arrays[0].rio.bounds()
    shape = urban_arrays[0].rio.shape
    crs = urban_arrays[0].rio.crs.to_string()
    for urb_ar in urban_arrays:
        assert urb_ar.rio.bounds() == bounds
        assert urb_ar.rio.shape == shape
        assert urb_ar.rio.crs.to_string() == crs

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


def get_year_fpath(fpath):
    """ Extract year from an input raster file name. """

    return int(str(fpath).split('_')[-1].split('.')[0])


def igrid_init(input_dir):
    # Input dir must be a Path object

    # We now load GHS rasters in Mollweide projection
    # which is equal area.
    # GHS rastres are already 100m and must be
    # processed using the Degree of Urbanizarion methodology
    # modified for 100m resolution.

    # Load rasters into DataArrays
    # We need to load the GHS raster first as they set
    # the projection.

    urban = load_urban_raster(input_dir, raster_to_match=None,
                              nodata=0)
    slope = load_slope_raster(input_dir, raster_to_match=urban, nodata=0.0)
    roads, road_i, road_j, road_dist = create_road_raster(
        input_dir, raster_to_match=urban)
    excluded = load_excluded_raster(input_dir, raster_to_match=urban,
                                    nodata=0)
    grids = [urban, slope, roads, road_i, road_j, road_dist, excluded]

    # Search for land class definition file
    # remap_dict = None
    # if lc_file is not None:
    #     lc_file = input_dir / lc_file
    #     with open(lc_file, 'r') as f:
    #         lc_dict = yaml.safe_load(f)
    #     if remap_file is not None:
    #         remap_file = input_dir / remap_file
    #         with open(remap_file, 'r') as f:
    #             remap_dict = yaml.safe_load(f)
    #     landcover = load_landcover_raster(input_dir, raster_to_match,
    #                                       lc_dict, remap=remap_dict)
    #     grids.append(landcover)

    # Create empty Z grid where urbanization takes place
    # z = xr.DataArray(
    #     data=np.zeros_like(urban[0].values),
    #     coords=slope.coords,
    #     name='Z'
    # )
    # grids.append(z)

    # Create delta grid for temporal urbanization storage
    # delta = xr.DataArray(
    #     data=np.zeros_like(urban[0].values),
    #     coords=slope.coords,
    #     name='delta'
    # )
    # grids.append(delta)

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
