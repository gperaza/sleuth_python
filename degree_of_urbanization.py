import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
import rioxarray as rxr
from scipy.ndimage import label, convolve, center_of_mass
from pathlib import Path
import sys
sys.path.append('..')
from src import data_sources as dts


def get_bbox(city, country, data_path, buff=10):
    """Calculates a bounding box for city in country.

    This functions uses data from the GHSL's 2015 definition of urban
    centers to locate a city and obtains bounding box with a buffer arround
    the defintion of functional urban area.

    Parameters
    ----------
    city : str
        The city to search for.
    country : str
        The country the city belongs to.
    data_path : Path
        Path where data files are stored in a subdirectory
        output/cities/{file}
    buff : int or float
        Buffer in kilometers arround the functional urban area used to
        define the bounding box.

    Returns
    -------
    bbox : Polygon
        Shapely polygon for the bounding box.
    uc : GeoDataFrame
        Single line GeoDataFrame with the urban center for city.
    fua : GeoDataFrame
        Single line GeoDataFrame with the functional urban area for city.

    """
    cities_uc = gpd.read_file(data_path / 'output/cities/cities_uc.gpkg')
    cities_fua = gpd.read_file(data_path / 'output/cities/cities_fua.gpkg')
    uc = cities_uc.loc[(cities_uc.country == country)
                       & (cities_uc.city == city)]
    fua = cities_fua.loc[(cities_fua.country == country)
                         & (cities_fua.city == city)]
    bbox = dts.get_roi(fua, buff=10, buff_type='km').geometry[0]

    return bbox, uc, fua


def tif_from_bbox_s3(s3_path, local_path, bbox,
                     bucket='tec-expansion-urbana-p', nodata_to_zero=False):
    """Downloads a windowed raster with bounds defined by bbox from an
    COG stored in an Amaxon S3 bucket and saves a local geotiff file.

    Uses bbox to define a search window to download a portion of a raster from
    a Cloud Optimized Geotiff stored in a public bucket in Amazon S3
    in s3_path.
    Saves the raster to a local geotiff file in local_path.

    Parameters
    ----------
    s3_path : str
        The relative path of the COG in S3.
    local_path : Path
        Path to local file to store raster.
    bbox : Polygon
        Shapely Polygon defining the raster's bounding box.
    bucket : str
        Name of the S3 bucket with the COG.
    nodata_to_zero : bool
        If True, sets the output raster's nodata attribute to 0.

    """
    subset, profile = np_from_bbox_s3(s3_path, bbox, bucket, nodata_to_zero)

    with rio.open(local_path, 'w', **profile) as dst:
        dst.write(subset)


def np_from_bbox_s3(s3_path, bbox,
                    bucket='tec-expansion-urbana-p',
                    nodata_to_zero=False):
    """Downloads a windowed raster with bounds defined by bbox from an
    COG stored in an Amaxon S3 bucket and stores it in memory in a numpy array.

    Uses bbox to define a search window to download a portion of a raster from
    a Cloud Optimized Geotiff stored in a public bucket in Amazon S3
    in s3_path.
    Store the raster in memory in a Numpy array.

    Parameters
    ----------
    s3_path : str
        The relative path of the COG in S3.
    bbox : Polygon
        Shapely Polygon defining the raster's bounding box.
    bucket : str
        Name of the S3 bucket with the COG.
    nodata_to_zero : bool
        If True, sets the output raster's nodata attribute to 0.

    Returns
    -------
    subset : np.array
        Numpy array with raster data.
    profile : dict
        Dictionary with geographical properties of the raster.

    """
    url = f'http://{bucket}.s3.amazonaws.com/{s3_path}'

    with rio.open(url) as src:
        profile = src.profile.copy()
        transform = profile['transform']
        window = rio.windows.from_bounds(*bbox.bounds, transform)
        window = window.round_lengths().round_offsets()
        # The transform is specified as (dx, rot_x, x_0 , rot_y, dy, y0)
        new_transform = src.window_transform(window)
        profile.update({
            'height': window.height,
            'width': window.width,
            'transform': new_transform})
        subset = src.read(window=window)
    if nodata_to_zero:
        subset[subset == profile['nodata']] = 0

    return subset, profile


def landscan_yearly_s3(bbox, data_path,
                       s3_dir='landscan_global',
                       bucket='tec-expansion-urbana-p'):
    """Downloads LandScan windowed rasters for each available year.

    Takes a bounding box (bbox) and downloads the corresponding raster from a
    set of yearly LandScan COGs stored on Amazon S3.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    data_path : Path
        Path to directory to store LandScan rasters.
    s3_dir : str
        Relative path to LandScan data on S3.
    bucket : str

    """
    for year in range(2000, 2021):
        tif_from_bbox_s3(
            f'{s3_dir}/landscan-global-{year}.tif',
            data_path / f'landscan_{year}.tif',
            bbox, bucket, nodata_to_zero=True
        )


def gisa_yearly_s3(bbox, data_path,
                   s3_path='GISA_v02_COG.tif',
                   bucket='tec-expansion-urbana-p'):
    """Downloads GISA v2 windowed rasters for each available year.

    Takes a bounding box (bbox) and downloads the corresponding raster from a
    the global GISA COG stored on Amazon S3. Then process the original raster
    data to extract yearly urbanization and creates a geotiff for each year.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    data_path : Path
        Path to directory to store yearly GISA rasters.
    s3_dir : str
        Relative path to GISA COG on S3.
    bucket : str

    """
    # Get gisa in original encoding
    gisa, profile = np_from_bbox_s3(s3_path, bbox, bucket)

    # Save original encoding
    with rio.open(data_path / 'GISA2_all.tif', 'w', **profile) as dst:
        dst.write(gisa)

    # Extract yearly tifs
    gisa_dict = (
        {1972: 1, 1978: 2}
        | {year: val + 3 for val, year in enumerate(range(1985, 2020))}
    )
    for year, pix_val in gisa_dict.items():
        gisa_binary = np.logical_and(0 < gisa, gisa <= pix_val).astype('uint8')
        with rio.open(data_path / f'GISA2_{year}.tif', 'w', **profile) as dst:
            dst.write(gisa_binary)


def osm_water_s3(bbox, data_path,
                 s3_path='OSM_WaterLayer.tif',
                 bucket='tec-expansion-urbana-p'):
    """Downloads a OSM Wayer Layer windowed raster.

    Takes a bounding box (bbox) and downloads the corresponding raster from a
    the global OSM Water Layer COG stored on Amazon S3.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    data_path : Path
        Path to directory to store the raster.
    s3_dir : str
        Relative path to OSM Water Layer COG on S3.
    bucket : str

    """
    tif_from_bbox_s3(s3_path, data_path / 'OSM_WaterLayer.tif',
                     bbox, bucket, nodata_to_zero=True)


def download_rasters_s3(bbox, data_path):
    """Downloads a data rasters requierd for Degree of Urbanization.

    Takes a bounding box (bbox) and downloads the OSM Water Layer,
    yearly GISA v2, and yearly LandScan windowed rasters from COGs
    stored on Amazon S3. If rasters already exist, ommits download.

    Parameters
    ----------
    bbox : Polygon
        Shapely Polygon defining the bounding box.
    data_path : Path
        Path to directory to store the rasters.

    """
    if not (data_path / 'OSM_WaterLayer.tif').exists():
        osm_water_s3(bbox, data_path)
    if not list(data_path.glob('landscan*.tif')):
        landscan_yearly_s3(bbox, data_path)
    if not list(data_path.glob('GISA*.tif')):
        gisa_yearly_s3(bbox, data_path)


def lat_2_meter(lat, delta):
    """Converts from degrees of latitued to meters.

    Takes a given latitud and a small distance delta in degreed,
    and ouput that same delta in meters.
    Conversion formula is taken from:
    https://en.wikipedia.org/wiki/Latitude#Meridian_distance_on_the_ellipsoid


    Parameters
    ----------
    lat : float or np.array
        Latitude at which to perform conversion.
    delta : float
        Small distance delta in degrees.

    Returns
    -------
    delta_in_m : float
        Delta distance in meters.

    """
    lat_rad = np.pi/180*abs(lat)

    a = 6378137.0
    b = 6356752.3142
    e2 = (a**2 - b**2)/a**2

    lat_in_m = a*(1 - e2) / (1 - e2 * np.sin(lat_rad)**2)**(3/2)
    lat_in_m *= np.pi/180
    delta_in_m = lat_in_m*delta

    return delta_in_m


def lon_2_meter(lat, delta):
    """Converts from degrees of longitude to meters.

    Takes a given longitude and a small distance delta in degrees,
    and ouput that same delta in meters.
    Conversion formula is taken from:
    https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude


    Parameters
    ----------
    lat : float or np.array
        Latitude at which to perform conversion.
    delta : float
        Small distance delta in degrees.

    Returns
    -------
    delta_in_m : float
        Delta distance in meters.

    """
    lat_rad = np.pi/180*abs(lat)

    a = 6378137.0
    b = 6356752.3142
    e2 = (a**2 - b**2)/a**2

    lon_in_m = a*np.cos(lat_rad) / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    lon_in_m *= np.pi/180

    delta_in_m = lon_in_m*delta

    return delta_in_m


def get_area_grid(raster_xr, units):
    """Takes an input raster in lat lon coordinates and outputs a corresponding
    grid with each pixel's area in {units}.

    Parameters
    ----------
    raster_xr : DataArray
        Input raster in lat-lon coordinates.
    units : str
        Area units of the output raster, either 'm', 'km', or 'ha'.

    Returns
    -------
    area_grid : np.array
        Numpy array with each pixels area in {units}.

    """

    c_factor = {'m': 1, 'km': 1/1e6, 'ha': 1/1e4}

    x_ar = raster_xr.coords['x'].values
    y_ar = raster_xr.coords['y'].values
    lon_grid, lat_grid = np.meshgrid(x_ar, y_ar)
    delta_x, delta_y = [abs(x) for x in raster_xr.rio.resolution()]

    area_grid = lat_2_meter(
        lat_grid, delta_y) * lon_2_meter(lat_grid, delta_x)
    area_grid *= c_factor[units]

    return area_grid


def pop_2_density(raster, units='ha', save=False):
    """Tranforms a populatiuon counts raster into a population density raster.

    Takes a raster in lat-lon with population counts as input
    and outputs a raster with population density in people per {units}^2.
    Takes into account pixel area variability in lat-lon coordinates.

    Parameters
    ----------
    raster : DataArray or Path
        Input raster in population counts in lat-lon.
    units : str
        Units of lenght to use when calculating areas.
        Can be 'm', 'km', or 'ha'.
    save : bool
        Wether to save a geotiff to disk.

    Returns
    -------
    density_xr : DataArray
        Raster array with population density.

    """

    # Load population raster
    if isinstance(raster, Path):
        pop_rxr = rxr.open_rasterio(raster)
    elif isinstance(raster, xr.DataArray):
        pop_rxr = raster
        save = False
    else:
        print('Input must be either path or DataArray.')
        return

    area_grid = get_area_grid(pop_rxr, units)

    density_ar = pop_rxr.values / area_grid
    density_xr = pop_rxr.copy(data=density_ar)
    if save:
        fname = f'{raster.stem}-density-{units}-{raster.suffix}'
        density_xr.rio.to_raster(raster.parent / fname)

    return density_xr


def clip_utm_box(raster, bbox, orig_crs, dest_crs):
    bbox_x, bbox_y = gpd.GeoDataFrame(
        {'geometry': [bbox]},
        crs=orig_crs
    ).to_crs(dest_crs).geometry[0].exterior.xy
    bbox_x = sorted(list(set(bbox_x)))
    bbox_y = sorted(list(set(bbox_y)))
    min_x, max_x = bbox_x[1], bbox_x[2]
    min_y, max_y = bbox_y[1], bbox_y[2]

    X = raster.coords['x']
    idx = np.searchsorted(X, min_x)
    min_x = X[idx].item()
    idx = np.searchsorted(X, max_x)
    max_x = X[idx - 1].item()

    Y = sorted(raster.coords['y'].values)
    idx = np.searchsorted(Y, min_y)
    min_y = Y[idx].item()
    idx = np.searchsorted(Y, max_y)
    max_y = Y[idx - 1].item()

    return raster.rio.clip_box(minx=min_x, miny=min_y, maxx=max_x, maxy=max_y)


def load_input_data(cache_path, year, bbox):

    # Get population density grid
    landscan = rxr.open_rasterio(cache_path / f'landscan_{year}.tif').squeeze()
    density = pop_2_density(landscan, units='km')
    utm_crs = density.rio.estimate_utm_crs()
    density_utm = density.rio.reproject(
        dst_crs=utm_crs, resolution=1000,
        resampling=rio.enums.Resampling.average,
        nodata=0)
    density_utm = clip_utm_box(density_utm, bbox, landscan.rio.crs, utm_crs)

    # Get built-up grid
    builtup = rxr.open_rasterio(cache_path / f'GISA2_{year}.tif').squeeze()
    builtup_utm = builtup.astype(float).rio.reproject_match(
        density_utm, resampling=rio.enums.Resampling.average, nodata=0)

    # Get land fraction grid
    water = rxr.open_rasterio(cache_path / 'OSM_WaterLayer.tif').squeeze()
    # Water is coded as > 0
    water.values = (water.values > 0).astype(float)
    water_utm = water.rio.reproject_match(
        density_utm, resampling=rio.enums.Resampling.average, nodata=0)
    land_fraction = 1 - water_utm

    return density_utm, builtup_utm, land_fraction


def find_urban_centers(pop_array, builtup_array,
                       u_center_density=1500, u_center_pop=50000,
                       builtup_trshld=0.5):

    u_center_array = np.zeros_like(pop_array, dtype='uint8')
    # Apply density based classification
    u_center_array[pop_array >= u_center_density] = 1
    # Apply builtup condition
    u_center_array[builtup_array >= builtup_trshld] = 1

    # Label deaults to 4-connectivity
    clusters, nclusters = label(u_center_array)

    # Find their total population and remove them from
    # urban center array if necessary
    labels = []
    for lbl in range(1, nclusters+1):
        mask = clusters == lbl
        total_pop = pop_array[mask].sum()
        if total_pop < u_center_pop:
            u_center_array[mask] = 0
            clusters[mask] = 0
        else:
            labels.append(lbl)

    # Fill gaps and smooth borders, majority rule
    # Apply per urban center, find all candidates
    # for allocation
    kernel = np.array([[1,  1, 1],
                       [1, -8, 1],
                       [1,  1, 1]])
    for lbl in labels:
        # This needs to be done iteratively until no more additions are performed
        current_center = (clusters == lbl).astype(int)
        while True:
            # Find number of neighbors of each cell
            # Non urban pixels have neighbor values 0-8, while urban
            # pixels have -8-0
            n_nbrs = convolve(current_center, kernel,
                              mode='constant', output=int)
            # New cells are non urban pixels with >=5 neighbors
            mask = n_nbrs >= 5
            if mask.sum() == 0:
                break
            # Update both current center and urban_center_array
            current_center[mask] = 1
            u_center_array[mask] += 1
    # Cells added to more than one urban center have counts > 1.
    # Remove them
    u_center_array[u_center_array > 1] = 0

    # Fill holes smaller than 15 km^2
    # Invert image
    inverted = 1 - u_center_array
    # Find all holes larger than 15
    holes, nholes = label(inverted)
    for h in range(1, nholes+1):
        mask = holes == h
        if mask.sum() <= 15:
            u_center_array[mask] = 1

    return u_center_array


def find_urban_clusters(pop_array,
                        u_cluster_density=300, u_cluster_pop=5000):

    u_cluster_array = np.zeros_like(pop_array, dtype='uint8')

    # Apply density based classification
    u_cluster_array[pop_array >= u_cluster_density] = 1

    # Label clusters using 8 contiguity
    kernel8 = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    clusters, nclusters = label(u_cluster_array, structure=kernel8)

    # Find their total population and remove them from
    # if necessary
    labels = []
    for lbl in range(1, nclusters+1):
        mask = clusters == lbl
        total_pop = pop_array[mask].sum()
        if total_pop < u_cluster_pop:
            u_cluster_array[mask] = 0
            clusters[mask] = 0
        else:
            labels.append(lbl)

    return u_cluster_array


def dou_lvl1(density_utm, builtup_utm, lvl_1_classes,
             u_center_density=1500, u_center_pop=50000,
             builtup_trshld=0.5,
             u_cluster_density=300, u_cluster_pop=5000):

    u_cluster_array = find_urban_clusters(density_utm.values,
                                          u_cluster_density, u_cluster_pop)
    u_center_array = find_urban_centers(density_utm.values,
                                        builtup_utm.values,
                                        u_center_density, u_center_pop,
                                        builtup_trshld)

    dou_array = np.full_like(density_utm.values,
                             lvl_1_classes['Rural'], dtype='uint8')
    dou_array[u_cluster_array > 0] = lvl_1_classes['Urban Cluster']
    dou_array[u_center_array > 0] = lvl_1_classes['Urban Center']
    dou_rxr = density_utm.copy(data=dou_array)

    return dou_rxr


def get_stats_dict(class_array, pop_array, builtup_array, classes, year,
                   cell_area=1, connectivity=4):
    stat_list = []
    if not isinstance(classes, dict):
        if connectivity == 8:
            class_array, ncenters = label(
                class_array, structure=np.ones((3, 3)))
        else:
            class_array, ncenters = label(class_array)
        classes = {f'{classes} {lbl}': lbl for lbl in range(1, ncenters+1)}

    for c, l in classes.items():
        mask = class_array == l
        stat_dict = {'Grupo': c}
        stat_dict['year'] = year
        stat_dict['Area'] = mask.sum()*cell_area
        stat_dict['Area_fraction'] = (stat_dict['Area']
                                      / (class_array.size*cell_area))
        stat_dict['Pob'] = pop_array[mask].sum()*cell_area
        stat_dict['Pop_density'] = stat_dict['Pob']/stat_dict['Area']
        stat_dict['Pop_fraction'] = stat_dict['Pob']/pop_array.sum()
        stat_dict['Builtup_area'] = builtup_array[mask].sum()*cell_area
        stat_dict['Builtup_fraction'] = (stat_dict['Builtup_area']
                                         / (builtup_array.sum()*cell_area))
        stat_dict['centroid'] = center_of_mass(mask)
        stat_list.append(stat_dict)
    return stat_list


def get_stats_df(dou_array, pop_array, builtup_array, lvl_1_classes, year):
    df = pd.DataFrame(
        get_stats_dict(
            dou_array, pop_array, builtup_array, lvl_1_classes, year)
        + get_stats_dict(
            (dou_array == lvl_1_classes['Urban Center']).astype(int),
            pop_array, builtup_array, 'Center', year
        )
        + get_stats_dict(
            (dou_array > 1).astype(int),
            pop_array, builtup_array, 'Cluster', year, connectivity=8
        )
    )
    return df


def dou_for_year(bbox, cache_path, year, lvl_1_classes,
                 u_center_density=1500, u_center_pop=50000,
                 builtup_trshld=0.5,
                 u_cluster_density=300, u_cluster_pop=5000):

    (density_utm,
     builtup_utm,
     land_fraction) = load_input_data(cache_path, year, bbox)

    dou_array = dou_lvl1(density_utm, builtup_utm, lvl_1_classes,
                         u_center_density, u_center_pop,
                         builtup_trshld,
                         u_cluster_density, u_cluster_pop)

    stats_df = get_stats_df(dou_array.values,
                            density_utm.values,
                            builtup_utm.values,
                            lvl_1_classes, year)

    return dou_array, stats_df


def find_closest(df, centroid):
    d = np.inf
    for i, cent in enumerate(df.centroid):
        dd = np.sum((cent - centroid)**2)
        if dd < d:
            d = dd
            idx = i
    return df.iloc[idx]


def stats_for_largest_cluster(df_stats):
    # We want only urban clusters
    df_clusts = df_stats[df_stats.Grupo.str.startswith('Cluster')]

    # Get year lists
    years = sorted(df_stats.year.unique())

    # For first year find largest cluster in population
    df_year = df_clusts[df_clusts.year == years[0]]
    centroid = df_year.iloc[df_year.Pob.argmax()].centroid

    # Loop over years choosing always the same cluster
    largest_clustrs = []
    for year in years:
        df_year = df_clusts[df_clusts.year == year]
        closest = find_closest(df_year, centroid)
        largest_clustrs.append(closest)
    df_largest = pd.concat(largest_clustrs, axis=1)
    return df_largest.T


def full_run(bbox, cache_path):
    lvl_1_classes = {
        'Urban Center': 3,
        'Urban Cluster': 2,
        'Rural': 1
    }

    download_rasters_s3(bbox, cache_path)

    df_list = []
    for year in range(2000, 2020):
        dou_xr, df_stats = dou_for_year(bbox, cache_path, year, lvl_1_classes)
        dou_xr.rio.to_raster(cache_path / f'dou_{year}.tif')
        df_list.append(df_stats)
    df_stats = pd.concat(df_list)
    df_stats['centroid'] = df_stats.centroid.apply(lambda x: np.array(x))
    df_largest = stats_for_largest_cluster(df_stats)

    df_stats.to_csv(cache_path / 'dou_stats.csv')
    df_largest.to_csv(cache_path / 'dou_largest.csv')
