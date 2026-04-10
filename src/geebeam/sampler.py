"""Helper for sampling locations across regions of interest"""

import warnings

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import ee


def _get_crs_scale(
        crs,
        scale_m
    ):
    """Find equivalent scale in m for crs"""
    transform = ee.Projection(crs=crs).atScale(scale_m).getInfo()['transform']
    return transform[0]

def _get_roi(
    sampling_region: str | ee.Geometry | gpd.GeoDataFrame,
    target_crs: str
    ) -> gpd.GeoDataFrame:

    if sampling_region is not None:
        if isinstance(sampling_region, gpd.GeoDataFrame):
            roi_df = sampling_region
        elif isinstance(sampling_region, str):
            roi_df = gpd.read_file(sampling_region)
        elif isinstance(sampling_region, ee.Geometry):
            # Looking for a better way to do this, this is silly
            roi_df = gpd.read_file(json.dumps(sampling_region.getInfo()))
        elif isinstance(sampling_region, shapely.Geometry):
            roi_df = gpd.GeoDataFrame(geometry=[sampling_region], crs=target_crs)
        else:
            raise ValueError("'sampling_region' must be one of"
                                "[str, ee.Geometry, gpd.GeoDataFrame]")
        source_crs = roi_df.crs.to_string()
        if source_crs != target_crs:
            warnings.warn(f'Converting ROI from crs {source_crs} to target_crs: {target_crs}')
            roi_df = roi_df.to_crs(target_crs)
        return roi_df

def _process_sampling_points(
        sampling_points: pd.DataFrame | ee.FeatureCollection | gpd.GeoDataFrame,
        target_crs: str
        ) -> gpd.GeoDataFrame:

    if isinstance(sampling_points, gpd.GeoDataFrame):
        if sampling_points.crs != target_crs:
            raise ValueError('sampling_points projection does not match target_crs.')
        points_gdf = sampling_points

    elif isinstance(sampling_points, pd.DataFrame):
        if 'x' not in sampling_points.columns or 'y' not in sampling_points.columns:
            raise ValueError('If provided as pd.DataFrame, sampling_points must have columns '
                             '`x` and `y` with coordinates for sampling in target crs.')
        points_gdf = gpd.GeoDataFrame(
            sampling_points,
            geometry=gpd.points_from_xy(sampling_points.x, sampling_points.y),
            crs=target_crs
        )

    elif isinstance(sampling_points, ee.FeatureCollection):
        fc_crs = sampling_points.first().geometry().projection().getInfo()
        if fc_crs != target_crs:
            raise ValueError('sampling_points projection does not match target_crs.')
        points_gdf = ee.data.computeFeatures({
                'expression': sampling_points,
                'fileFormat': 'GEOPANDAS_GEODATAFRAME'
                }
            ).set_crs(fc_crs)
    else:
        raise ValueError("'sampling_points' must be one of"
                         "[pd.DataFrame, gpd.GeoDataFrame, ee.FeatureCollection]")

    if 'id' not in points_gdf.columns:
        points_gdf['id'] = np.arange(points_gdf.shape[0])

    if 'split' not in points_gdf.columns:
        points_gdf['split'] = 'full'

    # Get unique splits
    splits = points_gdf['split'].unique()

    # Convert to list of dicts and return
    points_gdf['x'] = points_gdf.geometry.x
    points_gdf['y'] = points_gdf.geometry.y
    return points_gdf.drop(columns='geometry').to_dict('records'), splits

def sample_region_random(
        roi: gpd.GeoDataFrame,
        crs: str,
        n_sample: int,
        random_seed: int = 0,
        buffer_distance: float = 0
        ) -> gpd.GeoDataFrame:
    """Get random points within region of interest."""
    rng = np.random.default_rng(random_seed)
    roi = _get_roi(roi, crs)
    if buffer_distance != 0:
        scale_proj_1m = _get_crs_scale(roi.crs.to_string(), 1)
        roi = roi.dissolve().buffer(scale_proj_1m*buffer_distance)

    sampled_points = gpd.GeoDataFrame(geometry=roi.sample_points(n_sample, rng=rng).geometry.explode())
    sampled_points.index = np.arange(sampled_points.shape[0])
    return sampled_points

def sample_region_grid(
        roi: gpd.GeoDataFrame,
        crs: str,
        stride: int,
        scale: float,
        buffer_distance: float = 0,
        ) -> gpd.GeoDataFrame:
    """Get a regular grid of points covering region of interest"""
    roi = _get_roi(roi, crs)
    scale_proj = _get_crs_scale(roi.crs.to_string(), scale)
    if buffer_distance != 0:
        scale_proj_1m = scale_proj/scale
        roi = roi.dissolve().buffer(scale_proj_1m*buffer_distance)
    xmin, ymin, xmax, ymax = roi.total_bounds
    x_locs = np.arange(xmin, xmax+scale_proj*stride, scale_proj*stride)
    y_locs = np.arange(ymin, ymax+scale_proj*stride, scale_proj*stride)
    meshgrid = np.array(np.meshgrid(x_locs, y_locs)).T.reshape(-1, 2)
    print(meshgrid.shape)
    x_all, y_all = meshgrid[:,0],  meshgrid[:,1]

    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_all, y_all), crs=crs)

    # Clip to region
    points_gdf = gpd.clip(points_gdf, roi)
    points_gdf.index = np.arange(points_gdf.shape[0])

    return points_gdf

def _assign_splits_pandas(df, split_dict, random_seed=0, shuffle=True):
    rng = np.random.default_rng(random_seed)
    if shuffle:
        df = df.sample(frac=1, random_state=rng)

    cur_index = 0
    df['split'] = 'NA'
    split_col_loc = df.columns.get_loc('split')
    for split_name, split_count in split_dict.items():
        df.iloc[cur_index:(cur_index+split_count),split_col_loc] = split_name
        cur_index += split_count

    return df

def _assign_splits_ee(ee_fc, split_dict, random_seed=0, shuffle=True):
    cur_index = 0
    # Shuffle order
    if shuffle:
        ee_fc = ee_fc.randomColumn(seed=random_seed).sort('random')

    output_features = []
    cur_index = 0

    for split_name, split_count in split_dict.items():
        fc_slice = ee_fc.toList(count=split_count,
                               offset=cur_index)
        def _set_split_ee(f):
            return ee.Feature(f).set('split', split_name)

        output_features.append(fc_slice.map(_set_split_ee))
        cur_index += split_count

    # Flatten the list of lists back into a single FeatureCollection
    return ee.FeatureCollection(output_features).flatten()

def split_sets(
        points_gdf: gpd.GeoDataFrame | pd.DataFrame | ee.FeatureCollection,
        split_names: list[str],
        split_ratios: list[float] | None = None,
        split_counts: list[int] | None = None,
        random_seed: int = 0,
        shuffle: bool = True
        ) -> gpd.GeoDataFrame:


    # Find total size
    if isinstance(points_gdf, ee.FeatureCollection):
        total_points = points_gdf.size().getInfo()
    else:
        total_points = points_gdf.shape[0]

    # Some checks, and convert ratios to counts
    if len(split_names) > 0:
        if split_ratios:
            if not np.isclose(np.sum(split_ratios), 1, ):
                raise ValueError('Split ratios do not equal 1.')
            elif len(split_ratios) != len(split_names):
                raise ValueError(f'Length of `split_ratios` ({len(split_ratios)}) must match length of'
                                f' `split_names` ({len(split_names)})')
            else:
                split_counts = np.rint(np.array(split_ratios) * points_gdf.shape[0]).astype(int)
                # Look for rounding errors
                if np.sum(split_counts) > total_points:
                    split_counts[-1] = split_counts[-1] - 1

        elif split_counts:
            if np.sum(split_counts) != total_points:
                raise ValueError(f'Split counts do not sum to total observations {total_points}.')
            elif len(split_ratios) != len(split_names):
                raise ValueError(f'Length of `split_counts` ({len(split_ratios)}) must match length of'
                                f' `split_names` ({len(split_names)})')
            split_counts = np.rint(split_counts).astype(int)

        else:
            raise ValueError('One of `split_ratios` or `split_counts` must be defined')
    else:
        split_names = ['full']
        split_counts = [total_points]


    # Build split dictionary
    split_dict = dict(zip(split_names, split_counts))

    if isinstance(points_gdf, ee.FeatureCollection):
        return _assign_splits_ee(points_gdf, split_dict, random_seed, shuffle)
    else:
        return _assign_splits_pandas(points_gdf, split_dict, random_seed, shuffle)