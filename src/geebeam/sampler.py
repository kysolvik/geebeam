"""
Sample points from region of interest
"""

import warnings

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import ee


def get_roi(
    sampling_region: str | ee.Geometry | gpd.GeoDataFrame,
    image_list: list[ee.Image],
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
        else:
            raise ValueError("'sampling_region' must be one of"
                                "[str, ee.Geometry, gpd.GeoDataFrame]")
        if roi_df.crs.to_string() != target_crs:
            roi_df = roi_df.to_crs(target_crs)
        return roi_df
    else:
        warnings.warn('Neither sampling_region nor sampling_points specified.\n'
                      'Defaulting to footprint of the first image in image_list.')
        return image_list[0].geometry()


def sample_random_points(roi: gpd.GeoDataFrame, n_sample: int, rng: np.random.Generator) -> pd.DataFrame:
    """Get random points within region of interest."""
    sampled_points = roi.sample_points(n_sample, rng=rng).geometry.explode().get_coordinates()
    sampled_points.index = np.arange(sampled_points.shape[0])
    x = sampled_points.values[:,0]
    y = sampled_points.values[:,1]
    out_df = pd.DataFrame({
      'y': y,
      'x': x,
    }
    )
    return out_df

def split_train_validation(points_df: pd.DataFrame, validation_ratio: float, rng: np.random.Generator, shuffle: bool = True):
    # Shuffle order
    points_df['split'] = 'train'
    if shuffle:
        points_df = points_df.sample(frac=1, random_state=rng).reset_index(drop=True)
        points_df['id'] = points_df.index

    # Split
    num_train = round(points_df.shape[0]*(1-validation_ratio))
    points_df.loc[num_train:, 'split'] = 'val'

    return points_df

def process_sampling_points(
        sampling_points: pd.DataFrame | ee.FeatureCollection | gpd.GeoDataFrame,
        target_crs: str
        ) -> pd.DataFrame:

    if isinstance(sampling_points, pd.DataFrame):
        if 'x' not in sampling_points.columns or 'y' not in sampling_points.columns:
            raise ValueError('If provided as pd.DataFrame, sampling_points must have columns '
                             '`x` and `y` with coordinates for sampling in target crs.')
        points_df = sampling_points
    elif isinstance(sampling_points, gpd.GeoDataFrame):
        if sampling_points.crs != target_crs:
            raise ValueError('sampling_points projection does not match target_crs.')
        sampling_points['x'] = sampling_points.geometry.x
        sampling_points['y'] = sampling_points.geometry.y
        points_df = sampling_points
    elif isinstance(sampling_points, ee.FeatureCollection):
        fc_crs = sampling_points.first().geometry().projection().getInfo()
        if fc_crs != target_crs:
            raise ValueError('sampling_points projection does not match target_crs.')
        points_df = ee.data.computeFeatures({
                'expression': sampling_points,
                'fileFormat': 'GEOPANDAS_GEODATAFRAME'
                }
            ).set_crs(fc_crs)
        points_df['x'] = points_df.geometry.x
        points_df['y'] = points_df.geometry.y
    else: 
        raise ValueError("'sampling_points' must be one of"
                         "[pd.DataFrame, gpd.GeoDataFrame, ee.FeatureCollection]")
    
    return points_df
