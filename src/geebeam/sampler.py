"""
Sample points from region of interest
"""

import numpy as np
import pandas as pd
import geopandas as gpd


def get_roi(sampling_region):
    return gpd.read_file(sampling_region)


def sample_random_points(roi: gpd.GeoDataFrame, n_sample: int, rng: np.random.Generator)->pd.DataFrame:
    """Get random points within region of interest."""
    sampled_points = roi.sample_points(n_sample, rng=rng).geometry.explode().get_coordinates()
    sampled_points.index = np.arange(sampled_points.shape[0])
    lon = sampled_points.values[:,0]
    lat = sampled_points.values[:,1]
    out_df = pd.DataFrame({
      'lat': lat,
      'lon': lon,
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