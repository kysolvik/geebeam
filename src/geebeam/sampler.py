"""Helper for sampling locations across regions of interest"""

import json
import warnings
from functools import partial

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio import Affine


def _get_crs_scale(
        crs,
        scale_m
    ):
    """Find equivalent scale in m for crs"""
    transform = ee.Projection(crs=crs).atScale(scale_m).getInfo()['transform']
    return transform[0]

def _parse_transform(transform):
    """Return (x0, y0, scale_x, scale_y) from a rasterio Affine or 6-tuple.

    The transform is given in Affine order (a, b, c, d, e, f) =
    (scale_x, shear_x, translate_x, shear_y, scale_y, translate_y). scale_x/scale_y
    are returned as positive pixel-size magnitudes, matching the existing pipeline
    convention (scale_y = -transform[4]). Rotation/shear is not supported.
    """
    a, b, c, d, e, f = tuple(transform)[:6]
    if b != 0 or d != 0:
        raise ValueError('transform must have zero shear/rotation (b and d must be 0).')
    return c, f, abs(a), abs(e)

def _snap_to_grid(x, y, x0, y0, scale_x, scale_y):
    """Snap coordinate(s) to the nearest node of the reference grid.

    Works for both scalar and array inputs. Returns Python floats for scalar input.
    """
    xs = x0 + np.round((np.asarray(x, dtype=float) - x0) / scale_x) * scale_x
    ys = y0 + np.round((np.asarray(y, dtype=float) - y0) / scale_y) * scale_y
    if np.ndim(x) == 0:
        return float(xs), float(ys)
    return xs, ys

def _position_offset(position, patch_size, scale_x, scale_y):
    """Return (dx, dy) from a sampling point to its patch top-left corner.

    ``position`` names where the sampling point sits within the patch; adding the
    returned offset to the point gives the patch's top-left corner (the translateX/
    translateY used to extract pixels).
    """
    _valid_positions = {'center', 'top-left', 'top-right', 'bottom-left', 'bottom-right'}
    if position == 'top-left':
        dx, dy = 0, 0
    elif position == 'center':
        dx = -(patch_size / 2) * scale_x
        dy = -(patch_size / 2) * scale_y
    elif position == 'top-right':
        dx = -patch_size * scale_x
        dy = 0
    elif position == 'bottom-left':
        dx = 0
        dy = -patch_size * scale_y
    elif position == 'bottom-right':
        dx = -patch_size * scale_x
        dy = -patch_size * scale_y
    else:
        raise ValueError(f"Invalid position '{position}'. Must be one of: {sorted(_valid_positions)}")
    return dx, dy

def _pad_locs(locs, step, n):
    """Extend a 1-D array of grid node coords by ``n`` extra steps on each side.

    The added nodes lie on the same lattice as ``locs`` (same ``step``), so the original
    nodes keep their exact float values. Used to widen the candidate grid for the
    'center_clip'/'intersect' selectors without shifting the interior grid.
    """
    if n <= 0:
        return locs
    left = locs[0] - np.arange(n, 0, -1)*step
    right = locs[-1] + np.arange(1, n+1)*step
    return np.concatenate([left, locs, right])

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
            raise TypeError("'sampling_region' must be one of"
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
        raise TypeError("'sampling_points' must be one of"
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
        buffer_distance: float = 0,
        align_transform: Affine | tuple[float] | list[float] | None = None,
        ) -> gpd.GeoDataFrame:
    """Sample random points within a region of interest.

    Args:
        roi: Region to sample from. May be a path to a vector file (str), an
            ``ee.Geometry``, a ``shapely`` geometry, or a ``gpd.GeoDataFrame``;
            a differing CRS is reprojected to ``crs``.
        crs: Target CRS for the returned points (e.g. 'EPSG:4326').
        n_sample: Number of random points to sample/draw.
        random_seed: Seed for reproducible sampling. Defaults to 0.
        buffer_distance: Distance in meters to buffer ``roi`` before sampling, so
            points can fall slightly outside the region. Defaults to 0 (no buffer).
        align_transform: Optional rasterio ``Affine`` or length-6 tuple/list (in
            ``crs`` units). If given, each point is snapped onto that transform's
            pixel grid; a warning is emitted if snapping produces duplicate locations.

    Returns:
        GeoDataFrame of point geometries in ``crs``.
    """
    rng = np.random.default_rng(random_seed)
    roi = _get_roi(roi, crs)
    if buffer_distance != 0:
        scale_proj_1m = _get_crs_scale(roi.crs.to_string(), 1)
        roi = roi.dissolve().buffer(scale_proj_1m*buffer_distance)

    sampled_points = gpd.GeoDataFrame(geometry=roi.sample_points(n_sample, rng=rng).geometry.explode(),
                                      crs=crs)
    if align_transform is not None:
        x0, y0, scale_x, scale_y = _parse_transform(align_transform)
        xs, ys = _snap_to_grid(sampled_points.geometry.x.values,
                               sampled_points.geometry.y.values, x0, y0, scale_x, scale_y)
        n_unique = np.unique(np.column_stack([xs, ys]), axis=0).shape[0]
        n_collisions = len(xs) - n_unique
        if n_collisions > 0:
            warnings.warn(
                f'transform snapped {n_collisions} of {len(xs)} random point(s) onto '
                'locations already occupied by another point (duplicate locations). The '
                'alignment grid is coarse relative to the sampling density; use a finer '
                'align_transform or fewer points to avoid duplicates.',
                UserWarning,
                stacklevel=2
            )
        sampled_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)
    sampled_points.index = np.arange(sampled_points.shape[0])
    return sampled_points

def sample_region_grid(
        roi: gpd.GeoDataFrame,
        crs: str,
        stride: int,
        scale: float | None = None,
        align_transform: Affine | tuple[float] | list[float] | None = None,
        buffer_distance: float = 0,
        patch_size: int | None = None,
        position: str = 'center',
        tile_coverage: str = 'clip',
        ) -> gpd.GeoDataFrame:
    """Build a regular grid of sampling points covering a region of interest.

    Args:
        roi: Region to sample from. May be a path to a vector file (str), an
            ``ee.Geometry``, a ``shapely`` geometry, or a ``gpd.GeoDataFrame``;
            a differing CRS is reprojected to ``crs``.
        crs: Target CRS for the returned points (e.g. 'EPSG:4326').
        stride: Spacing between adjacent points, in pixels (the spacing in ``crs``
            units is ``stride`` times the pixel size).
        scale: Pixel size / export resolution in meters. Required unless
            ``align_transform`` is given.
        align_transform: Optional rasterio ``Affine`` or length-6 tuple/list (in
            ``crs`` units). If given, the grid uses this transform's pixel size
            (``scale`` is ignored) and is anchored to its origin, so every point
            lands on the transform's pixel grid. Defaults to None.
        buffer_distance: Distance in meters to buffer ``roi`` before gridding, to
            help ensure coverage at the edges. Defaults to 0 (no buffer).
        patch_size: Patch size in pixels. Only required when ``tile_coverage`` is not
            'clip'. Defaults to None.
        position: Where each point sits within its patch: 'center' (default),
            'top-left', 'top-right', 'bottom-left', or 'bottom-right'. Only
            matters when ``tile_coverage`` is not 'clip'.
        tile_coverage: Which tiles to keep relative to ``roi``. 'clip' (default)
            keeps a tile if its sampling point falls inside ``roi``; 
            'center_clip' keeps a tile if its patch center falls inside ``roi``.;
            'intersect' keeps a tile if any part of its patch footprint
            touches ``roi``. The latter two require ``patch_size``.

    Returns:
        GeoDataFrame of grid point geometries in ``crs``.

    Raises:
        ValueError: If ``tile_coverage`` is invalid, if 'center_clip'/'intersect' is
            requested without ``patch_size``, or if neither ``scale`` nor
            ``align_transform`` is provided.
    """
    if tile_coverage not in ('clip', 'center_clip', 'intersect'):
        raise ValueError(f"Invalid tile_coverage '{tile_coverage}'. "
                         "Must be 'clip', 'center_clip', or 'intersect'.")
    if tile_coverage in ('center_clip', 'intersect') and patch_size is None:
        raise ValueError(f"tile_coverage='{tile_coverage}' requires patch_size.")
    if align_transform is None and scale is None:
        raise ValueError('`scale` is required unless align_transform is provided.')
    roi = _get_roi(roi, crs)
    if align_transform is not None:
        x0, y0, scale_x, scale_y = _parse_transform(align_transform)
        if buffer_distance != 0:
            scale_proj_1m = _get_crs_scale(roi.crs.to_string(), 1)
            roi = roi.dissolve().buffer(scale_proj_1m*buffer_distance)
        xmin, ymin, xmax, ymax = roi.total_bounds
        step_x, step_y = scale_x*stride, scale_y*stride

        # Align the grid to the reference transform
        x_start = x0 + np.floor((xmin - x0)/scale_x)*scale_x
        y_start = y0 + np.floor((ymin - y0)/scale_y)*scale_y
        x_locs = np.arange(x_start, xmax+step_x, step_x)
        y_locs = np.arange(y_start, ymax+step_y, step_y)
    else:
        scale_proj = _get_crs_scale(roi.crs.to_string(), scale)
        if buffer_distance != 0:
            scale_proj_1m = scale_proj/scale
            roi = roi.dissolve().buffer(scale_proj_1m*buffer_distance)
        scale_x = scale_y = scale_proj
        xmin, ymin, xmax, ymax = roi.total_bounds
        step_x = step_y = scale_proj*stride
        x_locs = np.arange(xmin, xmax+step_x, step_x)
        y_locs = np.arange(ymin, ymax+step_y, step_y)

    if tile_coverage in ('center_clip', 'intersect'):
        # Pad the candidate grid by a full patch on all sides
        n_pad = int(np.ceil(patch_size/stride))
        x_locs = _pad_locs(x_locs, step_x, n_pad)
        y_locs = _pad_locs(y_locs, step_y, n_pad)

    meshgrid = np.array(np.meshgrid(x_locs, y_locs)).T.reshape(-1, 2)
    x_all, y_all = meshgrid[:,0],  meshgrid[:,1]

    points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x_all, y_all), crs=crs)

    if tile_coverage == 'clip':
        # Keep points whose location (the patch reference point) falls inside the region.
        points_gdf = gpd.clip(points_gdf, roi)
    else:
        roi_geom = roi.union_all() if hasattr(roi, 'union_all') else roi.unary_union
        dx, dy = _position_offset(position, patch_size, scale_x, scale_y)
        if tile_coverage == 'center_clip':
            # Keep points whose patch center falls inside the region (position-independent).
            cx, cy = dx + (patch_size/2)*scale_x, dy + (patch_size/2)*scale_y
            selectors = gpd.points_from_xy(x_all + cx, y_all + cy)
        else:  # 'intersect'
            # Keep points whose full patch footprint touches the region.
            w, h = patch_size * scale_x, patch_size * scale_y
            selectors = shapely.box(x_all + dx, y_all + dy, x_all + dx + w, y_all + dy + h)
        mask = gpd.GeoSeries(selectors, crs=crs).intersects(roi_geom)
        points_gdf = points_gdf[np.asarray(mask)]
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

def _set_split_ee(f, split_name):
    return ee.Feature(f).set('split', split_name)

def _assign_splits_ee(ee_fc, split_dict, random_seed=0, shuffle=True):
    cur_index = 0
    # Shuffle order
    if shuffle:
        ee_fc = ee_fc.randomColumn(seed=random_seed).sort('random')

    output_features = []
    cur_index = 0

    for split_name, split_count in split_dict.items():
        _set_cur_split = partial(_set_split_ee(split_name=split_name))
        fc_slice = ee_fc.toList(count=split_count,
                                offset=cur_index)
        output_features.append(fc_slice.map(_set_cur_split))
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
    """Assign sampling points to different splits (e.g., train, validation, test).

    Divides a collection of sampling points into named splits with specified
    proportions or counts. Supports optional shuffling before assignment to
    ensure random distribution across splits.

    Args:
        points_gdf: Collection of sampling points in one of the following formats:
            - gpd.GeoDataFrame: Point geometries with CRS information
            - pd.DataFrame: Must contain 'x' and 'y' coordinate columns
            - ee.FeatureCollection: Earth Engine FeatureCollection of points
        split_names: List of names for each split (e.g., ['train', 'validation', 'test']).
        split_ratios: List of floats specifying the proportion of points for each split.
            Must sum to 1.0 and match the length of split_names. Either this or
            split_counts must be provided.
        split_counts: List of integers specifying the exact number of points for each split.
            Must sum to the total number of points and match the length of split_names.
            Either this or split_ratios must be provided.
        random_seed: Seed for random number generation. Ensures reproducible splits
            when shuffle=True. Defaults to 0.
        shuffle: Whether to randomly shuffle points before assigning to splits.
            Defaults to True.

    Returns:
        GeoDataFrame or FeatureCollection with a 'split' column containing the assigned
        split name for each point. Also includes an 'id' column with point identifiers
        if not already present.

    Raises:
        ValueError: If split_ratios do not sum to 1.0, or if split_counts do not sum
            to total observations, or if the lengths of split_names, split_ratios,
            and/or split_counts do not match. Also raised if neither split_ratios
            nor split_counts is provided.
    """

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

    # Assign id so we can see splits
    if 'id' not in points_gdf.columns:
        points_gdf['id'] = np.arange(points_gdf.shape[0])

    if isinstance(points_gdf, ee.FeatureCollection):
        return _assign_splits_ee(points_gdf, split_dict, random_seed, shuffle)
    else:
        return _assign_splits_pandas(points_gdf, split_dict, random_seed, shuffle)
