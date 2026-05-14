"""
Utilities for cleaning, combining, and serializing/deserializing EE objects.
"""

import io
import warnings
from collections import Counter

import ee
import numpy as np

def _deserialize(obj_json):
    """Deserialize Earth Engine JSON DAG"""
    return ee.deserializer.fromJSON(obj_json)

def _serialize(obj_ee):
    """Serialize Earth Engine object to JSON for Dataflow workers"""
    return ee.serializer.toJSON(obj_ee)

def get_band_names(input_list):
    """Get simplified band_names for output (without prefixed image_id)

    TODO: Add year to distinguish multiple years?
    """
    return [
        image.bandNames()
        for image in input_list
    ]

def _dedupe_band_names(band_names):
    """Append _0, _1, ... to all occurrences of any band name that appears more than once."""
    dupes = {name for name, count in Counter(band_names).items() if count > 1}
    if not dupes:
        return band_names, False
    occurrence = {}
    result = []
    for name in band_names:
        if name in dupes:
            idx = occurrence.get(name, 0)
            result.append(f'{name}_{idx}')
            occurrence[name] = idx + 1
        else:
            result.append(name)
    return result, True

def build_prepped_image(input_list, split_processing=False):
    """Combine a list of EE images into single image"""
    band_names = get_band_names(input_list)
    # One API call: nested list gives both per-image structure and flat names
    per_image_bands = ee.List(band_names).getInfo()
    flat_bands = [b for bands in per_image_bands for b in bands]

    deduped_bands, had_duplicates = _dedupe_band_names(flat_bands)
    if had_duplicates:
        dupes = sorted({b for b in flat_bands if flat_bands.count(b) > 1})
        warnings.warn(
            f'Duplicate band names found across image_list {dupes}. '
            'Appending _0, _1, ... suffixes to all occurrences.',
            UserWarning,
            stacklevel=3
        )

    if split_processing:
        # Reconstruct per-image groups using deduplicated names
        band_groups = []
        i = 0
        for bands in per_image_bands:
            n = len(bands)
            band_groups.append(deduped_bands[i:i + n])
            i += n
    else:
        band_groups = [deduped_bands]

    prepped_im = ee.ImageCollection(input_list).toBands().rename(deduped_bands)
    return prepped_im, band_groups, deduped_bands

def list_to_im(input_list):
    prepped_im, *_ = build_prepped_image(input_list)
    return prepped_im

def get_pixels(im, point, patch_size, scale_x, scale_y, crs_code):
    # Make a request object.
    request = {
        'expression': im,
        'fileFormat': 'NPY',
        'grid': {
            'dimensions': {
                'width': patch_size,
                'height': patch_size
            },
            'affineTransform': {
                'scaleX': scale_x,
                'shearX': 0,
                'translateX': point.get('x_topleft', point['x']),
                'shearY': 0,
                'scaleY': scale_y,
                'translateY': point.get('y_topleft', point['y'])
            },
            'crsCode': crs_code
        },
    }

    raw = ee.data.computePixels(request)

    if not raw:
        raise RuntimeError("Empty EE response")

    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load NPY for coords {point['id']}") from e

    return arr

def get_pixels_allbands(im, band_groups, point, patch_size, scale_x, scale_y, crs_code):
    """Get pixels for all groups of bands"""
    out_ars = []
    for band_list in band_groups:
        prepped_image = im.select(band_list)
        out_ars.append(
            get_pixels(
            im=prepped_image,
            point=point,
            patch_size=patch_size,
            scale_x=scale_x,
            scale_y=scale_y,
            crs_code=crs_code
            )
        )
    return out_ars
