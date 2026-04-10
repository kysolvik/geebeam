"""
Utilities for cleaning, combining, and serializing/deserializing EE objects.
"""

import io

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

def build_prepped_image(input_list, split_processing=False):
    """Combine a list of EE images into single image"""
    band_names = get_band_names(input_list)
    band_names_flat = ee.List(band_names).flatten()

    if split_processing:
        band_groups = band_names
    else:
        band_groups = [band_names_flat]
    # Final prepped image
    prepped_im = ee.ImageCollection(input_list).toBands().rename(band_names_flat)
    return prepped_im, band_groups, band_names_flat.getInfo()

def list_to_im(input_list):
    prepped_im, _ = build_prepped_image(input_list)
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
                'translateX': point['x'],
                'shearY': 0,
                'scaleY': scale_y,
                'translateY': point['y']
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
