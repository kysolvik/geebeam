"""
Utilities for cleaning, combining, and serializing/deserializing EE objects.
"""
import ee

def deserialize(obj_json):
    """Deserialize Earth Engine JSON DAG"""
    return ee.deserializer.fromJSON(obj_json)

def serialize(obj_ee):
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
    return prepped_im, band_groups


def list_to_im(input_list):
    prepped_im, _ = build_prepped_image(input_list)
    return prepped_im
