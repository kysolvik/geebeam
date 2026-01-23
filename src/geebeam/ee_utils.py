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
    return ee.List([
        image.bandNames()
        for image in input_list
    ]).flatten()

def build_prepped_image(input_list):
    """Combine a list of EE images into single image"""
    band_names = get_band_names(input_list)

    # Final prepped image
    return ee.ImageCollection(input_list).toBands().rename(band_names)