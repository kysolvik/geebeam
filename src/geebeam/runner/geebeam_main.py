"""Prepare and run Beam pipeline to download image 'chips' from Earth Engine

Example execution:
    python geebeam_main.py \
        --output_path gs://aic-fire-amazon/results/ \
        --region_of_interest ./data/Limites_RAISG_2025/Lim_Raisg.shp \
        --runner DataflowRunner \
        --experiments=use_runner_v2 \
        --max_num_workers=16 \
        --num_workers=8 \
        --requirements_file ./pipeline_requirements.txt

"""
import ee
import io
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import geopandas as gpd
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import random
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tfx_bsl.coders import example_coder
import logging
import time
import argparse
import os
import json

from google.api_core import retry


SEED = 54
RNG = np.random.default_rng(SEED)


def deserialize(obj_json):
    return ee.deserializer.fromJSON(obj_json)

def serialize(obj_ee):
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
    band_names = get_band_names(input_list)

    # Final prepped image
    return ee.ImageCollection(input_list).toBands().rename(band_names)

def sample_random_points(roi: gpd.GeoDataFrame, n_sample: int, rng: np.random.Generator)->np.array:
    """Get random points within region of interest."""
    sample_df = roi.sample_points(n_sample, rng=rng).geometry.explode().get_coordinates()
    sample_df.index = np.arange(sample_df.shape[0])
    return sample_df.values

def points_to_df(sample_points: ArrayLike, validation_ratio: float) -> pd.DataFrame:
  """Produce training/validation Dataframe from points array"""
  lon = sample_points[:,0]
  lat = sample_points[:,1]
  out_df = pd.DataFrame({
    'lat': lat,
    'lon': lon,
    'split': 'train',
  }
  )

  # Shuffle order
  out_df = out_df.sample(frac=1, random_state=RNG).reset_index(drop=True)
  out_df['id'] = out_df.index

  # Split
  num_train = round(out_df.shape[0]*(1-validation_ratio))
  out_df.loc[num_train:, 'split'] = 'val'

  return out_df

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def dict_to_example(element):
    """"Convert structured numpy array to tf.Example proto."""
    # First add metadata
    feature = {
        'id': _int64_feature(element['id']),
        'lat': _float_feature(element['lat']),
        'lon': _float_feature(element['lon']),
    }

    # Build image feature with named bands
    for im_feat in element['array'].dtype.names:
        feature[im_feat] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = element['array'][im_feat].flatten()))

    # Build example and serialize
    return tf.train.Example(
        features = tf.train.Features(feature = feature)).SerializeToString()

def array_to_example(structured_array):
    """"Convert structured numpy array to tf.Example proto."""
    feature = {}
    for f in structured_array.dtype.names:
        feature[f] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = structured_array[f].flatten()))
    return tf.train.Example(
        features = tf.train.Features(feature = feature))

def split_dataset(element, n_partitions) -> int:
    split_mappings = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    return split_mappings[element['split']]

def prepare_run_metadata(config):
    ee.Initialize(project=config['project_id'])

    proj = ee.Projection(config['proj']).atScale(config['scale'])
    proj_dict = proj.getInfo()

    scale_x = proj_dict['transform'][0]
    scale_y = -proj_dict['transform'][4]

    return scale_x, scale_y

class EEComputePatch(beam.DoFn):
    """DoFn() for computing EE patch
    
    config (dict): Dictionary containing configuration settings 
        in the following key:value pairs:
            project_id (str): Google Cloud project id
            patch_size (int): Patch size, in pixels, of output chips
            scale (float): Final scale, in m
            target_year (int): Year of prediction
            target_key (str): Name of target data, corresponds to key in self.prep_dict
            inputs_keys (list): Names of input data, correspond to keys in self.prep_dict
            proj (str): Projection, e.g. "EPSG:4326"
    """
    def __init__(self, config, serialized_image, scale_x, scale_y):
        self.config = config
        self.serialized_image = serialized_image
        self.scale_x = scale_x
        self.scale_y = scale_y

    def setup(self):
        print(f"Initializing Earth Engine for project: {self.config['project_id']}")
        logging.warning("EE setup: starting")
        ee.Initialize(project=self.config['project_id'],
                      opt_url='https://earthengine-highvolume.googleapis.com')
        logging.warning("EE setup: finished")

    def deserialize(self, obj_json):
        return ee.deserializer.fromJSON(obj_json)

    @retry.Retry(tries=5, delay=1, backoff=2)
    def process(self, point):
        """Compute a patch of pixel, with upper-left corner defined by the coords."""
        t0 = time.time()
        logging.warning(f"EE start {point['id']}")
        prepped_image = self.deserialize(self.serialized_image)

        # Make a request object.
        request = {
            'expression': prepped_image,
            'fileFormat': 'NPY',
            'grid': {
                'dimensions': {
                    'width': self.config['patch_size'],
                    'height':self.config['patch_size']
                },
                'affineTransform': {
                    'scaleX': self.scale_x,
                    'shearX': 0,
                    'translateX': point['lon'],
                    'shearY': 0,
                    'scaleY': self.scale_y,
                    'translateY': point['lat']
                },
                'crsCode': 'EPSG:4326',
            },
        }

        raw = ee.data.computePixels(request)
        logging.warning(f"EE bytes: {len(raw)}")

        if not raw:
            raise RuntimeError("Empty EE response")

        try:
            arr = np.load(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(f"Failed to load NPY for coords {point['id']}") from e

        logging.warning(
            f"EE end {point['id']}, took {time.time() - t0:.1f}s, bytes={len(raw)}"
        )

        out_dict = dict(point)
        out_dict['array'] = arr
        yield out_dict

class WriteTFExample(beam.PTransform):
    """Write example"""
    def __init__(self, output_dir, file_name_suffix='.tfrecord.gz'):
        self.output_dir = output_dir
        self.file_name_suffix = file_name_suffix

    def expand(self, pcoll):
        return (
            pcoll
            | 'Write to TFRecord' >> beam.io.WriteToTFRecord(
                self.output_dir,
                file_name_suffix=self.file_name_suffix
            )
        )

def run(config, input_list):
    import logging

    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save TFRecord files (local or GCS).",
    )
    parser.add_argument(
        "--region_of_interest",
        required=True,
        help="Local geopandas-readable file of region to sample from randomly."
    )

    # Beam args are leftover after parsing known args
    args, beam_args = parser.parse_known_args()

    # Randomly sample points
    roi = gpd.read_file(args.region_of_interest)
    sample_points  = sample_random_points(roi, config['n_sample'], RNG)
     
    # Convert to dataframe with some metadata attached (including split)
    input_records = points_to_df(
        sample_points, config['validation_ratio']
        ).to_dict('records')

    # Pre-run info:
    scale_x, scale_y = prepare_run_metadata(config)

    # Set up pipeline
    beam_options = PipelineOptions(beam_args,
                                   project=config['project_id'],
                                   region='us-east1',
                                   temp_location='gs://aic-fire-amazon/tmp/',
                                   save_main_session=True,
                                   use_public_ips=False,
                                   network='default',
                                   subnetwork='regions/us-east1/subnetworks/default',
                                   )

    # Prepare and serialize inputs
    prepped_image = build_prepped_image(input_list)
    serialized_image = serialize(prepped_image)

    # Execute pipeline
    with beam.Pipeline(options=beam_options) as pipeline:

        # Gather data and split
        training_data, validation_data = (
            pipeline
            | 'Create points' >> beam.Create(input_records)
            | 'Get patch' >> beam.ParDo(EEComputePatch(config, serialized_image, scale_x, scale_y))
            | 'Split dataset' >> beam.Partition(split_dataset, 2)
        )

        # Convert to TF examples
        training_examples = (
            training_data
            | 'Train to tf.Example' >> beam.Map(dict_to_example)
        )
        validation_examples = (
            validation_data
            | 'Val to tf.Example' >> beam.Map(dict_to_example)
        )

        # Calculate stats on training data
        decoder = example_coder.ExamplesToRecordBatchDecoder()
        stats = (
            training_examples
            | 'Batch' >> beam.BatchElements(
                min_batch_size=10,
                max_batch_size=100)
            | 'Decode to arrow' >> beam.Map(lambda b: decoder.DecodeBatch(b))
            | 'Generate Statistics' >> tfdv.GenerateStatistics()
        )
        stats | 'Write stats' >> tfdv.WriteStatisticsToTFRecord(
            os.path.join(args.output_path, 'stats.tfrecord'))

        # Write out examples
        (training_examples 
         | 'Write training' >> WriteTFExample(
             os.path.join(args.output_path, 'training'))
        )
        (validation_examples 
         | 'Write validation' >> WriteTFExample(
             os.path.join(args.output_path, 'validation'))
        )
    
    # Infer schema
    stats = tfdv.load_statistics(
        os.path.join(args.output_path, 'stats.tfrecord')
    )

    schema = tfdv.infer_schema(stats)

    tfdv.write_schema_text(
        schema,
        os.path.join(args.output_path, 'schema.pbtxt')
    )