"""Prepare and run Beam pipeline to download image 'chips' from Earth Engine

Example execution:
    python geebeam_main.py \
        --config ./example_config.json \
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="JSON containing configuration dictionary.",
    )
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
    
    return args, beam_args

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

def get_band_names(config):
    # Setup Earth Engine image object with all target bands
    inputs_list = [
        prep_dict[k](config['target_year']-1)
        for k in config['inputs_keys']
    ]
    outputs_list = [prep_dict[config['target_key']](config['target_year'])]
    full_list = inputs_list + outputs_list
    # Get original band names, with system indices prepended (toBands() adds)
    band_names = [
        bn 
        for image in full_list
        for bn in image.bandNames().getInfo()
    ]
    return band_names

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
    def __init__(self, config, scale_x, scale_y):
        self.config = config
        self.prep_dict = {
            'embeddings': self._prep_embeddings,
            'mcd64': self._prep_mcd64,
            'mb_fire': self._prep_mb_burned_area

        }
        self.scale_x = scale_x
        self.scale_y = scale_y

    def setup(self):
        print(f"Initializing Earth Engine for project: {self.config['project_id']}")
        logging.warning("EE setup: starting")
        ee.Initialize(project=self.config['project_id'],
                      opt_url='https://earthengine-highvolume.googleapis.com')
        logging.warning("EE setup: finished")

    def build_prepped_image(self):
        # Set some params
        self.proj = ee.Projection(self.config['proj']).atScale(self.config['scale'])

        # Setup Earth Engine image object with all target bands
        inputs_list = [
            self.prep_dict[k](self.config['target_year']-1)
            for k in self.config['inputs_keys']
        ]
        outputs_list = [self.prep_dict[self.config['target_key']](self.config['target_year'])]
        full_list = inputs_list + outputs_list

        # Get simplified band_names for output (without prefixed image_id)
        # TODO: Add year to distinguish multiple years?
        band_names = ee.List([
            image.bandNames()
            for image in full_list
        ]).flatten()

        # Final prepped image
        return ee.ImageCollection(full_list).toBands().rename(band_names)

    @retry.Retry(tries=5, delay=1, backoff=2)
    def process(self, point):
        """Compute a patch of pixel, with upper-left corner defined by the coords."""
        t0 = time.time()
        logging.warning(f"EE start {point['id']}")
        prepped_image = self.build_prepped_image()

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

    def _prep_embeddings(self, year):
        return (
            ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .mosaic()
            .setDefaultProjection(self.proj)
            .reduceResolution('mean', maxPixels=500)
            )

    def _prep_mcd64(self, year):
        return (
            ee.ImageCollection('MODIS/061/MCD64A1')
            .select('BurnDate')
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .min()
            )

    def _prep_mb_burned_area(self, year):
        return (
            ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_v1')
            .select(['burned_area_{}'.format(year)])
            .reduceResolution('mean', maxPixels=500)
            )

    def _prep_default(self, year):
        """Example prep method"""
        return (
            ee.ImageCollection()
            .mean()
            .reduceResolution('mean', maxPixels=500)
            )

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

def run():
    import logging

    logging.getLogger().setLevel(logging.INFO)

    args, beam_args = parse_args()

    with open(args.config, 'r') as file:
        config_dict = json.loads(file.read())

    # Randomly sample points
    roi = gpd.read_file(args.region_of_interest)
    sample_points  = sample_random_points(roi, config_dict['n_sample'], RNG)
     
    # Convert to dataframe with some metadata attached (including split)
    input_records = points_to_df(
        sample_points, config_dict['validation_ratio']
        ).to_dict('records')

    # Pre-run info:
    scale_x, scale_y = prepare_run_metadata(config_dict)

    # Set up pipeline
    beam_options = PipelineOptions(beam_args,
                                   project=config_dict['project_id'],
                                   region='us-east1',
                                   temp_location='gs://aic-fire-amazon/tmp/',
                                   save_main_session=True,
                                   use_public_ips=False,
                                   network='default',
                                   subnetwork='regions/us-east1/subnetworks/default',
                                   )


    # Execute pipeline
    with beam.Pipeline(options=beam_options) as pipeline:

        # Gather data and split
        training_data, validation_data = (
            pipeline
            | 'Create points' >> beam.Create(input_records)
            | 'Get patch' >> beam.ParDo(EEComputePatch(config_dict, scale_x, scale_y))
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


if __name__ == '__main__':
    run()
