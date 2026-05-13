"""Integration tests for full Beam pipeline runs (no EE calls)."""

import glob
import os
import pytest
import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from unittest.mock import patch

from geebeam._tiff_writer import run_tiff_export
from geebeam._wds_writer import run_webdataset_export
from geebeam._tfrecord_writer import run_tfrecord_export


class _FakeComputePatch(beam.DoFn):
    """A drop-in replacement for EEComputePatch that returns fake data."""

    def __init__(self, config, serialized_image, scale_x, scale_y, band_groups):
        self.config = config
        self.serialized_image = serialized_image
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.band_groups = band_groups

    def process(self, point):
        yield {
            'metadata': dict(point),
            'array': {'band1': np.ones((4, 4), dtype=np.float32)},
        }

def _make_input_records():
    records = [
        {'id': 0, 'x': 10.0, 'y': 20.0, 'split': 'train'},
        {'id': 1, 'x': 11.0, 'y': 21.0, 'split': 'train'},
        {'id': 2, 'x': 12.0, 'y': 22.0, 'split': 'validation'},
    ]
    splits = np.array(['train', 'validation'])
    return records, splits

@pytest.mark.integration
def test_tiff_pipeline_integration(tmp_path):
    input_records, splits = _make_input_records()
    output_path = str(tmp_path)

    config = {
        'project_id': 'test-project',
        'patch_size': 4,
        'scale': 10.0,
        'crs': 'EPSG:4326',
    }
    serialized_image = '{"type": "Image"}'
    band_groups = [['band1']]
    scale_x = 0.0001
    scale_y = -0.0001
    extra_metadata = {}
    md_feature_dict = {'id': 'int', 'x': 'float', 'y': 'float', 'split': 'str'}

    pipeline_options = PipelineOptions()

    with patch('geebeam._transforms.EEComputePatch', _FakeComputePatch):
        run_tiff_export(
            input_records=input_records,
            splits=splits,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            md_feature_dict=md_feature_dict,
            pipeline_options=pipeline_options,
        )

    train_dir = os.path.join(output_path, 'train')
    val_dir = os.path.join(output_path, 'validation')
    assert os.path.isdir(train_dir), f"train dir missing: {train_dir}"
    assert os.path.isdir(val_dir), f"validation dir missing: {val_dir}"

    train_tiffs = glob.glob(os.path.join(train_dir, '*.tif'))
    val_tiffs = glob.glob(os.path.join(val_dir, '*.tif'))
    assert len(train_tiffs) == 2, f"Expected 2 train tiffs, found {train_tiffs}"
    assert len(val_tiffs) == 1, f"Expected 1 val tiff, found {val_tiffs}"

    parquet_files = glob.glob(os.path.join(output_path, 'metadata*.parquet'))
    assert len(parquet_files) >= 1, f"Expected parquet files, found {parquet_files}"

@pytest.mark.integration
def test_webdataset_pipeline_integration(tmp_path):
    input_records, splits = _make_input_records()
    output_path = str(tmp_path)

    config = {
        'project_id': 'test-project',
        'patch_size': 4,
        'scale': 10.0,
        'crs': 'EPSG:4326',
    }
    serialized_image = '{"type": "Image"}'
    band_groups = [['band1']]
    scale_x = 0.0001
    scale_y = -0.0001
    extra_metadata = {}

    pipeline_options = PipelineOptions()

    with patch('geebeam._transforms.EEComputePatch', _FakeComputePatch):
        run_webdataset_export(
            input_records=input_records,
            splits=splits,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            pipeline_options=pipeline_options,
        )

    tar_files = glob.glob(os.path.join(output_path, '*.tar'))
    assert len(tar_files) >= 1, f"Expected .tar files in {output_path}, found {tar_files}"

@pytest.mark.integration
def test_tfrecord_pipeline_integration(tmp_path):
    pytest.importorskip('tensorflow_data_validation')
    pytest.importorskip('tfx_bsl')
    import tensorflow as tf

    input_records, splits = _make_input_records()
    output_path = str(tmp_path)

    config = {
        'project_id': 'test-project',
        'patch_size': 4,
        'scale': 10.0,
        'crs': 'EPSG:4326',
    }
    md_feature_dict = {'id': 'int', 'x': 'float', 'y': 'float', 'split': 'str'}

    with patch('geebeam._transforms.EEComputePatch', _FakeComputePatch):
        run_tfrecord_export(
            input_records=input_records,
            splits=splits,
            output_path=output_path,
            config=config,
            serialized_image='{"type": "Image"}',
            band_groups=[['band1']],
            scale_x=0.0001,
            scale_y=-0.0001,
            extra_metadata={},
            md_feature_dict=md_feature_dict,
            pipeline_options=PipelineOptions(),
        )

    # WriteToTFRecord uses the split name as a filename prefix, not a subdirectory:
    # e.g. {output_path}/train-00000-of-00001.tfrecord.gz
    train_records = glob.glob(os.path.join(output_path, 'train-*.tfrecord.gz'))
    val_records = glob.glob(os.path.join(output_path, 'validation-*.tfrecord.gz'))
    assert len(train_records) >= 1, f"Expected train TFRecords, found {train_records}"
    assert len(val_records) >= 1, f"Expected validation TFRecords, found {val_records}"

    # Sidecar files from TFDV stats computation
    assert os.path.exists(os.path.join(output_path, 'stats.tfrecord'))
    assert os.path.exists(os.path.join(output_path, 'schema.pbtxt'))

    # Verify train TFRecord content is valid and complete
    dataset = tf.data.TFRecordDataset(train_records, compression_type='GZIP')
    examples = list(dataset.as_numpy_iterator())
    assert len(examples) == 2  # 2 train records from _make_input_records

    example = tf.train.Example()
    example.ParseFromString(examples[0])
    features = example.features.feature
    assert 'md_id' in features
    assert 'md_split' in features
    assert 'im_band1' in features
    # 4x4 patch flattened → 16 values, each 1.0 from _FakeComputePatch
    band_values = list(features['im_band1'].float_list.value)
    assert len(band_values) == 16
    assert all(v == pytest.approx(1.0) for v in band_values)
