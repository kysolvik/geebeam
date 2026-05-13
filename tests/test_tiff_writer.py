import os
import pytest
import numpy as np
import rasterio
from geebeam._tiff_writer import _build_tiff_name, WriteTiff, ProcessMetadataToParquet


def test_build_tiff_name():
    assert _build_tiff_name(42) == '00042.tif'
    assert _build_tiff_name(0) == '00000.tif'

def test_write_tiff_process(tmp_path):
    output_dir = str(tmp_path)
    crs = 'EPSG:4326'
    scale_x = 0.0001
    scale_y = -0.0001

    writer = WriteTiff(output_path=output_dir, crs=crs, scale_x=scale_x, scale_y=scale_y)
    writer.setup()

    array_dict = {
        'band1': np.ones((4, 4), dtype=np.float32),
        'band2': np.zeros((4, 4), dtype=np.float32),
    }
    element = {
        'metadata': {'id': 1, 'x': 10.0, 'y': 20.0, 'split': 'train'},
        'array': array_dict,
    }

    writer.process(element)

    tiff_path = os.path.join(output_dir, '00001.tif')
    assert os.path.exists(tiff_path)

    with rasterio.open(tiff_path) as ds:
        assert ds.count == 2
        assert ds.width == 4
        assert ds.height == 4

def test_process_metadata_to_parquet(tmp_path):
    output_dir = str(tmp_path)
    dofn = ProcessMetadataToParquet(output_path=output_dir)

    element = {
        'metadata': {'id': 5, 'x': 1.0, 'y': 2.0, 'split': 'train'},
        'array': {},
    }

    results = list(dofn.process(element))
    assert len(results) == 1
    row = results[0]
    assert 'image_path' in row
    assert 'image_name' in row
    assert row['image_name'] == '00005.tif'
