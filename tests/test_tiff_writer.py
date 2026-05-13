import os
import pytest
import numpy as np
import rasterio
from geebeam._tiff_writer import _build_tiff_name, WriteTiff, ProcessMetadataToParquet


def test_build_tiff_name():
    assert _build_tiff_name(42) == '00042.tif'
    assert _build_tiff_name(0) == '00000.tif'

def test_write_tiff_process(tmp_path):
    """WriteTiff uses x_topleft/y_topleft for the geotransform when present."""
    output_dir = str(tmp_path)
    scale_x = 0.0001
    scale_y = -0.0001

    writer = WriteTiff(output_path=output_dir, crs='EPSG:4326', scale_x=scale_x, scale_y=scale_y)
    writer.setup()

    array_dict = {
        'band1': np.ones((4, 4), dtype=np.float32),
        'band2': np.zeros((4, 4), dtype=np.float32),
    }
    element = {
        'metadata': {'id': 1, 'x': 10.0, 'y': 20.0, 'x_topleft': 5.0, 'y_topleft': 15.0, 'split': 'train'},
        'array': array_dict,
    }

    writer.process(element)

    tiff_path = os.path.join(output_dir, '00001.tif')
    assert os.path.exists(tiff_path)

    with rasterio.open(tiff_path) as ds:
        assert ds.count == 2
        assert ds.width == 4
        assert ds.height == 4
        assert ds.transform.c == pytest.approx(5.0)   # translateX = x_topleft
        assert ds.transform.f == pytest.approx(15.0)  # translateY = y_topleft

def test_write_tiff_process_fallback_to_xy(tmp_path):
    """WriteTiff falls back to x/y for the geotransform when x_topleft/y_topleft are absent."""
    output_dir = str(tmp_path)
    writer = WriteTiff(output_path=output_dir, crs='EPSG:4326', scale_x=0.0001, scale_y=-0.0001)
    writer.setup()

    element = {
        'metadata': {'id': 2, 'x': 10.0, 'y': 20.0, 'split': 'train'},
        'array': {'band1': np.ones((4, 4), dtype=np.float32)},
    }
    writer.process(element)

    with rasterio.open(os.path.join(output_dir, '00002.tif')) as ds:
        assert ds.transform.c == pytest.approx(10.0)
        assert ds.transform.f == pytest.approx(20.0)

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
