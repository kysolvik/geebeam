import io
import json
import numpy as np
import rasterio
from geebeam._wds_writer import _create_tiff_bytes, ProcessToWebDataset

def test_create_tiff_bytes():
    array_dict = {
        'band1': np.ones((4, 4), dtype=np.float32),
    }
    metadata = {'id': 0, 'x': 10.0, 'y': 20.0, 'split': 'train'}
    crs = 'EPSG:4326'
    scale_x = 0.0001
    scale_y = -0.0001

    result = _create_tiff_bytes(array_dict, metadata, crs, scale_x, scale_y)
    assert isinstance(result, bytes)
    assert len(result) > 0

    # Verify bytes are a readable GeoTIFF
    with rasterio.open(io.BytesIO(result)) as ds:
        assert ds.count == 1
        assert ds.width == 4
        assert ds.height == 4

def test_process_to_webdataset():
    crs = 'EPSG:4326'
    scale_x = 0.0001
    scale_y = -0.0001

    dofn = ProcessToWebDataset(crs=crs, scale_x=scale_x, scale_y=scale_y)

    array_dict = {
        'band1': np.ones((4, 4), dtype=np.float32),
    }
    element = {
        'metadata': {'id': 3, 'x': 10.0, 'y': 20.0, 'split': 'train'},
        'array': array_dict,
    }

    results = list(dofn.process(element))
    assert len(results) == 1
    record = results[0]

    assert '__key__' in record
    assert 'tif' in record
    assert 'json' in record

    assert record['__key__'] == '00003'
    assert isinstance(record['tif'], bytes)
    assert isinstance(record['json'], bytes)

    # Verify JSON is valid
    md = json.loads(record['json'].decode('utf-8'))
    assert md['id'] == 3
