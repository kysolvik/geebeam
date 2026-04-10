"""Pipeline for writing to WebDataset format (.tar files containing .tif and .json)."""

import os
import json
import webdataset as wds
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from rasterio.transform import Affine
from rasterio.io import MemoryFile
import uuid

from geebeam import _transforms

def _create_tiff_bytes(array_dict, metadata, crs, scale_x, scale_y):
    """Create TIFF bytes from array dict and metadata."""
    first_band = next(iter(array_dict.values()))
    height, width = first_band.shape
    count = len(array_dict)
    dtype = first_band.dtype
    
    transform = Affine(
        scale_x, 0, metadata['x'],
        0, scale_y, metadata['y']
    )
    
    with MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=count,
            dtype=dtype,
            crs=crs,
            transform=transform,
            tiled=True,
            compress='lzw'
        ) as dst:
            for i, (band_name, data) in enumerate(array_dict.items(), 1):
                dst.write(data, i)
                dst.set_band_description(i, band_name)
        return memfile.read()

class ProcessToWebDataset(beam.DoFn):
    """DoFn to prepare records for WebDataset output."""
    def __init__(self, crs, scale_x, scale_y):
        self.crs = crs
        self.scale_x = scale_x
        self.scale_y = scale_y

    def process(self, element):
        metadata = element['metadata']
        array_dict = element['array']
        basename = str(metadata['id']).zfill(5)
        
        tif_bytes = _create_tiff_bytes(array_dict, metadata, self.crs, self.scale_x, self.scale_y)
        
        json_bytes = json.dumps(metadata).encode('utf-8')
        
        yield {'__key__': basename,
               'tif': tif_bytes,
               'json':json_bytes
        }

class WriteToWebDataset(beam.DoFn):
    def __init__(self, output_path, split):
        worker_id = str(uuid.uuid4())[:8]
        self.out_pattern = f'{os.path.join(output_path, split)}-{worker_id}-%06d.tar'
        self.writer = None

    def start_bundle(self):
        self.writer = wds.ShardWriter(self.out_pattern)

    def process(self, element):
        self.writer.write(element)

    def finish_bundle(self):
        if self.writer:
            self.writer.close()

def run_webdataset_export(
    input_records: list[dict],
    splits: list[str],
    output_path: str,
    config: dict,
    serialized_image,
    band_groups: list,
    scale_x: float,
    scale_y: float,
    extra_metadata: dict,
    pipeline_options: PipelineOptions
    ):
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        points = pipeline | 'Create points' >> beam.Create(input_records)

        # Get patches and add metadata
        all_data = (
            points
            | 'Get patch' >> beam.ParDo(_transforms.EEComputePatch(
                config,
                serialized_image,
                scale_x,
                scale_y,
                band_groups
                ))
            | 'Add Metadata' >> beam.ParDo(_transforms.AddMetadata(extra_metadata))
        )

        for split in splits:
            
            (all_data
                | f'Filter {split}' >> beam.Filter(lambda record, s=split: record['metadata']['split'] == s)
                | f'Reshuffle {split}' >> beam.Reshuffle()
                | f'Format {split}' >> beam.ParDo(ProcessToWebDataset(
                    crs=config['crs'],
                    scale_x=scale_x,
                    scale_y=scale_y
                    ))
                | f'Write {split}' >> beam.ParDo(WriteToWebDataset(output_path, split))
            )
