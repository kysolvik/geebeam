"""Pipeline for writing image chips as Cloud-Optimized GeoTIFFs and metadata as Parquet."""

import os
import tempfile
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.filesystems import FileSystems
import rasterio
from rasterio.transform import Affine
import pyarrow as pa

from geebeam import transforms

def _build_tiff_name(id, min_digits=5):
    return f"{str(id).zfill(min_digits)}.tif"

class WriteTiff(beam.DoFn):
    """DoFn to write image arrays to Cloud-Optimized GeoTIFFs."""
    def __init__(self, output_path, crs, scale_x, scale_y):
        self.output_path = output_path
        self.crs = crs
        self.scale_x = scale_x
        self.scale_y = scale_y

    def setup(self):
        # Ensure the output directory for TIFFs exists
        if not FileSystems.exists(self.output_path):
            try:
                FileSystems.mkdirs(self.output_path)
            except Exception as e:
                logging.warning(f"Error creating directory {self.output_path}: {e}")

    def process(self, element):
        metadata = element['metadata']
        array_dict = element['array']
        
        id_val = metadata['id']
        tiff_name = _build_tiff_name(id_val)
        tiff_path = os.path.join(self.output_path, tiff_name)
        
        # Get dimensions and metadata for TIFF
        first_band = next(iter(array_dict.values()))
        height, width = first_band.shape
        count = len(array_dict)
        dtype = first_band.dtype
        
        # Construct affine transform
        transform = Affine(
            self.scale_x, 0, metadata['x'],
            0, self.scale_y, metadata['y']
        )
        
        # Write to a local temporary file first
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=True) as tmp:
            with rasterio.open(
                tmp.name,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=count,
                dtype=dtype,
                crs=self.crs,
                transform=transform,
                tiled=True,
                compress='lzw'
            ) as dst:
                for i, (band_name, data) in enumerate(array_dict.items(), 1):
                    dst.write(data, i)
                    dst.set_band_description(i, band_name)
            
            # Upload the temporary file to the final destination
            with FileSystems.create(tiff_path) as dest:
                with open(tmp.name, 'rb') as src:
                    dest.write(src.read())


class ProcessMetadataToParquet(beam.DoFn):
    """DoFn to write image arrays to Cloud-Optimized GeoTIFFs."""
    def __init__(self, output_path):
        self.output_path = output_path

    def process(self, element):
        metadata = element['metadata']
        id_val = metadata['id']
        tiff_name = _build_tiff_name(id_val)
        split = metadata['split']
        tiff_dir = os.path.join(self.output_path, split)
        tiff_path = os.path.join(tiff_dir, tiff_name)
        # Prepare metadata for Parquet output
        parquet_row = {**metadata}
        parquet_row['image_path'] = tiff_path
        parquet_row['image_name'] = tiff_name
        
        # Convert all metadata values to string or numeric to ensure Parquet compatibility
        for key, value in parquet_row.items():
            if hasattr(value, 'tolist'): # Handle numpy arrays if any
                parquet_row[key] = str(value.tolist())
        
        yield parquet_row


def run_tiff_export(
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
    """Run Beam pipeline to export TIFFs and a Parquet metadata file."""
    
    # Define a simple schema for Parquet if possible, or let it be inferred.
    # Beam's WriteToParquet requires a schema.
    
    # We'll infer a basic schema based on the first input record and extra_metadata.
    # This is a bit tricky because some fields are added during processing.
    # For now, we'll try to use a flexible approach or a predefined one for common fields.
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        points = pipeline | 'Create points' >> beam.Create(input_records)

        # Get patches
        all_data = (
            points
            | 'Get patch' >> beam.ParDo(transforms.EEComputePatch(
                config,
                serialized_image,
                scale_x,
                scale_y,
                band_groups
                ))
        )

        # Split data
        for split in splits:
            output_dir = os.path.join(output_path, split)
            (all_data
                | f'Filter {split}' >> beam.Filter(lambda record: record['metadata']['split'] == split)
                | f'Write {split} TIFFs' >> beam.ParDo(WriteTiff(
                    output_path=os.path.join(output_dir),
                    crs=config['crs'],
                    scale_x=scale_x,
                    scale_y=scale_y
                ))
            )
        
        # Write metadata to parquet
        # To use WriteToParquet, we need a pyarrow schema.
        # Define schema based on known fields and extra_metadata keys
        fields = [
            ('id', pa.int64()),
            ('x', pa.float64()),
            ('y', pa.float64()),
            ('split', pa.string()),
            ('image_path', pa.string()),
            ('image_name', pa.string()),
        ]
        
        for key in extra_metadata.keys():
            # Basic type inference for extra metadata
            val = extra_metadata[key]
            if isinstance(val, int):
                fields.append((key, pa.int64()))
            elif isinstance(val, float):
                fields.append((key, pa.float64()))
            else:
                fields.append((key, pa.string()))
        
        schema = pa.schema(fields)

        (all_data 
         | 'Add Metadata' >> beam.ParDo(transforms.AddMetadata(extra_metadata))
         | 'Clean Metadata' >> beam.ParDo(ProcessMetadataToParquet(output_path))
         | 'Write Metadata' >> beam.io.parquetio.WriteToParquet(
             file_path_prefix=os.path.join(output_path, 'metadata'),
             schema=schema,
             file_name_suffix='.parquet'
         )
        )
