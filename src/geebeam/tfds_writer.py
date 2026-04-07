"""Pipeline for tensorflow-dataset custom dataset"""

import tensorflow_datasets as tfds
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

from geebeam import transforms

class _GeebeamBuilderConfig(tfds.core.BuilderConfig):
    """Configuration object for builder."""
    def __init__(self, name, serialized_image, band_groups, all_bands,
                 input_records, crs, scale_x, scale_y, patch_size, validation_ratio,
                 project_id, extra_metadata,
                 version
                 ):
        self.name = name
        self.serialized_image = serialized_image
        self.band_groups = band_groups
        self.all_bands = all_bands
        self.project_id = project_id
        self.patch_size = patch_size
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.input_records = input_records
        self.validation_ratio = validation_ratio
        self.crs = crs
        self.extra_metadata = extra_metadata
        self.extra_metadata_keys = list(extra_metadata.keys())
        self.version = version
        self.supported_versions = [self.version]
        self.tags = []


class Geebeam(tfds.core.GeneratorBasedBuilder):
    """TFDS Builder for geebeam Earth Engine image chips dataset."""

    def _info(self):
        """Define dataset info."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=self._build_features(),
        )

    def _build_features(self):
        """Build feature dictionary from configuration."""
        config = self.builder_config
        patch_size = config.patch_size
        all_bands = config.all_bands
        extra_keys = config.extra_metadata_keys
        
        features = {
            'md_id': tfds.features.Scalar(dtype=tf.int64),
            'md_y': tfds.features.Scalar(dtype=tf.float32),
            'md_x': tfds.features.Scalar(dtype=tf.float32),
            'md_split': tfds.features.Text(),
        }
        
        # Add extra metadata features
        for key in extra_keys:
            features[f'md_{key}'] = tfds.features.Scalar(dtype=tf.float32)
        
        # Add image band features
        for band in all_bands:
            features[f'im_{band}'] = tfds.features.Tensor(
                shape=(patch_size * patch_size,), 
                dtype=tf.float32
            )
        
        return tfds.features.FeaturesDict(features)

    def _split_generators(self, dl_manager):
        """Define splits based on training/validation ratio."""
        config = self.builder_config
        
        # Get input data from config
        input_records = config.input_records
        serialized_image = config.serialized_image
        band_groups = config.band_groups
        scale_x = config.scale_x
        scale_y = config.scale_y
        
        training_data = self._generate_examples(
                    input_records=input_records,
                    serialized_image=serialized_image,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    config=config,
                    band_groups=band_groups,
                    split='train'
                )
        # Return examples, generating validation examples if necessary
        if config.validation_ratio > 0:
            return {
                'train': training_data,
                'validation': self._generate_examples(
                    input_records=input_records,
                    serialized_image=serialized_image,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    config=config,
                    band_groups=band_groups,
                    split='val'
                )
            }
        else:
            return {
                'train': training_data
            }


    def _generate_examples(self, input_records, serialized_image, scale_x, scale_y, 
                          config, band_groups, split):
        """Generate examples using Beam pipeline."""
        beam = tfds.core.lazy_imports.apache_beam
        # Create EE compute patch processor
        ee_config = {
            'project_id': config.project_id,
            'patch_size': config.patch_size,
            'crs': config.crs
        }
        processor = transforms.EEComputePatch(
            ee_config,
            serialized_image,
            scale_x,
            scale_y,
            band_groups
        )
        processor.setup()

        def _filter_by_split(record):
            """Filter records by split."""
            return record.get('split') == split
        
        def _process_and_convert(record):
            """Process a record and convert to TFDS example format."""
            
            # Process the record
            try:
                results = list(processor.process(record))
                if results:
                    out_dict = results[0]
                    
                    # Add extra metadata
                    merged_metadata = {
                        **out_dict.get("metadata", {}),
                        **config.extra_metadata
                    }
                    
                    # Convert dict to TFDS example format
                    example = {}
                    
                    # Add metadata
                    example['md_id'] = merged_metadata['id']
                    example['md_y'] = merged_metadata['y']
                    example['md_x'] = merged_metadata['x']
                    example['md_split'] = merged_metadata['split']
                    
                    # Add extra metadata
                    for key in config.extra_metadata.keys():
                        if key in merged_metadata:
                            example[f'md_{key}'] = merged_metadata[key]
                    
                    # Add image bands
                    for band_name, band_data in out_dict['array'].items():
                        example[f'im_{band_name}'] = band_data.flatten().astype('float32')
                    
                    return record['id'], example
            except Exception as e:
                import logging
                logging.error(f"Error processing record {record['id']}: {e}")
                raise
        
        return (
            beam.Create(input_records)
            | f'Filter {split}' >> beam.Filter(_filter_by_split) 
            | f'Reshuffle {split}' >> beam.Reshuffle()
            | f'Process {split}' >> beam.Map(_process_and_convert)
        )

def run_tfds_export(
    input_records: list[dict],
    output_path: str,
    config: dict,
    serialized_image,
    band_groups: list,
    all_bands: list,
    scale_x: float,
    scale_y: float,
    extra_metadata: dict,
    pipeline_options: PipelineOptions,
    dataset_name: str,
    dataset_version: str,
    ):

    builder_config = _GeebeamBuilderConfig(
        name=dataset_name,
        version=dataset_version,
        input_records=input_records,
        serialized_image=serialized_image,
        band_groups=band_groups,
        all_bands=all_bands,
        crs=config['crs'],
        scale_x=scale_x,
        scale_y=scale_y,
        patch_size=config['patch_size'],
        validation_ratio=config['validation_ratio'],
        project_id=config['project_id'],
        extra_metadata=extra_metadata
    )

    # Create builder
    builder = Geebeam(
        data_dir=output_path,
        config=builder_config,
    )

    # Set up download configuration with Beam options
    dl_config = tfds.download.DownloadConfig(beam_options=pipeline_options)

    # Download and prepare the dataset using TFDS
    builder.download_and_prepare(download_config=dl_config)

