"""Writer for raw tfrecords (with statistics and schema sidecar files)"""

import os
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow_data_validation as tfdv
from tfx_bsl.coders import example_coder

from geebeam import transforms

def run_tfrecord_export(
    input_records: list[dict],
    output_path: str,
    config: dict,
    serialized_image,
    band_groups: list,
    scale_x: float,
    scale_y: float,
    extra_metadata: dict,
    is_local: bool,
    pipeline_options: PipelineOptions

):
    with beam.Pipeline(options=pipeline_options) as pipeline:

        points = pipeline | 'Create points' >> beam.Create(input_records)

        if is_local:
            batches = (
                points
                | 'Add Dummy Key' >> beam.Map(lambda x: (None, x))
                | 'Reshuffle' >> beam.Reshuffle()
                | 'Force Single Batches' >> beam.GroupIntoBatches(batch_size=1)
                | 'Extract' >> beam.FlatMap(lambda x: x[1])
            )
        else:
            batches = points

        training_data, validation_data = (
            batches
            | 'Get patch' >> beam.ParDo(transforms.EEComputePatch(
                config,
                serialized_image,
                scale_x,
                scale_y,
                band_groups
                ))
            | 'Add metadata' >> beam.ParDo(transforms.AddMetadata(extra_metadata))
            | 'Split dataset' >> beam.Partition(transforms.split_dataset, 2)
        )

        # Convert to TF examples
        training_examples = (
            training_data
            | 'Train to tf.Example' >> beam.Map(transforms.dict_to_example)
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
            os.path.join(output_path, 'stats.tfrecord'))

        # Write out examples
        (training_examples
         | 'Write training' >> transforms.WriteTFExample(
             os.path.join(output_path, 'training'))
        )
        if config['validation_ratio'] > 0:
            validation_examples = (
                validation_data
                | 'Val to tf.Example' >> beam.Map(transforms.dict_to_example)
            )

            (validation_examples
            | 'Write validation' >> transforms.WriteTFExample(
                os.path.join(output_path, 'validation'))
            )


    # Infer schema and write as separate pbtxt
    stats = tfdv.load_statistics(
        os.path.join(output_path, 'stats.tfrecord')
    )

    schema = tfdv.infer_schema(stats)

    tfdv.write_schema_text(
        schema,
        os.path.join(output_path, 'schema.pbtxt')
    )