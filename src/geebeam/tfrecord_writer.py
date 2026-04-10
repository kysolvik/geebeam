"""Pipeline for writing to raw tfrecords (with statistics and schema sidecar files)"""

import os
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from geebeam import transforms, tf_utils

def run_tfrecord_export(
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
    import tensorflow_data_validation as tfdv
    from tfx_bsl.coders import example_coder
    with beam.Pipeline(options=pipeline_options) as pipeline:

        points = pipeline | 'Create points' >> beam.Create(input_records)

        all_data = (
            points
            | 'Get patch' >> beam.ParDo(transforms.EEComputePatch(
                config,
                serialized_image,
                scale_x,
                scale_y,
                band_groups
                ))
            | 'Add metadata' >> beam.ParDo(transforms.AddMetadata(extra_metadata))
        )

        # Write first split and calculate stats from it
        split = splits[0]
        output_dir = os.path.join(output_path, split)
        train_data = (
            all_data
            | f'Filter {split}' >> beam.Filter(lambda record: record['metadata']['split'] == split)
            | f'{split} to tf.Example' >> beam.Map(tf_utils.dict_to_example)
        )

        # Write
        _ = (train_data
            | f'Write {split}' >> tf_utils.WriteTFExample(output_dir)
        )

        # Calculate stats on training data
        decoder = example_coder.ExamplesToRecordBatchDecoder()
        stats = (
            train_data
            | 'Batch' >> beam.BatchElements(
                min_batch_size=10,
                max_batch_size=100)
            | 'Decode to arrow' >> beam.Map(lambda b: decoder.DecodeBatch(b))
            | 'Generate Statistics' >> tfdv.GenerateStatistics()
        )
        stats | 'Write stats' >> tfdv.WriteStatisticsToTFRecord(
            os.path.join(output_path, 'stats.tfrecord'))

        # Write out other splits
        if len(splits) > 1:
            for split in splits[1:]:
                output_dir = os.path.join(output_path, split)
                _ = (
                    all_data
                    | f'Filter {split}' >> beam.Filter(lambda record: record['metadata']['split'] == split)
                    | f'{split} to tf.Example' >> beam.Map(tf_utils.dict_to_example)
                    | f'Write {split}' >> tf_utils.WriteTFExample(output_dir)
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