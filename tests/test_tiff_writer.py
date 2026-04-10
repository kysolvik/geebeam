import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import rasterio
import pandas as pd
from apache_beam.options.pipeline_options import PipelineOptions
from geebeam import pipeline

class TestTiffWriter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.project_id = 'test-project'
        self.output_path = os.path.join(self.test_dir, 'output')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('ee.Initialize')
    @patch('ee.Projection')
    @patch('geebeam._ee_utils.build_prepped_image')
    @patch('geebeam._ee_utils._serialize')
    @patch('geebeam.sampler.sample_region_random')
    @patch('geebeam.sampler._get_roi')
    @patch('geebeam._ee_utils.get_pixels_allbands')
    @patch('ee.deserializer.fromJSON')
    def test_run_pipeline_tif(self, mock_from_json, mock_get_pixels, mock_get_roi, 
                             mock_sample_points, mock_serialize, mock_build_image, 
                             mock_projection, mock_init):
        
        # Setup mocks
        mock_projection.return_value.atScale.return_value.getInfo.return_value = {
            'transform': [10.0, 0.0, 0.0, 0.0, -10.0, 0.0]
        }
        
        mock_build_image.return_value = (MagicMock(), [['B1', 'B2']], ['B1', 'B2'])
        mock_serialize.return_value = 'serialized_image'
        
        # Mock sample points
        mock_sample_points.return_value = pd.DataFrame({
            'id': [1, 2],
            'x': [100.0, 200.0],
            'y': [100.0, 200.0],
            'split': ['train', 'validation']
        })
        
        # Mock pixel data
        # EEComputePatch calls _join_struct_arrays_to_dict which expects a list of structured arrays
        # Each structured array from get_pixels should have bands as fields
        data1 = np.zeros((1,), dtype=[('B1', 'f4', (16, 16)), ('B2', 'f4', (16, 16))])
        data1['B1'] = np.random.rand(16, 16)
        data1['B2'] = np.random.rand(16, 16)
        
        mock_get_pixels.side_effect = [data1, data1]

        # Run pipeline
        pipeline.sample_and_run_pipeline(
            image_list=['test_image'],
            output_path=self.output_path,
            project=self.project_id,
            patch_size=16,
            scale=10.0,
            n_sample=2,
            output_type='tif',
            sampling_region=MagicMock(),
            beam_options={'runner': 'DirectRunner'},
        )

        # Check if outputs exist
        tiff_dir = os.path.join(self.output_path, 'tiffs')
        self.assertTrue(os.path.exists(tiff_dir))
        self.assertTrue(os.path.exists(os.path.join(tiff_dir, '1.tif')))
        self.assertTrue(os.path.exists(os.path.join(tiff_dir, '2.tif')))
        
        # Verify TIFF content
        with rasterio.open(os.path.join(tiff_dir, '1.tif')) as src:
            self.assertEqual(src.count, 2)
            self.assertEqual(src.width, 16)
            self.assertEqual(src.height, 16)
            self.assertEqual(src.descriptions, ('B1', 'B2'))
            # Check transform (scale_x=10, scale_y=-10, x=100, y=100)
            self.assertEqual(src.transform[0], 10.0)
            self.assertEqual(src.transform[4], -10.0)
            self.assertEqual(src.transform[2], 100.0)
            self.assertEqual(src.transform[5], 100.0)

        # Verify Parquet metadata
        # Beam writes multiple parquet files, usually metadata-00000-of-00001.parquet
        parquet_files = [f for f in os.listdir(self.output_path) if f.endswith('.parquet')]
        self.assertTrue(len(parquet_files) > 0)
        
        df = pd.read_parquet(os.path.join(self.output_path, parquet_files[0]))
        self.assertIn('id', df.columns)
        self.assertIn('image_path', df.columns)
        self.assertEqual(len(df), 2)
        self.assertTrue(all(df['image_path'].str.contains('.tif')))

if __name__ == '__main__':
    unittest.main()
