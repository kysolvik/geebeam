"""Execute GEE tile extraction in Beam + Dataflow

Example dataflow execution:
python examples/geebeam_run.py \
    --region us-east1 \
    --worker_zone us-east1-b \
    --network='default' \
    --subnetwork='regions/us-east1/subnetworks/default' \
    --sampling_region ../data/Limites_RAISG_2025/Lim_Raisg.shp \
    --output_path gs://aic-fire-amazon/results_2024_test/ \
    --runner DataflowRunner \
    --max_num_workers=2 \
    --num_workers=1 \
    --experiments=use_runner_v2 \
    --machine_type=n2-standard-2\
    --setup_file='./setup.py'

Example local execution:
    python geebeam_run.py \
        --output_path './results'
        --sampling_region ./data/Limites_RAISG_2025/Lim_Raisg.shp \
        --runner DirectRunner
"""

import os
os.environ['GRPC_ARG_KEEPALIVE_TIME_MS'] = '30000'
os.environ['GRPC_ARG_KEEPALIVE_TIMEOUT_MS'] = '10000'
os.environ['GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS'] = '1'
os.environ['GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA'] = '0'
os.environ['no_proxy'] = 'google.com,googleapis.com'

import logging

import ee
import geebeam

RANDOM_SEED = 54
# Define your configuration
config = {
  "project_id": "ksolvik-misc",
  "target_year": 2024,
  "patch_size": 128,
  "scale": 500,
  "proj": "EPSG:4326",
  "n_sample": 100,
  "validation_ratio": 0.2,
}

ee.Initialize(project='ksolvik-misc')

# Define your images for sampling
embeddings_proj = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').first().projection()
embeddings = (
            ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
            .filter(ee.Filter.calendarRange(config['target_year']-1,
                                            config['target_year']-1,
                                            'year'))
            .mosaic()
            # Special step for embeddings: after mosaic, they don't have proj info
           .setDefaultProjection(embeddings_proj).resample('bicubic')
            )
mcd64 = (ee.ImageCollection('MODIS/061/MCD64A1')
    .select('BurnDate')
    .filter(ee.Filter.calendarRange(config['target_year'], config['target_year'], 'year'))
    .min()
    )
mb_burned_area = (
    ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_v1')
    .reduceResolution('mean', maxPixels=500)
    )
mb_deforestation = (
    ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_deforestation_secondary_vegetation_v2')
   .reduceResolution('mode', maxPixels=500)
)

im_list = [embeddings, mcd64, mb_burned_area, mb_deforestation]



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # Execute
    geebeam.runner.run(config=config, image_list=im_list, random_seed=RANDOM_SEED)
