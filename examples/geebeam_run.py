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
  "n_sample": 5000,
  "validation_ratio": 0.2,
}

ee.Initialize(project='ksolvik-misc')

# Define images for sampling
# MODIS MCD64
def prep_mcd64_year(y):
    mcd64 = (ee.ImageCollection('MODIS/061/MCD64A1')
             .select('BurnDate')
             .filter(ee.Filter.calendarRange(y, y, 'year'))
             .min()
             )
    band_names = mcd64.bandNames().getInfo()
    band_names_new = ['{}_{}'.format(b, y) for b in band_names]
    mcd64 = mcd64.rename(band_names_new)
    return mcd64
mcd64_list = [prep_mcd64_year(y) for y in range(2001, 2026)]

# MB Land-use/land-cover
mb_amz_lulc = (
    ee.Image('projects/mapbiomas-public/assets/amazon/lulc/collection6/mapbiomas_collection60_integration_v1')
    .lt(10)
   .reduceResolution('mean', maxPixels=500)
)
mb_amz_lulc_list = [mb_amz_lulc]

# MODIS MOD13
def prep_mod13_year(y):
    mod13 = (
        ee.ImageCollection('MODIS/061/MOD13A1')
        .select(['NDVI', 'EVI'])
        .filter(ee.Filter.calendarRange(y,
                                        y,
                                        'year'))
        .max()
    )

    band_names = mod13.bandNames().getInfo()
    band_names_new = ['{}_{}'.format(b, y) for b in band_names]
    mod13 = mod13.rename(band_names_new)
    return mod13
mod13_list = [prep_mod13_year(y) for y in range(2001, 2026)]

def prep_mod13_begyear(y):
    mod13 = (
        ee.ImageCollection('MODIS/061/MOD13A1')
        .select(['NDVI', 'EVI', 'sur_refl_b01', 'sur_refl_b02',
                 'sur_refl_b03','sur_refl_b07'])
        .filter(ee.Filter.date('{}-01-01'.format(y+1),
                               '{}-04-30'.format(y+1)))
        .max()
    )

    band_names = mod13.bandNames().getInfo()
    band_names_new = ['{}_begyear_{}'.format(b, y) for b in band_names]
    mod13 = mod13.rename(band_names_new)
    return mod13
mod13_begyear_list = [prep_mod13_begyear(y) for y in range(2001, 2026)]


# Define your images for sampling
def prep_mcwd_year(y):
    mcwd = (
        ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE')
        .select('def')
        .filter(ee.Filter.calendarRange(y,
                                        y,
                                        'year'))
        .max()
    )
    mcwd = mcwd.rename('mcwd_{}'.format(y))
    return mcwd
mcwd_list = [prep_mcwd_year(y) for y in range(2001, 2026)]

# Embeddings
ee_proj = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').first().projection()
def prep_embeddings_year(y):
    embeddings = (
                ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filter(ee.Filter.calendarRange(y,y,
                                                'year'))
                .mosaic()
                # Special step for embeddings: after mosaic, they don't have proj info
                .setDefaultProjection(ee_proj)
                .resample('bicubic')
                )
    band_names = embeddings.bandNames().getInfo()
    band_names_new = ['{}_{}'.format(b, y) for b in band_names]
    embeddings = embeddings.rename(band_names_new)
    return embeddings


embeddings_list = [prep_embeddings_year(2023)]

im_list = (mcd64_list + mb_amz_lulc_list + mod13_list +
           mod13_begyear_list + mcwd_list)
print(im_list)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    # Execute
    geebeam.runner.run(config=config, image_list=im_list, random_seed=RANDOM_SEED)
