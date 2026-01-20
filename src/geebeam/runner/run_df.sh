python geebeam_main.py \
    --config ./example_config.json \
    --region_of_interest ../data/Limites_RAISG_2025/Lim_Raisg.shp \
    --output_path gs://aic-fire-amazon/results_2022_5k/ \
    --runner DataflowRunner \
    --max_num_workers=24 \
    --min_num_workers=4 \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-east1-docker.pkg.dev/ksolvik-misc/columbia-aic-risk-modeling/fire-risk-preprocess/beam_python_prebuilt_sdk:546924a5-4b91-466c-896f-2299c0106eb1


# Uncomment to build container image for faster deployment
# Pushes to Google Archive Registry
#     --prebuild_sdk_container_engine=local_docker \
#     --docker_registry_push_url=us-east1-docker.pkg.dev/ksolvik-misc/columbia-aic-risk-modeling/fire-risk-preprocess \
#     --requirements_file ./pipeline_requirements.txt
