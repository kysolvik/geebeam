"""Execute GEE tile extraction in Beam + Dataflow

Example execution:
    python geebeam_run.py \
        --config ./example_config.json \
        --output_path gs://aic-fire-amazon/results/ \
        --region_of_interest ./data/Limites_RAISG_2025/Lim_Raisg.shp \
        --runner DataflowRunner \
        --experiments=use_runner_v2 \
        --max_num_workers=16 \
        --num_workers=8 \
        --requirements_file ./pipeline_requirements.txt
"""

import logging

from .geebeam import geebeam

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    geebeam.run()
