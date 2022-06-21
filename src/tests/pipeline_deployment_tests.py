# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test training pipeline using local runner."""

import sys
import os
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
import tensorflow as tf
import logging

from src.pipeline import config
from src.pipeline import pipeline


def test_e2e_pipeline():
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    pipeline_name = os.getenv("PIPELINE_NAME")
    gcs_location = config.GCS_BUCKET_NAME
    assert project, "Environment variable GOOGLE_CLOUD_PROJECT is None!"
    assert pipeline_name, "Environment variable PIPELINE_NAME is None!"

    # if tf.io.gfile.exists(gcs_location):
    #     tf.io.gfile.rmtree(gcs_location)
    # logging.info(f"Pipeline e2e tests artifacts stored in: {gcs_location}")

    pipeline_root = config.PIPELINE_ROOT

    runner = LocalDagRunner()
    _trainer_module_file = 'trainer.py'
    pipeline = pipeline._create_pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=config.PIPELINE_ROOT,
        data_root=config.DATA_ROOT,
        module_file=os.path.join(config.MODULE_ROOT, _trainer_module_file),
        endpoint_name=config.ENDPOINT_NAME,
        project_id=config.GOOGLE_CLOUD_PROJECT,
        region=config.GOOGLE_CLOUD_REGION,
        use_gpu=False,
        serving_model_dir=config.SERVING_MODEL_DIR)

    runner.run(pipeline)

    logging.info(f"Model output: {'gs://' + project + '/best_model'}")
    assert tf.io.gfile.exists('gs://' + project + '/best_model')
