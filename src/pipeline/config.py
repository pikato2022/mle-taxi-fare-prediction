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
"""TFX pipeline configurations."""


PIPELINE_NAME = 'chicago-vertex-pipelines'
GOOGLE_CLOUD_PROJECT = 'mle-chicago-taxi-trip'     # <--- ENTER THIS
GOOGLE_CLOUD_REGION = 'us-central1'
# GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-gcs'
GCS_BUCKET_NAME = GOOGLE_CLOUD_PROJECT
# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME
print(ENDPOINT_NAME)
# Path to various pipeline artifact.
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(
    GCS_BUCKET_NAME, PIPELINE_NAME)
print(PIPELINE_ROOT)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(
    GCS_BUCKET_NAME, PIPELINE_NAME)
print(MODULE_ROOT)

# Paths for input data.
DATA_ROOT = 'gs://{}/datas/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)
print(DATA_ROOT)

# This is the path where your model will be pushed for serving.
SERVING_MODEL_DIR = 'gs://{}/serving_model/{}'.format(
    GCS_BUCKET_NAME, PIPELINE_NAME)
print(SERVING_MODEL_DIR)
