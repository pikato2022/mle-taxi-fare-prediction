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
"""Test model functions."""

import sys
import logging
import tensorflow as tf

from src.common import features
from src.pipeline import trainer_tune

root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)
EXPECTED_HYPERPARAMS_KEYS = [
    "learning_rate",
    "n_layers",
    "n_units_1",
]


def test_hyperparams_defaults():
    hyperparams = trainer_tune._get_hyperparameters()
    assert set(hyperparams.keys()) == set(EXPECTED_HYPERPARAMS_KEYS)

def test_create_binary_classifier():
    pass
    hyperparams = trainer_tune._get_hyperparameters()

    # model_inputs = {
    #     "dropoff_grid_xf": tf.convert_to_tensor([0, 0, 0]),
    #     "euclidean_xf": tf.convert_to_tensor([-0.9066112, -0.9066112, -0.9066112]),
    #     "loc_cross_xf": tf.convert_to_tensor([0, 0, 0]),
    #     "payment_type_xf": tf.convert_to_tensor([1, 0, 0]),
    #     "pickup_grid_xf": tf.convert_to_tensor([0, 0, 0]),
    #     "trip_day_of_week_xf": tf.convert_to_tensor([5, 4, 4]),
    #     "trip_day_xf": tf.convert_to_tensor([26, 24, 1]),
    #     "trip_hour_xf": tf.convert_to_tensor([0, 4, 2]),
    #     "trip_miles_xf": tf.convert_to_tensor([5.9717827, -0.7121308, -0.7601589]),
    #     "trip_month_xf": tf.convert_to_tensor([4, 3, 4]),
    #     "trip_seconds_xf": tf.convert_to_tensor([4.9029775, -0.34146854, -0.34479955]),
    # }
    #
    # feature_vocab_sizes = {
    #     feature_name: 100 for feature_name in features.categorical_feature_names()
    # }
    # models = trainer_tune._make_keras_model(hyperparams)
    # model_outputs = models(model_inputs)  # .numpy()
    # assert model_outputs.shape == (3, 1)
    # assert model_outputs.dtype == "float32"
