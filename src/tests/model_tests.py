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

from src.pipeline import trainer_tune

EXPECTED_HYPERPARAMS_KEYS = [
    "learning_rate",
    "n_layers",
    "n_units_1",
]


def test_hyperparams_defaults():
    hyperparams = trainer_tune._get_hyperparameters()
    assert set(hyperparams.keys()) == set(EXPECTED_HYPERPARAMS_KEYS)
