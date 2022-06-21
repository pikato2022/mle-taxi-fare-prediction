# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple

from typing import List
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft
import os 

from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform import TFTransformOutput

from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tfx.components.tuner.component import TunerFnResult
from tensorflow_metadata.proto.v0 import schema_pb2
import keras_tuner
from keras_tuner import HyperParameters
import functools

_LABEL_KEY = 'trip_total'

_TRAIN_BATCH_SIZE = 40 #dataset_size / batch size = # of steps 128
_EVAL_BATCH_SIZE = 20 # 64 

def _fill_in_missing(x):

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(tf.sparse.to_dense(tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),default_value),axis=1)

def transformed_name(key):
    return key + '_xf'

def _make_one_hot(x, key):
    # Number of vocabulary terms used for encoding categorical features.
    _VOCAB_SIZE = 1000
    # Count of out-of-vocab buckets in which unrecognized categorical are hashed.
    _OOV_SIZE = 10
    
    integerized = tft.compute_and_apply_vocabulary(x,top_k=_VOCAB_SIZE,num_oov_buckets=_OOV_SIZE,vocab_filename=key, name=key)
    depth = (tft.experimental.get_vocabulary_size_by_name(key) + _OOV_SIZE)
    one_hot_encoded = tf.one_hot(integerized,
                                 depth=tf.cast(depth, tf.int32),
                                 on_value=1.0,
                                 off_value=0.0)
    
    return tf.reshape(one_hot_encoded, [-1, depth])

def preprocessing_fn(inputs):

    NUMERIC_FEATURE_KEYS = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude','euclidean']

    CATEGORICAL_FEATURE_KEYS = ['month','day']

    LABEL_KEY = 'trip_total'  
  
    ##############################################################
     
    outputs = {}
  
  # Scale numerical features.
    for key in NUMERIC_FEATURE_KEYS:
        outputs[transformed_name(key)] = tft.scale_to_z_score(_fill_in_missing(inputs[key]), name=key)

  # One hot encode the categorical features.
    for key in CATEGORICAL_FEATURE_KEYS:
        outputs[transformed_name(key)] = _make_one_hot(_fill_in_missing(inputs[key]), key)

  # Convert Cover_Type to dense tensor.
    # outputs[transformed_name(LABEL_KEY)] = _fill_in_missing(inputs[LABEL_KEY])
    outputs[LABEL_KEY] = _fill_in_missing(inputs[LABEL_KEY])
    return outputs
