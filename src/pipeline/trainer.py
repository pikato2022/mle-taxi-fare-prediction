# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple

from typing import List
from absl import logging
import tensorflow as tf
import tensorflow_transform as tft

from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform import TFTransformOutput
from tfx import v1 as tfx
from tfx_bsl.public import tfxio
from tensorflow_metadata.proto.v0 import schema_pb2


# _FEATURE_KEYS = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'euclidean','month']
_LABEL_KEY = 'trip_total'
# _LABEL_KEY = 'trip_total_xf'

_TRAIN_BATCH_SIZE = 40 #dataset_size / batch size = # of steps 128
_EVAL_BATCH_SIZE = 20 # 64 


# _FEATURE_SPEC = {
#     **{
#         feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
#            for feature in _FEATURE_KEYS
#        },
#     _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
# }

# 1. Input function to read the input to the main function block run_fn
####################################################################
def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              # schema: schema_pb2.Schema,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int) -> tf.data.Dataset:

    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
        # schema=schema).repeat()
        tf_transform_output.transformed_metadata.schema)

# 2. function block to make the ANN
####################################################################
def _make_keras_model(tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    
    # read the inputs to the function 
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    # pop the lavel column    
    feature_spec.pop(_LABEL_KEY)
    
    # define empty inputs     
    inputs = {}
    
    # create the input layer     
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(shape=[None], name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
        # TODO(b/208879020): Move into schema such that spec.shape is [1] and not
        # [] for scalars.
            inputs[key] = tf.keras.layers.Input(shape=spec.shape or [1], name=key, dtype=spec.dtype)  
        else:
            raise ValueError('Spec type is not supported: ', key, spec)
    
    # build the hidden layers     
    output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    output = tf.keras.layers.Dense(100, activation='relu')(output)
    output = tf.keras.layers.Dense(70, activation='relu')(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dense(20, activation='relu')(output)
    output = tf.keras.layers.Dense(1, activation='linear')(output)
    
    # inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    # d = keras.layers.concatenate(inputs)
    # for _ in range(2):
    #     # d = keras.layers.Dense(8, activation='relu')(d)
    #     d = keras.layers.Dense(64, activation='relu')(d)
    #     d = keras.layers.Dense(32, activation='relu')(d)
    #     d = keras.layers.Dense(16, activation='relu')(d)
    #     # outputs = keras.layers.Dense(3)(d)
    #     outputs = keras.layers.Dense(1, activation='linear')(d)

    # model = keras.Model(inputs=inputs, outputs=outputs)

  # model.compile(
  #     optimizer=keras.optimizers.Adam(1e-2),
  #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  #     metrics=[keras.metrics.SparseCategoricalAccuracy()])
    
#     model.compile(
#         optimizer=keras.optimizers.Adam(0.0001),
#         loss=tf.keras.losses.MeanSquaredError(),
#         metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()])

#     model.summary(print_fn=logging.info)
    # return model
    return tf.keras.Model(inputs=inputs, outputs=output)

# 3. function block relating to the training on vertex A.I
####################################################################
# NEW: Read `use_gpu` from the custom_config of the Trainer.
#      if it uses GPU, enable MirroredStrategy.
def _get_distribution_strategy(fn_args: tfx.components.FnArgs):
    if fn_args.custom_config.get('use_gpu', False):
        logging.info('Using MirroredStrategy with one GPU.')
        return tf.distribute.MirroredStrategy(devices=['device:GPU:0'])
    return None

# fb 1 to be used in export_serving_model   
####################################################################
def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(_LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        # TODO(b/154085620): Convert the predicted labels from the model using a
        # reverse-lookup (opposite of transform.py).
        return {'outputs': outputs}

    return serve_tf_examples_fn

# fb 2 to be used in export_serving_model   
####################################################################
def _get_transform_features_signature(model, tf_transform_output):
  # """Returns a serving signature that applies tf.Transform to features."""

  # We need to track the layers in the model in order to save it.
  # TODO(b/162357359): Revise once the bug is resolved.
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn

# function to save the trained model  
####################################################################
def export_serving_model(tf_transform_output, model, output_dir):
    """Exports a keras model for serving.
    Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    model: A keras model to export for serving.
    output_dir: A directory where the model will be exported to.
    """
    # The layer has to be saved to the model for keras tracking purpases.
    model.tft_layer = tf_transform_output.transform_features_layer()

    signatures = {
      'serving_default':
          _get_tf_examples_serving_signature(model, tf_transform_output),
      'transform_features':
          _get_transform_features_signature(model, tf_transform_output),
    }

    model.save(output_dir, save_format='tf', signatures=signatures)


# main function block 
####################################################################
# TFX Trainer  tfx.components.Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):

    # This schema is usually either an output of SchemaGen or a manually-curated
    # version provided by pipeline author. A schema can also derived from TFT
    # graph if a Transform component is used. In the case when either is missing,
    # `schema_from_feature_spec` could be used to generate schema from very simple
    # feature_spec, but the schema returned would be very primitive.
    #Load the schema from Feature Specs
    # schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    
    # wrapper function to get the output of tftranform  
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        # schema,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        # schema,
        tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)
    
    # NEW: If we have a distribution strategy, build a model in a strategy scope.
    # make the model      
    # strategy = _get_distribution_strategy(fn_args)
    # if strategy is None:
    #     model = _make_keras_model(tf_transform_output)
    # else:
    #     with strategy.scope():
    #         model = _make_keras_model(tf_transform_output)
    
    model = _make_keras_model(tf_transform_output)
    # compile the model     
    model.compile(
        optimizer=keras.optimizers.Adam(0.0001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')
    
    model.fit(
        train_dataset,
        epochs = 10,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    # model.save(fn_args.serving_model_dir, save_format='tf')
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)
    

#########################################################################################
#########################################################################################
# For transform component 
#########################################################################################
# import tensorflow_transform as tft
# import tensorflow as tf

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
