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

# # 0. function block to get hyperparameters 
# ####################################################################
# def _get_hyperparameters(lr=1e-3,layer=3,neu=16) -> keras_tuner.HyperParameters:
#     """Returns hyperparameters for building Keras model."""
#     hp = keras_tuner.HyperParameters()
#     # Defines search space.
#     hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4], default=lr)
#     hp.Int('n_layers', 1, 2, 3, default=layer)
#     with hp.conditional_scope('n_layers', 1):
#         hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=neu)
#     with hp.conditional_scope('n_layers', 2):
#         hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=neu)
#         hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=neu)    
#     with hp.conditional_scope('n_layers', 3):
#         hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=neu)
#         hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=neu)  
#         hp.Int('n_units_3', min_value=8, max_value=128, step=8, default=neu)  

#     return hp

# # 1. Input function to read the input to the main function block run_fn
# ####################################################################
# def _input_fn(file_pattern: List[str],
#               data_accessor: tfx.components.DataAccessor,
#               # schema: schema_pb2.Schema,
#               tf_transform_output: tft.TFTransformOutput,
#               batch_size: int) -> tf.data.Dataset:

#     return data_accessor.tf_dataset_factory(
#         file_pattern,
#         tfxio.TensorFlowDatasetOptions(
#           batch_size=batch_size, label_key=_LABEL_KEY),
#         # schema=schema).repeat()
#         tf_transform_output.transformed_metadata.schema)

# # 2. function block to make the ANN
# ####################################################################
# def _make_keras_model(hparams: HyperParameters,
#                       tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    
#     print('MY FIRST PARAM !')
#     print('LR Jeff!!!!:',hparams.get('learning_rate'))
#     print('layers Jeff!!!!:',hparams.get('n_layers'))
#     for n in range(int(hparams.get('n_layers'))):
#         print('layer',n,'is',hparams.get('n_units_' + str(n + 1)),'neurons')
    
#     # read the inputs to the function 
#     feature_spec = tf_transform_output.transformed_feature_spec().copy()
#     # pop the lavel column    
#     feature_spec.pop(_LABEL_KEY)
    
#     # define empty inputs     
#     inputs = {}
    
#     # create the input layer     
#     for key, spec in feature_spec.items():
#         if isinstance(spec, tf.io.VarLenFeature):
#             inputs[key] = tf.keras.layers.Input(shape=[None], name=key, dtype=spec.dtype, sparse=True)
#         elif isinstance(spec, tf.io.FixedLenFeature):
#         # TODO(b/208879020): Move into schema such that spec.shape is [1] and not
#         # [] for scalars.
#             inputs[key] = tf.keras.layers.Input(shape=spec.shape or [1], name=key, dtype=spec.dtype)  
#         else:
#             raise ValueError('Spec type is not supported: ', key, spec)
    
#     # build the first input layer !     
#     output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    
#     # build the remaining layers based on the hparams
#     for n in range(int(hparams.get('n_layers'))):
#         print('Making layer:',n)
#         print('Number of neurons in this layer:', hparams.get('n_units_' + str(n + 1)))
#         output = tf.keras.layers.Dense(units=hparams.get('n_units_' + str(n + 1)), activation='relu')(output)
    
#     # link to the output layer      
#     output = tf.keras.layers.Dense(1, activation='linear')(output)
    
#     # make model      
#     model = tf.keras.Model(inputs=inputs, outputs=output)
    
#     # compile the model     
#     model.compile(
#         optimizer=keras.optimizers.Adam(0.0001),
#         loss=tf.keras.losses.MeanSquaredError(),
#         metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    
#     # output model summary      
#     print('Please see the model summary below:')
#     model.summary(print_fn=logging.info)
    
#     return model

# # 3. function block relating to the training on vertex A.I
# ####################################################################
# # NEW: Read `use_gpu` from the custom_config of the Trainer.
# #      if it uses GPU, enable MirroredStrategy.
# def _get_distribution_strategy(fn_args: tfx.components.FnArgs):
#     if fn_args.custom_config.get('use_gpu', False):
#         logging.info('Using MirroredStrategy with one GPU.')
#         return tf.distribute.MirroredStrategy(devices=['device:GPU:0'])
#     return None

# # fb 1 to be used in export_serving_model   
# ####################################################################
# def _get_tf_examples_serving_signature(model, tf_transform_output):
#     """Returns a serving signature that accepts `tensorflow.Example`."""

#   # We need to track the layers in the model in order to save it.
#   # TODO(b/162357359): Revise once the bug is resolved.
#     model.tft_layer_inference = tf_transform_output.transform_features_layer()

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    
#     def serve_tf_examples_fn(serialized_tf_example):
#         """Returns the output to be used in the serving signature."""
#         raw_feature_spec = tf_transform_output.raw_feature_spec()
#         # Remove label feature since these will not be present at serving time.
#         raw_feature_spec.pop(_LABEL_KEY)
#         raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
#         transformed_features = model.tft_layer_inference(raw_features)
#         logging.info('serve_transformed_features = %s', transformed_features)

#         outputs = model(transformed_features)
#         # TODO(b/154085620): Convert the predicted labels from the model using a
#         # reverse-lookup (opposite of transform.py).
#         return {'outputs': outputs}

#     return serve_tf_examples_fn

# # fb 2 to be used in export_serving_model   
# ####################################################################
# def _get_transform_features_signature(model, tf_transform_output):
#   # """Returns a serving signature that applies tf.Transform to features."""

#   # We need to track the layers in the model in order to save it.
#   # TODO(b/162357359): Revise once the bug is resolved.
#     model.tft_layer_eval = tf_transform_output.transform_features_layer()

#     @tf.function(input_signature=[
#       tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
#     ])
#     def transform_features_fn(serialized_tf_example):
        
#         """Returns the transformed_features to be fed as input to evaluator."""
#         raw_feature_spec = tf_transform_output.raw_feature_spec()
#         raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
#         transformed_features = model.tft_layer_eval(raw_features)
#         logging.info('eval_transformed_features = %s', transformed_features)
#         return transformed_features

#     return transform_features_fn

# # function to save the trained model  
# ####################################################################
# def export_serving_model(tf_transform_output, model, output_dir):
#     """Exports a keras model for serving.
#     Args:
#     tf_transform_output: Wrapper around output of tf.Transform.
#     model: A keras model to export for serving.
#     output_dir: A directory where the model will be exported to.
#     """
#     # The layer has to be saved to the model for keras tracking purpases.
#     model.tft_layer = tf_transform_output.transform_features_layer()

#     signatures = {
#       'serving_default':
#           _get_tf_examples_serving_signature(model, tf_transform_output),
#       'transform_features':
#           _get_transform_features_signature(model, tf_transform_output),
#     }

#     model.save(output_dir, save_format='tf', signatures=signatures)


# # main function block 
# ####################################################################
# # TFX Trainer  tfx.components.Trainer will call this function.
# def run_fn(fn_args: tfx.components.FnArgs):

#     # This schema is usually either an output of SchemaGen or a manually-curated
#     # version provided by pipeline author. A schema can also derived from TFT
#     # graph if a Transform component is used. In the case when either is missing,
#     # `schema_from_feature_spec` could be used to generate schema from very simple
#     # feature_spec, but the schema returned would be very primitive.
#     #Load the schema from Feature Specs
#     # schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    
#     # if else logic here to get the parameter 
#     if fn_args.hyperparameters:
#         # load user defined params          
#         hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
#         print('PLAN A')
#     else:
#         # load the default params from the function below          
#         hparams = _get_hyperparameters(lr=1e-3,layer=3,neu=16)
#         print('PLAN B')
#     # log the information     
#     print('HEY LOOK HERE !!!')
#     logging.info('HyperParameters for training: %s' % hparams.get_config())
    
#     # wrapper function to get the output of tftranform 
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
#     # process the tf.examples into batches 
#     train_dataset = _input_fn(
#         fn_args.train_files,
#         fn_args.data_accessor,
#         # schema,
#         tf_transform_output,
#         batch_size=_TRAIN_BATCH_SIZE)
    
#     # process the tf.examples into batches
#     eval_dataset = _input_fn(
#         fn_args.eval_files,
#         fn_args.data_accessor,
#         # schema,
#         tf_transform_output,
#         batch_size=_EVAL_BATCH_SIZE)
    
#     # NEW: If we have a distribution strategy, build a model in a strategy scope.
#     # make the model      
#     # strategy = _get_distribution_strategy(fn_args)
#     # if strategy is None:
#     #     model = _make_keras_model(tf_transform_output)
#     # else:
#     #     with strategy.scope():
#     #         model = _make_keras_model(tf_transform_output)
    
#     model = _make_keras_model(hparams=hparams, tf_transform_output=tf_transform_output)

#     tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')
    
#     model.fit(
#         train_dataset,
#         epochs = 10,
#         steps_per_epoch=fn_args.train_steps,
#         validation_data=eval_dataset,
#         validation_steps=fn_args.eval_steps,
#         callbacks=[tensorboard_callback])
    
#     # evaluate performance of model      
#     results = model.evaluate(eval_dataset,batch_size=100,steps=50,return_dict=True,verbose=1)
#     # show the results      
#     print(results)

#     # The result of the training should be saved in `fn_args.serving_model_dir`
#     # directory.
#     # model.save(fn_args.serving_model_dir, save_format='tf')
#     print('Saved Here !!!',fn_args.serving_model_dir)
#     export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)

# # tuner function block 
# ####################################################################    
# def tuner_fn(fn_args: tfx.components.FnArgs) -> TunerFnResult:
    
# #########################################################################################    
    
#     # wrapper function to get the output of tftranform 
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
#     # Construct a build_keras_model_fn that just takes hyperparams from get_hyperparameters as input.
#     build_keras_model_fn = functools.partial(_make_keras_model, tf_transform_output=tf_transform_output) 
    
#     # BayesianOptimization is a subclass of kerastuner.Tuner which inherits from BaseTuner.    
#     tuner = keras_tuner.BayesianOptimization(
#         build_keras_model_fn,
#         max_trials=10,
#         hyperparameters=_get_hyperparameters(),
#         allow_new_entries=False,
#         tune_new_entries=False,
#         objective=keras_tuner.Objective('mean_absolute_error', 'min'),
#         directory=fn_args.working_dir,
#         # directory='gs://licheng-test-06/pipeline_broot/chicago-vertex-pipelines/working-directory',
#         project_name='covertype_tuning')
    
#     train_dataset = _input_fn(
#       fn_args.train_files,
#       fn_args.data_accessor,
#       tf_transform_output,
#       batch_size=_TRAIN_BATCH_SIZE)

#     eval_dataset = _input_fn(
#       fn_args.eval_files,
#       fn_args.data_accessor,
#       tf_transform_output,
#       batch_size=_EVAL_BATCH_SIZE)
    
#     return TunerFnResult(
#       tuner=tuner,
#       fit_kwargs={
#           'x': train_dataset,
#           'validation_data': eval_dataset,
#           'steps_per_epoch': fn_args.train_steps,
#           'validation_steps': fn_args.eval_steps
#       })
    
#########################################################################################

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
