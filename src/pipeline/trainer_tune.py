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

# 0. function block to get hyperparameters 

def _get_hyperparameters(lr=1e-3,layer=3,neu=16) -> keras_tuner.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hp = keras_tuner.HyperParameters()
    # Defines search space.
    hp.Choice('learning_rate', [1e-1, 1e-2, 1e-3, 1e-4], default=lr)
    hp.Int('n_layers', 1, 2, 3, default=layer)
    with hp.conditional_scope('n_layers', 1):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=neu)
    with hp.conditional_scope('n_layers', 2):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=neu)
        hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=neu)    
    with hp.conditional_scope('n_layers', 3):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=neu)
        hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=neu)  
        hp.Int('n_units_3', min_value=8, max_value=128, step=8, default=neu)  

    return hp

# 1. Input function to read the input to the main function block run_fn

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

def _make_keras_model(hparams: HyperParameters,
                      tf_transform_output: TFTransformOutput) -> tf.keras.Model:
    
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
            inputs[key] = tf.keras.layers.Input(shape=spec.shape or [1], name=key, dtype=spec.dtype)  
        else:
            raise ValueError('Spec type is not supported: ', key, spec)
    
    # build the first input layer !     
    output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    print('we use:',hparams.get('learning_rate'),hparams.get('n_layers'), hparams.get('n_units_1'))
    
    # build the remaining layers based on the hparams
    for n in range(int(hparams.get('n_layers'))):
        output = tf.keras.layers.Dense(units=hparams.get('n_units_' + str(n + 1)), activation='relu')(output)
    
    # link to the output layer      
    output = tf.keras.layers.Dense(1, activation='linear')(output)
    
    # make model      
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    # compile the model     
    model.compile(
        optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError(), keras.metrics.MeanAbsoluteError()])
    
    # output model summary      
    # print('Please see the model summary below:')
    # model.summary(print_fn=logging.info)
    
    return model

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
    
    # if else logic here to get the parameter 
    if fn_args.hyperparameters:
        # load user defined params          
        hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
        print('PLAN A')
    else:
        # load the default params from the function below          
        hparams = _get_hyperparameters()
        print('PLAN B')
    # log the information     
    logging.info('HyperParameters for training: %s' % hparams.get_config())
    
    # wrapper function to get the output of tftranform 
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # process the tf.examples into batches 
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        # schema,
        tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    
    # process the tf.examples into batches
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
    
    # define search space      
    LR = [0.001]
    LAYER = [3]
    NEU = [8,16]

    mae = 1000.0 
    ROUND_ID = 100000

    best_lr = 0.0
    best_layer = 0
    best_neu = 0 
    
    ROUND = 0 
    print('COMMENCE TUNING PROCESS ! ')    
    for lr in LR: 
        for layer in LAYER:
            for neu in NEU: 
                ROUND = ROUND + 1
                print('STARTING TUNING ROUND:',ROUND,)
                print('WITH','Learning Rate:',lr,'Layers:',layer,'Neurons Per Layer:',neu)
                model = _make_keras_model(hparams=_get_hyperparameters(lr=lr,layer=layer,neu=neu), tf_transform_output=tf_transform_output)
                model.fit(
                    train_dataset,
                    epochs = 5,
                    steps_per_epoch=fn_args.train_steps,
                    validation_data=eval_dataset,
                    validation_steps=fn_args.eval_steps,)
                results = model.evaluate(eval_dataset,batch_size=100,steps=50,return_dict=True,verbose=1)
                print('End of round',ROUND,'here are the results:')
                print(results)
                if results['mean_absolute_error'] < mae: 
                        best_lr = lr
                        best_layer = layer
                        best_neu = neu
                        mae = results['mean_absolute_error']
                        ROUND_ID = ROUND
                        print('Best result thus far:','Round',ROUND_ID,'mae', mae)
                        print('WITH','Learning Rate:',best_lr,'Layers:',best_layer,'Neurons Per Layer:',best_neu)
                else: 
                    print('THIS ROUND HAS BEEN DISCARDED')
                    print('Best result thus far:','Round',ROUND_ID,'mae', mae)
                    print('WITH','Learning Rate:',best_lr,'Layers:',best_layer,'Neurons Per Layer:',best_neu)
     
    print('Starting final training !')
    model = _make_keras_model(hparams=_get_hyperparameters(lr=best_lr,layer=best_layer,neu=best_neu), tf_transform_output=tf_transform_output)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, update_freq='batch')
    model.fit(
        train_dataset,
        epochs = 1,
        # steps_per_epoch=fn_args.train_steps,
        steps_per_epoch=2, #only use 2 step, leave the rest to vertex
        validation_data=eval_dataset,
        # validation_steps=fn_args.eval_steps, 
        validation_steps=2, #only use 2 step, leave the rest to vertex
        callbacks=[tensorboard_callback])
    
    # evaluate performance of model      
    results = model.evaluate(eval_dataset,batch_size=100,steps=50,return_dict=True,verbose=1)
    # show the results      
    print(results)

    # The result of the training should be saved in `fn_args.serving_model_dir` directory.
    # model.save(fn_args.serving_model_dir, save_format='tf')
    print('Saved Here !!!',fn_args.serving_model_dir)
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)
    