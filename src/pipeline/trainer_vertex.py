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
    hp.Int('n_layers',min_value=1, max_value=3, step=1, default=layer)
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
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:

    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
        schema=schema).repeat()

# 2. function block to make the ANN
def _make_keras_model(hparams: HyperParameters, schema:schema_pb2.Schema) -> tf.keras.Model:
    
    feature_keys = [f.name for f in schema.feature if f.name != _LABEL_KEY]
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in feature_keys]
    output = keras.layers.concatenate(inputs) 
    
    print('START BUILD NOW . . .')
    # build the remaining layers based on the hparams
    for n in range(int(hparams.get('n_layers'))):
        print('LAYER:', n)
        print('WITH NEURONS:', hparams.get('n_units_' + str(n + 1)))
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


# main function block 
####################################################################
# TFX Trainer  tfx.components.Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    
    # if else logic here to get the parameter 
    if fn_args.hyperparameters:
        # load user defined params          
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
        print('PLAN A: I HAVE THE BEST PARAMETER FROM TUNER !!!!')
    else:
        # load the default params from the function below          
        hparams = _get_hyperparameters()
        print('PLAN B: I JUST USE THE DEFAULT PARAMETER')
    # log the information     
    logging.info('HyperParameters for training: %s' % hparams.get_config())
    
    schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())
    
    # process the tf.examples into batches 
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=_TRAIN_BATCH_SIZE)
    
    # process the tf.examples into batches
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=_EVAL_BATCH_SIZE)
    
    # enable this like if we are not using vertex A.I    
    # model = _make_keras_model(hparams=hparams, schema=schema)
    
    # enable this like if we are using vertex A.I
    # NEW: If we have a distribution strategy, build a model in a strategy scope.      
    strategy = _get_distribution_strategy(fn_args)
    if strategy is None:
        print('Situation A')
        model = _make_keras_model(hparams=hparams, schema=schema)
        print('MODEL LOADED!!')
    else:
        print('Situation B')
        with strategy.scope():
            model = _make_keras_model(hparams=hparams, schema=schema)
            print('MODEL LOADED!!')
     
    print('Starting training !')
    # https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/tensorboard_profiling_keras.ipynb#scrollTo=OkAo1BanlEeB
    # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard
    # https://www.tensorflow.org/guide/profiler
    # https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=fn_args.model_run_dir, 
                                                          update_freq='batch', 
                                                          histogram_freq = 1,
                                                          write_graph = True,
                                                          write_images = True,)
    model.fit(
        train_dataset,
        epochs = 20,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps, 
        callbacks=[tensorboard_callback])
    
    # evaluate performance of model      
    results = model.evaluate(eval_dataset,batch_size=100,steps=50,return_dict=True,verbose=1)
    # show the results      
    print(results)

    # The result of the training should be saved in `fn_args.serving_model_dir` directory.
    print('Saved Here !!!',fn_args.serving_model_dir)
    model.save(fn_args.serving_model_dir, save_format='tf')
