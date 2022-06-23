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
from tfx.utils import io_utils
from tensorflow_metadata.proto.v0 import schema_pb2
import keras_tuner
from keras_tuner import HyperParameters
import functools

# from tfx.utils import io_utils
# from tensorflow_metadata.proto.v0 import schema_pb2

# _FEATURE_KEYS = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'euclidean', 'monday', 
#                  'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

# _DATA_TYPE = {'pickup_latitude':tf.float32,
#              'pickup_longitude':tf.float32,
#              'dropoff_latitude':tf.float32,
#              'dropoff_longitude':tf.float32,
#              'euclidean':tf.float32,
#              'monday':tf.int64,
#              'tuesday':tf.int64,
#              'wednesday':tf.int64,
#              'thursday':tf.int64,
#              'friday':tf.int64,
#               'saturday':tf.int64,
#               'sunday':tf.int64,
#              }

_LABEL_KEY = 'trip_total'

# _FEATURE_SPEC = {
#     **{
#         feature: tf.io.FixedLenFeature(shape=[1], dtype=_DATA_TYPE[feature])
#            for feature in _FEATURE_KEYS
#        },
#     _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
# }

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
              # tf_transform_output: tft.TFTransformOutput,
              batch_size: int) -> tf.data.Dataset:

    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
            schema=schema).repeat()


# 2. function block to make the ANN

def _make_keras_model(hparams: HyperParameters, schema:schema_pb2.Schema) -> tf.keras.Model:
                      # tf_transform_output: TFTransformOutput) -> tf.keras.Model:

    # build the first input layer !
    # inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    
    feature_keys = [f.name for f in schema.feature if f.name != _LABEL_KEY]
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in feature_keys]
    output = keras.layers.concatenate(inputs) 
    
    # print('we use:',hparams.get('learning_rate'),hparams.get('n_layers'), hparams.get('n_units_1'))
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
        print('PLAN A')
    else:
        # load the default params from the function below          
        hparams = _get_hyperparameters()
        print('PLAN B')
    # log the information     
    logging.info('HyperParameters for training: %s' % hparams.get_config())
    
    # # wrapper function to get the output of tftranform 
    # tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    
    # schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)
    schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())
    
    # schema = io_utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())
    
    # process the tf.examples into batches 
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        # fn_args.schema_path,
        schema,
        # tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    
    # process the tf.examples into batches
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        # fn_args.schema_path,
        schema,
        # tf_transform_output,
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
    NEU = [16]

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
                model = _make_keras_model(hparams=_get_hyperparameters(lr=lr,layer=layer,neu=neu), schema=schema)
                model.fit(
                    train_dataset,
                    epochs = 2,
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
    model = _make_keras_model(hparams=_get_hyperparameters(lr=best_lr,layer=best_layer,neu=best_neu), schema=schema)
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
    print('Saved Here !!!',fn_args.serving_model_dir)
    model.save(fn_args.serving_model_dir, save_format='tf')
    
#############################################################
#just testing out the tuner function for fun !
#############################################################

def tuner_fn(fn_args: tfx.components.FnArgs) -> TunerFnResult:
    
    schema = tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema_pb2.Schema())
    
    build_keras_model_fn = functools.partial(_make_keras_model, schema=schema)
    
    # BayesianOptimization is a subclass of kerastuner.Tuner which inherits from BaseTuner.    
    tuner = keras_tuner.RandomSearch(
        build_keras_model_fn,
        max_trials=5,
        hyperparameters=_get_hyperparameters(),
      # New entries allowed for n_units hyperparameter construction conditional on n_layers selected.
#       allow_new_entries=True,
#       tune_new_entries=True,
        objective=keras_tuner.Objective('mean_absolute_error', 'min'),
        directory=fn_args.working_dir,
        project_name='covertype_tuning')
    
    # process the tf.examples into batches 
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        # fn_args.schema_path,
        schema,
        # tf_transform_output,
        batch_size=_TRAIN_BATCH_SIZE)
    
    # process the tf.examples into batches
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        # fn_args.schema_path,
        schema,
        # tf_transform_output,
        batch_size=_EVAL_BATCH_SIZE)
    
    return TunerFnResult(
    tuner=tuner,
    fit_kwargs={
        'x': train_dataset,
        'validation_data': eval_dataset,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps
      })