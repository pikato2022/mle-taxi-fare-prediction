# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple and
# slightly modified because we don't need `metadata_path` argument.
from tfx import v1 as tfx
from src.pipeline import config
import kfp #kubeflow pipeline
import tensorflow_model_analysis as tfma

_trainer_module_file = 'trainer.py'


# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple and
# slightly modified because we don't need `metadata_path` argument.

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str, endpoint_name: str, project_id: str, region: str, use_gpu: bool
                     ) -> tfx.dsl.Pipeline:

  # NEW: Configuration for Vertex AI Training.
  # This dictionary will be passed as `CustomJobSpec`.
    vertex_job_spec = {
        'project': project_id,
        'worker_pool_specs': [{
            'machine_spec': {
                'machine_type': 'e2-standard-16',
            },
            'replica_count': 1,
            'container_spec': {
                'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
            },
        }],
    }

    if use_gpu:
        # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
        # for available machine types.
        vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })

        
    """Creates a pipeline with TFX."""
    ########################################
    #01 Brings data into the pipeline.
    ########################################
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)
    
    ########################################
    #02 generate the statistics from the example input 
    ########################################
    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])
    
    ########################################
    #03 generate the schema
    ########################################
    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'],infer_feature_shape=True)
    
    ########################################
    #04 import the schema
    ########################################
    ImportSchemaGen = tfx.components.ImportSchemaGen(schema_file=config.DATA_ROOT[:-30]+'user_area/schema.pbtxt')
    
    #######################################
    # 05 choose the schema 
    schema_choice = ImportSchemaGen.outputs['schema']
    # schema_choice = schema_gen.outputs['schema']
    #######################################
    
    ########################################
    #06 components to validate the examples
    ########################################
    example_validator = tfx.components.ExampleValidator(statistics=statistics_gen.outputs['statistics'],schema=schema_choice)
    
    ########################################
    #07 components to transform the examples
    ########################################
    transform = tfx.components.Transform(examples=example_gen.outputs['examples'],
                                         schema=schema_gen.outputs['schema'],module_file=module_file[:-3]+'_transform.py')
    
    
    ########################################
    #08 Lucky Tuner
    ########################################    
    # tuner component      
    tuner = tfx.components.Tuner(
        module_file=module_file[:-3]+'_tune.py',
        examples=example_gen.outputs['examples'],
        schema=schema_choice,
        train_args=tfx.proto.TrainArgs(num_steps=1600),
        eval_args=tfx.proto.EvalArgs(num_steps=1600),) 
    
    
    # Trains a model using Vertex AI Training.
    # NEW: We need to specify a Trainer for GCP with related configs.
    # trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
    # trainer = tfx.components.Trainer(
    #     module_file=module_file[:-3]+'_transform.py',
    #     examples=transform.outputs['transformed_examples'],
    #     transform_graph=transform.outputs['transform_graph'],
    #     schema=schema_gen.outputs['schema'],
    #     train_args=tfx.proto.TrainArgs(num_steps=1600), #66k/128
    #     eval_args=tfx.proto.EvalArgs(num_steps=1600),) #34k/64
        # custom_config={
        #     tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
        #         True,
        #     tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
        #         region,
        #     tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
        #         vertex_job_spec,
        #     'use_gpu':
        #         use_gpu,
        # })
        
    ########################################
    #08 pseudo custom component modified from standard trainer component for tuning purposes 
    ########################################
    # tuner_custom = tfx.components.Trainer(
    #     module_file=module_file[:-3]+'_tune.py',
    #     examples=example_gen.outputs['examples'],
    #     # transform_graph=transform.outputs['transform_graph'],
    #     schema=schema_gen.outputs['schema'],
    #     train_args=tfx.proto.TrainArgs(num_steps=1600), #66k/128
    #     eval_args=tfx.proto.EvalArgs(num_steps=1600),).with_id('Tuner_Custom') #34k/64
    #     # custom_config={
    #     #     tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
    #     #         True,
    #     #     tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
    #     #         region,
    #     #     tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
    #     #         vertex_job_spec,
    #     #     'use_gpu':
    #     #         use_gpu,
    #     # })  
        
    ########################################
    #09 submit job to vertex training 
    ########################################
    # using the watmstart strategy 
    # https://github.com/tensorflow/tfx/issues/3423
    # trainer_vertex = tfx.components.Trainer(
    trainer_vertex = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=module_file[:-3]+'_vertex.py',
        examples=example_gen.outputs['examples'],
        schema=schema_choice,
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=tfx.proto.TrainArgs(num_steps=1600), #66k/128 1600
        eval_args=tfx.proto.EvalArgs(num_steps=1600), #34k/64 1600
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                vertex_job_spec,
            'use_gpu':
                use_gpu,
        }).with_id('Trainer_Vertex')
    
    ########################################
    # 10 resolver to find the latest blessed model
    # if the latest blessed model doesn not exist, the component will ignore and auto bless current model     
    # NEW: RESOLVER Get the latest blessed model for Evaluator.
    ########################################
    model_resolver = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(
          type=tfx.types.standard_artifacts.ModelBlessing)).with_id(
              'latest_blessed_model_resolver')

    ########################################
    # 11 evaluator component 
    ########################################
    # Eval component      
    accuracy_threshold = tfma.MetricThreshold(
            value_threshold=tfma.GenericValueThreshold(lower_bound={'value':1.0},upper_bound={'value':3.5}),
            change_threshold=tfma.GenericChangeThreshold(absolute={'value':0.6},direction=tfma.MetricDirection.LOWER_IS_BETTER))

    metrics_specs = tfma.MetricsSpec(
                   metrics = [
                       tfma.MetricConfig(class_name='MeanAbsoluteError',
                           threshold=accuracy_threshold),
                       tfma.MetricConfig(class_name='MeanSquaredError')])

    eval_config = tfma.EvalConfig(model_specs=[tfma.ModelSpec(label_key='trip_total')], 
                                  metrics_specs=[metrics_specs], 
                                  slicing_specs=[tfma.SlicingSpec(),
                                                tfma.SlicingSpec(feature_keys=['monday']),
                                                tfma.SlicingSpec(feature_keys=['tuesday']),])
    
    model_analyzer = tfx.components.Evaluator(
        # examples=example_gen.outputs['examples'],
        examples=example_gen.outputs['examples'],
        model=trainer_vertex.outputs['model'],
        eval_config=eval_config,
        baseline_model=model_resolver.outputs['model'],
        )
    
    ########################################
    # 12 Pushes the model to a vertex endpoint 
    ########################################
    # NEW: Configuration for pusher.
    vertex_serving_spec = {
        'project_id': project_id,
        'endpoint_name': endpoint_name,
        # Remaining argument is passed to aiplatform.Model.deploy()
        # See https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api#deploy_the_model
        # for the detail.
        #
        # Machine type is the compute resource to serve prediction requests.
        # See https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types
        # for available machine types and acccerators.
        'machine_type': 'n1-standard-2',
    }
    
    # Vertex AI provides pre-built containers with various configurations for
    # serving.
    # See https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
    # for available container images.
    serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
    if use_gpu:
        vertex_serving_spec.update({
            'accelerator_type': 'NVIDIA_TESLA_K80',
            'accelerator_count': 1
        })
        serving_image = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'
        
    # NEW: Pushes the model to Vertex AI.
    pusher_vertex = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer_vertex.outputs['model'],
        model_blessing=model_analyzer.outputs['blessing'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
                serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
                vertex_serving_spec,
        }).with_id('Pusher Vertex')
    
    ########################################
    # 13 Pushes the model to a filesystem destination.
    ########################################
    pusher_local = tfx.components.Pusher(
        model=trainer_vertex.outputs['model'],
        model_blessing=model_analyzer.outputs['blessing'],
        push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
        # base_directory=serving_model_dir))).with_id('Pusher Local')
        base_directory= 'gs://' + project_id +'/user_area/best_model'))).with_id('Pusher_Local')

    ########################################
    # 14 Select the components you want to activate
    ########################################
    # Following three components will be included in the pipeline.
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        ImportSchemaGen,
        example_validator,
        tuner,
        trainer_vertex,
        model_resolver,
        model_analyzer,
        # pusher_local,
        pusher_vertex,
        
        ## transform,
        ## tuner_custom,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=False)

def compile_pipeline(pl):
    import os
    PIPELINE_DEFINITION_FILE = pl + '_pipeline.json'

    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=PIPELINE_DEFINITION_FILE)
    # Following function will write the pipeline definition to PIPELINE_DEFINITION_FILE.
    _ = runner.run(
        _create_pipeline(
            pipeline_name=pl,
            pipeline_root=config.PIPELINE_ROOT,
            data_root=config.DATA_ROOT,
            module_file=os.path.join(config.MODULE_ROOT, _trainer_module_file),
            endpoint_name=config.ENDPOINT_NAME,
            project_id=config.GOOGLE_CLOUD_PROJECT,
            region=config.GOOGLE_CLOUD_REGION,
            use_gpu=False,
            serving_model_dir=config.SERVING_MODEL_DIR))


def run_pipeline(pl):
    from google.cloud import aiplatform
    from google.cloud.aiplatform import pipeline_jobs
    PIPELINE_DEFINITION_FILE = pl + '_pipeline.json'
    aiplatform.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_REGION)

    job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                    display_name=pl)
    job.run(sync=False)
    return "success"


