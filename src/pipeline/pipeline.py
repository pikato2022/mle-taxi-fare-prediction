# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple and
# slightly modified because we don't need `metadata_path` argument.
from tfx import v1 as tfx
from src.pipeline import config
import kfp #kubeflow pipeline

_trainer_module_file = 'trainer.py'


# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple and
# slightly modified because we don't need `metadata_path` argument.

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str, endpoint_name: str, project_id: str, region: str, use_gpu: bool
                     ) -> tfx.dsl.Pipeline:
    """_summary_

    Args:
        pipeline_name (str): 
        pipeline_root (str):
        data_root (str): 
        module_file (str): 
        serving_model_dir (str):
        endpoint_name (str): 
        project_id (str): 
        region (str):
        use_gpu (bool): 

    Returns:
        tfx.dsl.Pipeline: _description_
    """
  # NEW: Configuration for Vertex AI Training.
  # This dictionary will be passed as `CustomJobSpec`.
    vertex_job_spec = {
        'project': project_id,
        'worker_pool_specs': [{
            'machine_spec': {
                'machine_type': 'n1-standard-4',
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

        
    """Creates a five pipeline with TFX."""
    # Brings data into the pipeline.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    statistics_gen = tfx.components.StatisticsGen(examples=example_gen.outputs['examples'])

    schema_gen = tfx.components.SchemaGen(statistics=statistics_gen.outputs['statistics'],infer_feature_shape=False)

    example_validator = tfx.components.ExampleValidator(statistics=statistics_gen.outputs['statistics'],schema=schema_gen.outputs['schema'])

    # Trains a model using Vertex AI Training.
    # NEW: We need to specify a Trainer for GCP with related configs.
    trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=515), #66k/128
        eval_args=tfx.proto.EvalArgs(num_steps=265), #34k/64
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
                vertex_job_spec,
            'use_gpu':
                use_gpu,
        })

        
    # # Uses user-provided Python function that trains a model.
    # trainer = tfx.components.Trainer(
    #     module_file=module_file,
    #     examples=example_gen.outputs['examples'],
    #     train_args=tfx.proto.TrainArgs(num_steps=1500), #100
    #     eval_args=tfx.proto.EvalArgs(num_steps=1500)) #5
        
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
        'machine_type': 'n1-standard-4',
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
    pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
        model=trainer.outputs['model'],
        custom_config={
            tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
                True,
            tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
                region,
            tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
                serving_image,
            tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
                vertex_serving_spec,
        })

    # # Pushes the model to a filesystem destination.
#    pusher = tfx.components.Pusher(
#         model=trainer.outputs['model'],
#         push_destination=tfx.proto.PushDestination(
#             filesystem=tfx.proto.PushDestination.Filesystem(
#                 base_directory=serving_model_dir)))

    # Following three components will be included in the pipeline.
    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        trainer,
        pusher,
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


