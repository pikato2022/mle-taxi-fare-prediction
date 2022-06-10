# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple and
# slightly modified because we don't need `metadata_path` argument.
from tfx import v1 as tfx
from src.pipeline import config

_trainer_module_file = 'trainer.py'


def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     ) -> tfx.dsl.Pipeline:
    """Creates a three component penguin pipeline with TFX."""
    # Brings data into the pipeline.
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # Uses user-provided Python function that trains a model.
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

    # Pushes the model to a filesystem destination.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    # Following three components will be included in the pipeline.
    components = [
        example_gen,
        trainer,
        pusher,
    ]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True)


def compile_pipeline(pl):
    print(pl)

    import os
    # import config
    PIPELINE_DEFINITION_FILE = pl + '_pipeline.json'

    runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
        config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
        output_filename=PIPELINE_DEFINITION_FILE)
    # Following function will write the pipeline definition to PIPELINE_DEFINITION_FILE.
    return runner.run(
        _create_pipeline(
            pipeline_name=pl,
            pipeline_root=config.PIPELINE_ROOT,
            data_root=config.DATA_ROOT,
            module_file=os.path.join(config.MODULE_ROOT, _trainer_module_file),
            serving_model_dir=config.SERVING_MODEL_DIR))


def run_pipeline(PIPELINE_NAME):
    from google.cloud import aiplatform
    from google.cloud.aiplatform import pipeline_jobs
    PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'
    aiplatform.init(project=config.GOOGLE_CLOUD_PROJECT, location=config.GOOGLE_CLOUD_REGION)

    job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                    display_name=PIPELINE_NAME)
    job.run(sync=False)
    return "success"
