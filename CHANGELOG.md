
# Changelog
All notable changes to this project will be documented in this file.

### [2.2.3]

#### Updated

- Keep only opencv-python-headless as a dependency for quadra to avoid conflicts with the non-headless version

### [2.2.2]

#### Updated

- Terminate quadra with error if it's not possible to export an ONNX model with automatic mixed precision instead of falling back to full precision

#### Fixed

- Fix default quadra installation requiring extra dependencies incorrectly
- Fix matplotlib using interactive backend by default
- Fix documentation errors

### [2.2.1]

#### Updated

- Update anomalib version, improve release workflow

### [2.2.0]

#### Updated

- Update dependencies to support publishing Quadra to PyPI

### [2.1.13]

#### Updated

- Improve safe batch size computation for sklearn based classification tasks

### [2.1.12]

#### Fixed

- Fix wrong dtype used when evaluating finetuning or anomaly models trained in fp16 precision

### [2.1.11]

#### Fixed

- Fix sklearn automatic batch finder not working properly with ONNX backbones

### [2.1.10]

#### Fixed

- Fix anomaly visualizer callback showing wrong heatmaps after anomaly score refactoring

### [2.1.9]

#### Updated

- Update anomalib to v0.7.0+obx.1.3.3
- Update network builders to support loading model checkpoints from disk

### [2.1.8]

#### Added

- Add onnxconverter-common to the dependencies in order to allow exporting onnx models in mixed precision if issues
are encountered exporting the model entirely in half precision.

### [2.1.7]

#### Fixed

- Fix lightning implementation of batch size finder not working properly when the initial batch size is bigger than the dataset length, now also the code checks that the last iteration works properly.

### [2.1.6]

#### Updated

- Remove poetry dependencies from quadra toml
- Update readme to explain how to use poetry properly

### [2.1.5]

#### Fixed

- Fix classification val_dataloader shuffling data when it shouldn't

### [2.1.4]

#### Updated

- Remove black from pre-commit hooks
- Use ruff as the main formatter and linting tool
- Upgrade mypy version
- Upgrade mlflow version
- Apply new pre-commits to all tests
- Update most of typing to py310 style

### [2.1.3]

#### Updated

- Update anomalib to v0.7.0+obx.1.3.2

### [2.1.2]

#### Updated

- Update anomalib to v0.7.0+obx.1.3.1
- The optimal anomaly threshold is now computed as the average between the max good and min bad score when the F1 is 1

### [2.1.1]

#### Updated

- Anomaly test task now exports results based on the normalized anomaly scores instead of the raw scores. The normalized anomaly scores and the optimal threshold are computed based on the training threshold of the model.

### [2.1.0]

#### Updated

- Change the way anomaly scores are normalized by default, instead of using a [0-1] range with a 0.5 threshold, the scores are now normalized to a [0-1000] range with a threshold of 100, the new score represents the distance from the selected threshold, for example, a score of 200 means that the anomaly score is 100% of the threshold above the threshold itself, a score of 50 means that the anomaly score is 50% of the threshold below. 
- Change the default normalization config name for anomaly from `min_max_normalization` to `score_normalization`.

#### Fixed

- Fix the output heatmaps and preditions of anomaly inference tasks not being saved properly when images belonged to
different classes but had the same name.

### [2.0.4]

#### Fixed 

- Fix segmentation num_data_train sorting

#### Added

- Add default presorting to segmentation samples

### [2.0.3]

#### Fixed 

- Fix anomaly visualizer callback not working properly after lightning upgrade

### [2.0.2]

#### Fixed

- Fix deepcopy removing model signature wrapper from the model for classification

### [2.0.1]

#### Fixed

- Fix pytorch model mlflow upload. Not supported.

### [2.0.0]

#### Updated

- Update torch to 2.1.2 with CUDA 12 support
- Update pytorch lightning to 2.1.*

#### Changed

- Refactor hydra plugins to use optional dev groups intend of extras to avoid dragging local packages around in external installations
- Refactor extra dev dependencies to use poetry groups instead
- Improve trainer configs to avoid wrong overrides when calling different trainer overrides
### [1.5.8]

#### Fix

- Fix ONNX import call for the utilities when ONNX is not installed.

### [1.5.7]

#### Added

- Add upload_models to core config

#### Refactored

- infer_signature_torch_model refactored to infer_signature_model

### [1.5.6]

#### Added

- Add support for half precision training and inference for sklearn based tasks
- Add gradcam export for sklearn training

### [1.5.5]

#### Fixed 

- Fix poetry version not updating init file properly
- Fix model_summary.txt not being saved correctly 

### [1.5.4]

#### Added

- Add test for half precision export

#### Fixed

- Fix half precision export not working properly with onnx iobindings
- Change full precision tolerance to avoid test failures

### [1.5.3]

#### Fixed

- Fix multiclass segmentation analysis report.

### [1.5.2]

#### Fixed

- Fix hydra plugin not working properly when the library is installed from external sources.

### [1.5.1]

#### Fixed

- Fix hydra plugin not working properly.

### [1.5.0]

#### Changed

- Change default build system from `setup.py` to `poetry` for better dependency management.

### [1.4.1]

#### Changed

- Change weights of resnet18 and wideresnet50 to old ones in anomaly model configs

#### Updated

- Update anomalib to [v0.7.0+obx.1.2.9] (added default padim n_features for resnets' old weights)

### [1.4.0]

#### Added

- Add new backbones for classification
- Add parameter to save a model summary for sklearn based classification tasks
- Add results csv file for anomaly detection task
- Add a way to freeze backbone layers by index for the finetuning task

#### Updated

- Update timm requirements to 0.9.12

#### Fixed

- Fix ModelSignatureWrapper not returing the correct instance when cpu, to and half functions are called
- Fix failure in model logging on mlflow whe half precision is used


### [1.3.8]

#### Updated

- Update anomalib to [v0.7.0+obx.1.2.7] (Efficient_ad pre_padding and smaller memory footprint during training)

### [1.3.7]

#### Fixed

- Fix BatchSizeFinder calling wrong super functions
- Fix ModelManager get_latest_version calling an hardcoded model

### [1.3.6]

#### Fixed

- Changed matplotlib backend in anomaly visualizer to solve slowdown on some devices

#### Updated

- Update anomalib to [v0.7.0+obx.1.2.6] (Efficient_ad now keeps maps always on gpu during forward)

### [1.3.5]

#### Fixed

- Anomaly Dataset samples are initially ordered to strngthen reproducibility.

### [1.3.4]

#### Updated

- Update anomalib to [v0.7.0+obx.1.2.5] (logical anomaly is now compatible with trainer.deterministic=True)

### [1.3.3]

#### Updated

- Relax pytest version requirement to 7.x
- Add pytest env variables and pytest-env requirement

### [1.3.2]

#### Updated

- Update `mlflow` requirements for `mlflow-skinny` package to align with the same version of main `mlflow` package.

### [1.3.1]

#### Updated

- Update pandas requirements to use a more recent version and avoid slow build time when python 3.10 is used.

### [1.3.0]

#### Added

- Add batch_size_finder callback for lightning based models (disabled by default).
- Add automatic_batch_size parameter to sklearn based training tasks (disabled by default).
- Add automatic_batch_size decorator to automatically fix the batch size of test functions for evaluation tasks if any out of memory error occurs.
- Add --mock-training flag for tests to skip running the actual training and just run the test.

#### Fixed

- Fix lightning based tasks not working properly when no checkpoint was provided.
- Fix list and dict config not handled properly as input_shapes parameter.

#### Updated 

- Greatly reduce the dimension of test datasets to improve testing speed.

#### Updated

- Make `disable` a quadra reserved keyword for all callbacks, to disable a callback just set it to `disable: true` in the configuration file.

### [1.2.7]

#### Fixed

- Fix test classification task crash when only images with no labels are used.

### [1.2.6]

#### Added

- Add optional `training_threshold_type` for anomaly detection inference task.
#### Changed

- Compute results using the training image threshold instead of zero when running anomaly inference with no labelled data.

### [1.2.5]

#### Fixed

- Fix generic classification experiment crashing due to missing class to index configuration.

### [1.2.4]

#### Added

- Return also probabilities in Classification's module predict step and add them to `self.res`.


### [1.2.3]

#### Fixed

- Fix patch datamodule error when only a single image is available for validation or test.

### [1.2.2]

#### Added

- Add tests for efficient ad model export.
#### Updated

- Update `anomalib` library from version 0.7.0+obx.1.2.0 to 0.7.0+obx.1.2.1
- Update default imagenette dir for efficient ad

### [1.2.1]

#### Added

- Add automatic num_classes computation in Classification Task.

#### Changed 

- Align Classification `test_results` format to the SklearnClassification one (dataframe).

### [1.2.0]

#### Added

- Add plot_raw_outputs feature to class VisualizerCallback in anomaly detection, to save the raw images of the segmentation and heatmap output.
- Add support for onnx exportation of trained models.
- Add support for onnx model import in all evaluation tasks.
- Add `export` configuration group to regulate exportation parameters.
- Add `inference` configuration group to regulate inference parameters.
- Add EfficientAD configuration for anomaly detection.
- Add `acknowledgements` section to `README.md` file.
- Add hashing parameters to datamodule configurations.

#### Updated

- Update anomalib library from version 0.4.0 to 0.7.0
- Update mkdocs library from version 1.4.3 to 1.5.2
- Update mkdocs-material library from version 9.1.18 to 9.2.8
- Update mkdocstrings library by fixing the version to 0.23.0
- Update mkdocs-material-extensions library by fixing the version to 1.1.1
- Update mkdocs-autorefs library by fixing the version to 0.5.0
- Update mkdocs-section-index library from version 0.3.5 to 0.3.6
- Update mkdocstrings-python library from version 1.2.0 to 1.6.2
- Update datamodule documentation for hashing.

#### Changed 

- Move `export_types` parameter from `task` configuration group to `export` configuration group under `types` parameter.
- Refactor export model function to be more generic and be availble from the base task class.
- Remove `save_backbone` parameter for scikit-learn based tasks.

#### Fixed

- Fix failures when trying to override `hydra` configuration groups due to wrong override order.
- Fix certain anomalib models not loaded on the correct device.
- Fix quadra crash when launching an experiment inside a git repository not fully initialized (e.g. without a single commit).
- Fix documentation build failing due to wrong `mkdocstring` version.
- Fix SSL docstrings 
- Fix reference page URL to segmentation page in module management tutorial.
- Fix `Makefile` command.

### [1.1.4]

#### Fixed

- Fix input shape not extracted properly in ModelSignatureWrapper when wrapping around a torchscript model.
### [1.1.3]

#### Fixed

- Fix penultimate patch extracted containing replicated data, now the penultimate patch contain the right data. Fix also the patch size and step computation to avoid generating more than one extra patch.

### [1.1.2]

#### Fixed

- Fix best checkpoint not used for testing when available in `LightningTask` class.

### [1.1.1]

#### Fixed

- Fix deprecated link for tutorials in `README.md` file.

### [1.1.0]

#### Added

- Add ModelManager class to manage model deployments on model tracking platforms (e.g. MLFlow).
- Add automatic storage of file hashes in BaseDataModule class for better experiment reproducibility and tracking.
- Automatically load transforms parameters from model info file in base Evaluation task.
- Add support for vit explainability using Attention Gradient Rollout method.
- Add export of pytorch model for Classification and SklearnClassification tasks.
- Add automatical detection of model input shapes for better exportation capabilities. Add support for custom input shapes like models with multiple inputs.
- Add documentation landing page, improve colore themes and logo.
- Add github actions to automatically build and deploy documentation of main and dev PRs.

#### Changed

- Refactor evaluation task to be more generic and improve inheritance capabilities.
- Refactor export_types parameter to export for better configurability of export parameters.
- Change input_size model info parameter from HXW to a list of actual model parameters for inference (e.g [(3, 224, 224)]).

#### Fixed

- Fix gradcam not working properly with non rectangular images.
- Fix logistic regression wrapper not working properly with 2 classes for torch classification.
- Fix wrong typings in NetworkBuilder's init method.
- Fix broken links in documentation.
- Fix minor documentations issues.

### [1.0.2]

#### Fixed

- Fix anomaly detection training not working when images with non lowercase allowed extension are used (e.g. .BMP).

#### Added

- Increase the number of available extensions for classification and anomaly detection training.

### [1.0.1]

#### Fixed

- Fix training dataset used instead of validation dataset in segmentation datamodules.

### [1.0.0]

#### Added

- All required files for the first release.
