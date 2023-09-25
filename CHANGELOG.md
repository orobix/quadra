
# Changelog
All notable changes to this project will be documented in this file.

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
